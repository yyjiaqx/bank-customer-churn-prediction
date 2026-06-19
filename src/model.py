# -*- coding: utf-8 -*-
"""
电信客户流失预测 — 模型训练
支持 Logistic Regression、Random Forest、XGBoost 三种模型，
含 GridSearchCV 调参、系数分析、特征重要性分析。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb

BLUE = (0.122, 0.467, 0.706)
ORANGE = (1.000, 0.498, 0.055)
GREEN = (0.172, 0.627, 0.172)


# ============================================================
#  Logistic Regression
# ============================================================

def train_logistic_regression(X_train, y_train, random_state=42):
    print("\n" + "=" * 60)
    print("  模型训练: Logistic Regression")
    print("=" * 60)

    params = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'max_iter': [2000],
        'solver': ['lbfgs'],
    }
    lr = LogisticRegression(random_state=random_state, max_iter=2000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(lr, params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"\n  最佳参数:      {grid.best_params_}")
    print(f"  最佳 CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


# ============================================================
#  Random Forest
# ============================================================

def train_random_forest(X_train, y_train, random_state=42):
    print("\n" + "=" * 60)
    print("  模型训练: Random Forest")
    print("=" * 60)

    params = {
        'n_estimators': [100, 200],
        'max_depth': [8, 12, None],
        'min_samples_split': [5, 10],
        'class_weight': ['balanced', None],
    }
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(rf, params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"\n  最佳参数:      {grid.best_params_}")
    print(f"  最佳 CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


# ============================================================
#  XGBoost
# ============================================================

def train_xgboost(X_train, y_train, random_state=42):
    print("\n" + "=" * 60)
    print("  模型训练: XGBoost")
    print("=" * 60)

    scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

    params = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
    }
    xgb_model = xgb.XGBClassifier(
        random_state=random_state, n_jobs=-1,
        use_label_encoder=False, eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(xgb_model, params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"\n  最佳参数:      {grid.best_params_}")
    print(f"  最佳 CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


# ============================================================
#  统一训练入口
# ============================================================

def train_all_models(X_train, y_train, random_state=42):
    results = {}

    lr_model, lr_params = train_logistic_regression(X_train, y_train, random_state)
    results['Logistic Regression'] = (lr_model, lr_params)

    rf_model, rf_params = train_random_forest(X_train, y_train, random_state)
    results['Random Forest'] = (rf_model, rf_params)

    xgb_model, xgb_params = train_xgboost(X_train, y_train, random_state)
    results['XGBoost'] = (xgb_model, xgb_params)

    return results


# ============================================================
#  Logistic Regression 系数分析
# ============================================================

def get_coefficients(model, feature_names):
    coef = model.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef,
        'Odds_Ratio': np.exp(coef),
        'Abs_Coefficient': np.abs(coef),
    }).sort_values('Abs_Coefficient', ascending=False).reset_index(drop=True)
    return coef_df


def plot_coefficients(coef_df, output_dir="outputs/figures", top_n=15):
    top = coef_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [ORANGE if c > 0 else BLUE for c in top['Coefficient'].values]
    bars = ax.barh(range(len(top)), top['Coefficient'].values, color=colors)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['Feature'].values, fontsize=10)
    ax.set_xlabel('系数 (对数胜率)', fontsize=12)
    ax.set_title('Logistic Regression — 特征系数排名 (Top {:d})'.format(top_n), fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)

    for bar, val in zip(bars, top['Coefficient'].values):
        label_pos = bar.get_width() + 0.02 if val >= 0 else bar.get_width() - 0.1
        ax.text(label_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ORANGE, label='正系数 (增加流失风险)'),
        Patch(facecolor=BLUE, label='负系数 (降低流失风险)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'coefficients.png')
    plt.savefig(path)
    plt.close()
    print(f"  [已保存] {path}")


def print_coefficient_insights(coef_df):
    print("\n" + "=" * 60)
    print("  特征系数解读 (Top 10)")
    print("=" * 60)
    print(f"{'排名':<5} {'特征':<30} {'系数':>8} {'胜率比':>10} {'方向':>12}")
    print("-" * 70)

    for i, row in coef_df.head(10).iterrows():
        direction = "↑ 增加流失" if row['Coefficient'] > 0 else "↓ 降低流失"
        print(f"{i+1:<5} {row['Feature']:<30} {row['Coefficient']:>8.4f} {row['Odds_Ratio']:>10.4f} {direction:>12}")

    print("\n  解读说明:")
    print("  — 正系数 → 该特征值越大，客户流失概率越高")
    print("  — 负系数 → 该特征值越大，客户流失概率越低")
    print("  — 胜率比 > 1 → 特征每增加 1 标准差，流失胜率乘以该值")
    print("  — 胜率比 < 1 → 特征每增加 1 标准差，流失胜率除以该值的倒数")


# ============================================================
#  特征重要性分析（树模型）
# ============================================================

def get_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print(f"  [警告] {model_name} 不支持特征重要性提取")
        return None

    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    return fi_df


def plot_feature_importance(fi_df, model_name, output_dir="outputs/figures", top_n=20):
    if fi_df is None:
        return
    top = fi_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top)), top['Importance'].values, color=BLUE)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['Feature'].values, fontsize=10)
    ax.set_xlabel('特征重要性', fontsize=12)
    ax.set_title(f'{model_name} — 特征重要性 (Top {top_n})', fontsize=14, fontweight='bold')

    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row['Importance'] + 0.002, i, f"{row['Importance']:.4f}", va='center', fontsize=9)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"  [已保存] {path}")

# ============================================================
#  SHAP 可解释性分析
# ============================================================

def shap_analysis(model, X_train, X_test, feature_names, model_name, output_dir="outputs/figures", max_display=20):
    """对模型进行 SHAP 可解释性分析：Summary Plot、Waterfall、
    Dependence Plot。

    参数
    ----------
    model : sklearn/xgboost 模型
    X_train : np.ndarray  — 用于构建 explainer
    X_test : np.ndarray   — 用于解释预测
    feature_names : list of str
    model_name : str
    output_dir : str
    max_display : int
        图中显示的最多特征数。
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        import shap
    except ImportError:
        print("  [警告] 未安装 shap，跳过 SHAP 分析。pip install shap")
        return

    print(f"\n  SHAP 可解释性分析 — {model_name}")
    print("  " + "-" * 50)

    # 用训练集的抽样构建 explainer（节省内存）
    sample_size = min(500, X_train.shape[0])
    X_train_sample = X_train[:sample_size]
    X_test_sample = X_test[:min(500, X_test.shape[0])]

    # 根据模型类型选择 explainer
    if hasattr(model, 'predict_proba') and hasattr(model, 'coef_'):
        # Logistic Regression — LinearExplainer
        explainer = shap.LinearExplainer(model, X_train_sample)
        shap_values = explainer.shap_values(X_test_sample)
        # LinearExplainer 返回的 shap_values 可能 shape 为 (n, n_features)
        # 如果不是 list，那就是单输出
    elif hasattr(model, 'feature_importances_'):
        # 树模型 — TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sample)
        # XGBoost 有时返回 list [neg, pos]，取正类
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
    else:
        print(f"  [警告] {model_name} 不支持 SHAP 分析")
        return

    # 确保 shap_values 是 2D
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    safe_name = model_name.lower().replace(' ', '_')

    # 图 1: Summary Plot (蜂群图)
    print("  生成 Summary Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    path = os.path.join(output_dir, f'shap_summary_{safe_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [已保存] {path}")

    # 图 2: Summary Bar Plot (特征重要性排序)
    print("  生成 Summary Bar Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                      plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()
    path = os.path.join(output_dir, f'shap_bar_{safe_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [已保存] {path}")

    # 图 3: Waterfall Plot (解释第一个测试样本的预测)
    print("  生成 Waterfall Plot...")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(values=shap_values[0],
                             base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                                          else explainer.expected_value[1],
                             data=X_test_sample[0],
                             feature_names=feature_names),
            max_display=15, show=False
        )
        plt.tight_layout()
        path = os.path.join(output_dir, f'shap_waterfall_{safe_name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [已保存] {path}")
    except Exception as e:
        print(f"  [提示] Waterfall 图跳过: {e}")

    # 图 4: Dependence Plot (对最重要的特征)
    print("  生成 Dependence Plot...")
    try:
        # 找到最重要的特征索引
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = int(np.argmax(mean_abs_shap))

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(top_idx, shap_values, X_test_sample,
                             feature_names=feature_names, show=False)
        plt.tight_layout()
        path = os.path.join(output_dir, f'shap_dependence_{safe_name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [已保存] {path}")
    except Exception as e:
        print(f"  [提示] Dependence 图跳过: {e}")

    print(f"  SHAP 分析完成 — {model_name}")
