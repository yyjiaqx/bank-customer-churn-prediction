# -*- coding: utf-8 -*-
"""
电信客户流失预测 — 模型训练（Logistic Regression）
使用 GridSearchCV 训练 LR，提取系数并提供可解释性分析。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

BLUE = (0.122, 0.467, 0.706)
ORANGE = (1.000, 0.498, 0.055)


def train_logistic_regression(X_train, y_train, random_state=42):
    """使用 GridSearchCV 训练 Logistic Regression。

    参数
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    random_state : int

    返回
    -------
    model : LogisticRegression（最佳估计器）
    best_params : dict
    """
    print("\n" + "=" * 60)
    print("  模型训练: Logistic Regression")
    print("=" * 60)

    # 超参数搜索空间：正则化强度 C
    params = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'max_iter': [2000],
        'solver': ['lbfgs'],
    }
    lr = LogisticRegression(random_state=random_state, max_iter=2000)
    # 5 折分层交叉验证，以 ROC-AUC 为评分指标
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(lr, params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"\n  最佳参数:      {grid.best_params_}")
    print(f"  最佳 CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


def get_coefficients(model, feature_names):
    """提取并排序 Logistic Regression 系数。

    参数
    ----------
    model : LogisticRegression
    feature_names : list of str

    返回
    -------
    pd.DataFrame
        列: ['Feature', 'Coefficient', 'Odds_Ratio', 'Abs_Coefficient']
        按系数绝对值降序排列。
    """
    coef = model.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef,
        'Odds_Ratio': np.exp(coef),       # 胜率比 = e^系数
        'Abs_Coefficient': np.abs(coef),
    }).sort_values('Abs_Coefficient', ascending=False).reset_index(drop=True)

    return coef_df


def plot_coefficients(coef_df, output_dir="outputs/figures", top_n=15):
    """绘制特征系数水平条形图。

    参数
    ----------
    coef_df : pd.DataFrame
    output_dir : str
    top_n : int
        显示排名前 N 的特征。
    """
    top = coef_df.head(top_n).iloc[::-1]  # 反转顺序（最大值在上）

    fig, ax = plt.subplots(figsize=(10, 6))
    # 正系数橙色（增加流失风险），负系数蓝色（降低流失风险）
    colors = [ORANGE if c > 0 else BLUE for c in top['Coefficient'].values]
    bars = ax.barh(range(len(top)), top['Coefficient'].values, color=colors)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['Feature'].values, fontsize=10)
    ax.set_xlabel('系数 (对数胜率)', fontsize=12)
    ax.set_title('Logistic Regression — 特征系数排名 (Top 15)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)

    # 在条形旁标注系数值
    for bar, val in zip(bars, top['Coefficient'].values):
        label_pos = bar.get_width() + 0.02 if val >= 0 else bar.get_width() - 0.1
        ax.text(label_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    # 图例
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
    """打印 Top 10 系数的业务解读。"""
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
