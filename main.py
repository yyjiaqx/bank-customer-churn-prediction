# -*- coding: utf-8 -*-
"""
电信客户流失预测 — 主入口
运行方式: python main.py

流程:
  1. 加载数据 + EDA
  2. 数据预处理（编码、标准化、划分）
  3. SMOTE 过采样（处理类别不平衡）
  4. 多模型训练（Logistic Regression / Random Forest / XGBoost）
  5. 多模型评估与对比 + SHAP 可解释性分析
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data, run_eda
from src.preprocessing import preprocess, apply_smote
from src.model import (
    train_logistic_regression, train_all_models,
    get_coefficients, plot_coefficients, print_coefficient_insights,
    get_feature_importance, plot_feature_importance,
    shap_analysis
)
from src.evaluation import (
    evaluate_model, evaluate_all_models,
    plot_confusion_matrix, plot_all_confusion_matrices,
    plot_roc_curve, plot_roc_curves_all,
    plot_pr_curves_all, plot_model_comparison
)
from sklearn.metrics import confusion_matrix


def main():
    DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    OUTPUT_DIR = "outputs/figures"
    RANDOM_STATE = 42
    USE_SMOTE = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============================================================
    # 步骤 1：加载数据 + EDA
    # ============================================================
    print("\n" + "#" * 60)
    print("#  步骤 1: 数据加载与 EDA")
    print("#" * 60)
    df = load_data(DATA_PATH)
    run_eda(df, OUTPUT_DIR)

    # ============================================================
    # 步骤 2：预处理
    # ============================================================
    print("\n" + "#" * 60)
    print("#  步骤 2: 数据预处理")
    print("#" * 60)
    X_train, X_test, y_train, y_test, feature_names = preprocess(df, random_state=RANDOM_STATE)
    print(f"\n[信息] 最终特征集 ({len(feature_names)} 个):")
    for i, name in enumerate(feature_names):
        print(f"  {i:2d}. {name}")

    # ============================================================
    # 步骤 3：SMOTE 过采样
    # ============================================================
    if USE_SMOTE:
        print("\n" + "#" * 60)
        print("#  步骤 3: SMOTE 过采样")
        print("#" * 60)
        X_train, y_train = apply_smote(X_train, y_train, RANDOM_STATE)

    # ============================================================
    # 步骤 4：多模型训练
    # ============================================================
    print("\n" + "#" * 60)
    print("#  步骤 4: 多模型训练")
    print("#" * 60)
    models = train_all_models(X_train, y_train, RANDOM_STATE)

    # Logistic Regression 系数分析
    lr_model = models['Logistic Regression'][0]
    coef_df = get_coefficients(lr_model, feature_names)
    print_coefficient_insights(coef_df)
    plot_coefficients(coef_df, OUTPUT_DIR)

    # ============================================================
    # 步骤 5：多模型评估与对比
    # ============================================================
    print("\n" + "#" * 60)
    print("#  步骤 5: 多模型评估与对比")
    print("#" * 60)

    results_df = evaluate_all_models(models, X_test, y_test)
    print("\n" + "=" * 60)
    print("  模型指标对比总表")
    print("=" * 60)
    print(results_df.to_string())

    plot_all_confusion_matrices(models, X_test, y_test, OUTPUT_DIR)
    plot_roc_curves_all(models, X_test, y_test, OUTPUT_DIR)
    plot_pr_curves_all(models, X_test, y_test, OUTPUT_DIR)
    plot_model_comparison(results_df, OUTPUT_DIR)

    # 特征重要性
    print("\n" + "=" * 60)
    print("  特征重要性分析")
    print("=" * 60)
    for model_name in ['Random Forest', 'XGBoost']:
        if model_name in models:
            fi_df = get_feature_importance(models[model_name][0], feature_names, model_name)
            if fi_df is not None:
                print(f"\n  {model_name} — Top 10 重要特征:")
                print(fi_df.head(10).to_string())
                plot_feature_importance(fi_df, model_name, OUTPUT_DIR)

    # ============================================================
    # 步骤 6：SHAP 可解释性分析
    # ============================================================
    print("\n" + "#" * 60)
    print("#  步骤 6: SHAP 可解释性分析")
    print("#" * 60)

    ranked = results_df.sort_values('ROC-AUC', ascending=False)
    best_model_name = ranked.index[0]

    shap_analysis(models['Logistic Regression'][0], X_train, X_test,
                  feature_names, 'Logistic Regression', OUTPUT_DIR)
    shap_analysis(models[best_model_name][0], X_train, X_test,
                  feature_names, best_model_name, OUTPUT_DIR)

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("  管线执行完毕 — 最终对比排名")
    print("=" * 60)

    print(ranked[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string())

    best_auc = ranked.iloc[0]['ROC-AUC']
    print(f"\n  [Best] 最佳模型: {best_model_name} (ROC-AUC = {best_auc:.4f})")
    print(f"\n  图表和结果已保存至: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
