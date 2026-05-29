# -*- coding: utf-8 -*-
"""
电信客户流失预测 — 主入口
运行方式: python main.py

流程:
  1. 加载数据
  2. EDA（统计摘要 + 图表生成）
  3. 预处理（编码、标准化、划分）
  4. 模型训练（Logistic Regression + GridSearchCV）
  5. 评估（指标、混淆矩阵、ROC 曲线、系数分析）
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data, run_eda
from src.preprocessing import preprocess
from src.model import (
    train_logistic_regression, get_coefficients,
    plot_coefficients, print_coefficient_insights
)
from src.evaluation import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve
)
from sklearn.metrics import confusion_matrix


def main():
    # 路径配置
    DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    OUTPUT_DIR = "outputs/figures"
    RANDOM_STATE = 42

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
    # 步骤 3：模型训练
    # ============================================================
    print("\n" + "#" * 60)
    print("#  步骤 3: 模型训练 (Logistic Regression)")
    print("#" * 60)
    model, best_params = train_logistic_regression(X_train, y_train, RANDOM_STATE)

    # 系数分析与可视化
    coef_df = get_coefficients(model, feature_names)
    print_coefficient_insights(coef_df)
    plot_coefficients(coef_df, OUTPUT_DIR)

    # ============================================================
    # 步骤 4：模型评估
    # ============================================================
    print("\n" + "#" * 60)
    print("#  步骤 4: 模型评估")
    print("#" * 60)
    metrics = evaluate_model(model, X_test, y_test, OUTPUT_DIR)
    cm = confusion_matrix(y_test, model.predict(X_test))
    plot_confusion_matrix(cm, "Logistic Regression", OUTPUT_DIR)
    plot_roc_curve(model, X_test, y_test, OUTPUT_DIR)

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("  管线执行完毕")
    print("=" * 60)
    print(f"  模型:     Logistic Regression")
    print(f"  配置:     {best_params}")
    print(f"  准确率:   {metrics['Accuracy']:.4f}")
    print(f"  精确率:   {metrics['Precision']:.4f}")
    print(f"  召回率:   {metrics['Recall']:.4f}")
    print(f"  F1 分数:  {metrics['F1-Score']:.4f}")
    print(f"  ROC-AUC:  {metrics['ROC-AUC']:.4f}")
    print(f"\n  图表和结果已保存至: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
