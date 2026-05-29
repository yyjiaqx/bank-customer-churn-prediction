# -*- coding: utf-8 -*-
"""
电信客户流失预测 — 模型评估
指标计算、混淆矩阵、ROC 曲线、分类报告。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, f1_score, recall_score, precision_score
)
import os

BLUE = (0.122, 0.467, 0.706)
ORANGE = (1.000, 0.498, 0.055)


def evaluate_model(model, X_test, y_test, output_dir="outputs/figures"):
    """评估训练好的模型，打印所有关键指标。

    参数
    ----------
    model : LogisticRegression
    X_test : np.ndarray
    y_test : np.ndarray
    output_dir : str

    返回
    -------
    dict
        包含所有评估指标的字典。
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 计算核心指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n" + "=" * 60)
    print("  模型评估 — Logistic Regression")
    print("=" * 60)
    print(f"  准确率 (Accuracy):   {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  精确率 (Precision):  {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  召回率 (Recall):     {recall:.4f}  ({recall*100:.2f}%)")
    print(f"  F1 分数:             {f1:.4f}")
    print(f"  ROC-AUC:             {auc:.4f}")
    print(f"\n  混淆矩阵:")
    print(f"  TP={tp:4d} (真正)  FP={fp:4d} (假正)")
    print(f"  FN={fn:4d} (假负)  TN={tn:4d} (真负)")
    print(f"\n  分类报告:")
    print(classification_report(y_test, y_pred, target_names=['未流失 (0)', '已流失 (1)']))

    return {
        'Model': 'Logistic Regression',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': auc,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
    }


def plot_confusion_matrix(cm, title_prefix, output_dir="outputs/figures"):
    """绘制并保存混淆矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测未流失', '预测已流失'],
                yticklabels=['实际未流失', '实际已流失'],
                annot_kws={'fontsize': 14}, cbar=False, linewidths=1)
    ax.set_title(f'{title_prefix} — 混淆矩阵', fontsize=14, fontweight='bold')
    ax.set_ylabel('实际', fontsize=12)
    ax.set_xlabel('预测', fontsize=12)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(path)
    plt.close()
    print(f"  [已保存] {path}")


def plot_roc_curve(model, X_test, y_test, output_dir="outputs/figures"):
    """绘制并保存 ROC 曲线。"""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=ORANGE, linewidth=2.5,
            label=f'Logistic Regression (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机分类器 (AUC = 0.50)')
    ax.fill_between(fpr, tpr, alpha=0.1, color=ORANGE)
    ax.set_xlabel('假正率 (False Positive Rate)', fontsize=12)
    ax.set_ylabel('真正率 (True Positive Rate)', fontsize=12)
    ax.set_title('ROC 曲线 — Logistic Regression', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(path)
    plt.close()
    print(f"  [已保存] {path}")
