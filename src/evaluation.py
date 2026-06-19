# -*- coding: utf-8 -*-
"""
电信客户流失预测 — 模型评估
多模型指标对比、混淆矩阵、ROC曲线、PR曲线、模型对比图。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, f1_score, recall_score, precision_score,
    precision_recall_curve, average_precision_score
)
import os

BLUE = (0.122, 0.467, 0.706)
ORANGE = (1.000, 0.498, 0.055)
GREEN = (0.172, 0.627, 0.172)
RED = (0.839, 0.153, 0.157)
COLORS = [BLUE, ORANGE, GREEN, RED]


def evaluate_model(model, X_test, y_test, model_name="Model", output_dir="outputs/figures"):
    """评估单个模型，打印关键指标并返回指标字典。"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n" + "=" * 60)
    print(f"  模型评估 — {model_name}")
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
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': auc,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
    }


def evaluate_all_models(models_dict, X_test, y_test):
    """评估所有模型并返回对比 DataFrame。"""
    all_metrics = []
    for name, (model, _) in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)

    results_df = pd.DataFrame(all_metrics).set_index('Model')
    return results_df


# ------------------------------------------------------------
#  可视化: 混淆矩阵
# ------------------------------------------------------------

def plot_confusion_matrix(cm, title_prefix, output_dir="outputs/figures"):
    """绘制并保存单个混淆矩阵热力图。"""
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


def plot_all_confusion_matrices(models_dict, X_test, y_test, output_dir="outputs/figures"):
    """绘制所有模型的混淆矩阵（横向并排）。"""
    n = len(models_dict)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, (model, _)) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['预测未流失', '预测已流失'],
                    yticklabels=['实际未流失', '实际已流失'],
                    annot_kws={'fontsize': 12}, cbar=False, linewidths=1, ax=ax)
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        ax.set_ylabel('实际', fontsize=11)
        ax.set_xlabel('预测', fontsize=11)

    plt.suptitle('各模型混淆矩阵对比', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'confusion_matrices_all.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [已保存] {path}")


# ------------------------------------------------------------
#  可视化: ROC 曲线
# ------------------------------------------------------------

def plot_roc_curve(model, X_test, y_test, model_name="Model", output_dir="outputs/figures", single=True):
    """绘制 ROC 曲线（单模型或多模型）。"""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    if single:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color=ORANGE, linewidth=2.5,
                label=f'{model_name} (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机分类器 (AUC = 0.50)')
        ax.fill_between(fpr, tpr, alpha=0.1, color=ORANGE)
        ax.set_xlabel('假正率 (False Positive Rate)', fontsize=12)
        ax.set_ylabel('真正率 (True Positive Rate)', fontsize=12)
        ax.set_title(f'ROC 曲线 — {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(path)
        plt.close()
        print(f"  [已保存] {path}")

    return fpr, tpr, auc


def plot_roc_curves_all(models_dict, X_test, y_test, output_dir="outputs/figures"):
    """绘制所有模型的 ROC 曲线对比图。"""
    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, (name, (model, _)) in enumerate(models_dict.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=COLORS[idx], linewidth=2.5,
                label=f'{name} (AUC = {auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机分类器 (AUC = 0.50)')
    ax.set_xlabel('假正率 (False Positive Rate)', fontsize=12)
    ax.set_ylabel('真正率 (True Positive Rate)', fontsize=12)
    ax.set_title('ROC 曲线 — 多模型对比', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'roc_curves_all.png')
    plt.savefig(path)
    plt.close()
    print(f"  [已保存] {path}")


# ------------------------------------------------------------
#  可视化: PR 曲线 (Precision-Recall)
# ------------------------------------------------------------

def plot_pr_curves_all(models_dict, X_test, y_test, output_dir="outputs/figures"):
    """绘制所有模型的 PR 曲线对比图。"""
    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, (name, (model, _)) in enumerate(models_dict.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, color=COLORS[idx], linewidth=2.5,
                label=f'{name} (AP = {ap:.4f})')

    # 基线：正样本比例
    baseline = np.sum(y_test) / len(y_test)
    ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.6,
               label=f'基线 (正样本比例 = {baseline:.3f})')

    ax.set_xlabel('召回率 (Recall)', fontsize=12)
    ax.set_ylabel('精确率 (Precision)', fontsize=12)
    ax.set_title('精确率-召回率曲线 (PR Curve) — 多模型对比', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'pr_curves_all.png')
    plt.savefig(path)
    plt.close()
    print(f"  [已保存] {path}")


# ------------------------------------------------------------
#  可视化: 模型指标对比柱状图
# ------------------------------------------------------------

def plot_model_comparison(results_df, output_dir="outputs/figures"):
    """绘制多模型核心指标对比柱状图。"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    models = results_df.index.tolist()
    n_models = len(models)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.7 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model_name in enumerate(models):
        values = results_df.loc[model_name, metrics].values.astype(float)
        bars = ax.bar(x + i * width, values, width,
                      label=model_name, color=COLORS[i], alpha=0.85)
        # 数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('多模型核心指标对比', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  [已保存] {path}")
