"""
Telco Customer Churn — Model Evaluation
Metrics, confusion matrix, ROC curve, classification report.
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

BLUE = '#1f77b4'
ORANGE = '#ff7f0e'


def evaluate_model(model, X_test, y_test, output_dir="outputs/figures"):
    """Evaluate trained model and return metrics.

    Parameters
    ----------
    model : LogisticRegression
    X_test : np.ndarray
    y_test : np.ndarray
    output_dir : str

    Returns
    -------
    dict with evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION — Logistic Regression")
    print("=" * 60)
    print(f"  Accuracy:   {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision:  {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  Recall:     {recall:.4f}  ({recall*100:.2f}%)")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  ROC-AUC:    {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  TP={tp:4d}  FP={fp:4d}")
    print(f"  FN={fn:4d}  TN={tn:4d}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn (0)', 'Churn (1)']))

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
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No Churn', 'Predicted Churn'],
                yticklabels=['Actual No Churn', 'Actual Churn'],
                annot_kws={'fontsize': 14}, cbar=False, linewidths=1)
    ax.set_title(f'{title_prefix} — Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(path)
    plt.close()
    print(f"  [SAVED] {path}")


def plot_roc_curve(model, X_test, y_test, output_dir="outputs/figures"):
    """Plot and save ROC curve."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=ORANGE, linewidth=2.5,
            label=f'Logistic Regression (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier (AUC = 0.50)')
    ax.fill_between(fpr, tpr, alpha=0.1, color=ORANGE)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — Logistic Regression', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(path)
    plt.close()
    print(f"  [SAVED] {path}")
