"""
Telco Customer Churn — Model Training (Logistic Regression)
Trains LR with GridSearchCV and returns coefficients for interpretation.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

BLUE = '#1f77b4'
ORANGE = '#ff7f0e'


def train_logistic_regression(X_train, y_train, random_state=42):
    """Train Logistic Regression with GridSearchCV.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    random_state : int

    Returns
    -------
    model : LogisticRegression (best estimator)
    best_params : dict
    """
    print("\n" + "=" * 60)
    print("  TRAINING: Logistic Regression")
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

    print(f"\n  Best params:      {grid.best_params_}")
    print(f"  Best CV ROC-AUC:  {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


def get_coefficients(model, feature_names):
    """Extract and sort Logistic Regression coefficients.

    Parameters
    ----------
    model : LogisticRegression
    feature_names : list of str

    Returns
    -------
    pd.DataFrame with columns ['Feature', 'Coefficient', 'Odds_Ratio', 'Abs_Coefficient']
        sorted by absolute coefficient value descending.
    """
    coef = model.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef,
        'Odds_Ratio': np.exp(coef),
        'Abs_Coefficient': np.abs(coef),
    }).sort_values('Abs_Coefficient', ascending=False).reset_index(drop=True)

    return coef_df


def plot_coefficients(coef_df, output_dir="outputs/figures", top_n=15):
    """Plot horizontal bar chart of top feature coefficients.

    Parameters
    ----------
    coef_df : pd.DataFrame
    output_dir : str
    top_n : int
    """
    top = coef_df.head(top_n).iloc[::-1]  # Reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [ORANGE if c > 0 else BLUE for c in top['Coefficient'].values]
    bars = ax.barh(range(len(top)), top['Coefficient'].values, color=colors)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['Feature'].values, fontsize=10)
    ax.set_xlabel('Coefficient (Log-Odds)', fontsize=12)
    ax.set_title('Logistic Regression — Top Feature Coefficients', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)

    # Value labels
    for bar, val in zip(bars, top['Coefficient'].values):
        label_pos = bar.get_width() + 0.02 if val >= 0 else bar.get_width() - 0.1
        ax.text(label_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ORANGE, label='Positive (↑ churn risk)'),
        Patch(facecolor=BLUE, label='Negative (↓ churn risk)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'coefficients.png')
    plt.savefig(path)
    plt.close()
    print(f"  [SAVED] {path}")


def print_coefficient_insights(coef_df):
    """Print business interpretation of top coefficients."""
    print("\n" + "=" * 60)
    print("  FEATURE COEFFICIENT ANALYSIS (Top 10)")
    print("=" * 60)
    print(f"{'Rank':<5} {'Feature':<30} {'Coef':>8} {'Odds Ratio':>10} {'Direction':>12}")
    print("-" * 70)

    for i, row in coef_df.head(10).iterrows():
        direction = "↑ Churn Risk" if row['Coefficient'] > 0 else "↓ Churn Risk"
        print(f"{i+1:<5} {row['Feature']:<30} {row['Coefficient']:>8.4f} {row['Odds_Ratio']:>10.4f} {direction:>12}")

    print("\n  Business Interpretation:")
    print("  — Positive coefficient → feature increases churn probability")
    print("  — Negative coefficient → feature reduces churn probability")
    print("  — Odds Ratio > 1 → multiplicative increase in odds of churning")
    print("  — Odds Ratio < 1 → multiplicative decrease in odds of churning")
