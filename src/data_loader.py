"""
Telco Customer Churn — Data Loader & EDA
Loads the Telco dataset, prints summary statistics, and generates EDA charts.
"""
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 120
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# Color palette
BLUE = (0.122, 0.467, 0.706)
ORANGE = (1.000, 0.498, 0.055)
COLORS = [BLUE, ORANGE]


def load_data(data_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """Load the Telco Customer Churn dataset.

    Parameters
    ----------
    data_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            f"Please place the Telco Customer Churn CSV in the data/ folder."
        )
    df = pd.read_csv(data_path, engine='python')
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def basic_info(df):
    """Print basic dataset information."""
    print("\n" + "=" * 60)
    print("  DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Shape:        {df.shape}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"\n  Column dtypes:")
    print(df.dtypes.to_string())
    print(f"\n  Descriptive statistics (numeric):")
    print(df.describe().to_string())


def check_missing(df):
    """Check and report missing values."""
    print("\n" + "=" * 60)
    print("  MISSING VALUES CHECK")
    print("=" * 60)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("  No missing values detected.")
    else:
        print(missing.to_string())


def class_distribution(df, output_dir="outputs/figures"):
    """Pie chart of churn vs. non-churn."""
    counts = df['Churn'].value_counts()
    print("\n" + "=" * 60)
    print("  CHURN DISTRIBUTION")
    print("=" * 60)
    for label, cnt in counts.items():
        print(f"  {label}:  {cnt:5d}  ({cnt/len(df)*100:.2f}%)")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
           colors=COLORS, startangle=90, explode=(0, 0.05),
           textprops={'fontsize': 13})
    ax.set_title('Customer Churn Distribution', fontsize=15, fontweight='bold')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'churn_distribution.png')
    plt.savefig(path)
    plt.close()
    print(f"  [SAVED] {path}")


def numeric_distributions(df, output_dir="outputs/figures"):
    """Histograms of numeric features split by Churn — using bar for compatibility."""
    import numpy as np
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, col in enumerate(numeric_cols):
        for churn_val, color, label in [('No', BLUE, 'No Churn'), ('Yes', ORANGE, 'Churn')]:
            raw = df[df['Churn'] == churn_val][col]
            # Convert to numeric if needed (TotalCharges is object)
            if raw.dtype == object:
                raw = pd.to_numeric(raw, errors='coerce').dropna()
            subset = raw.values.astype(float)
            counts, bins = np.histogram(subset, bins=30)
            axes[i].bar(bins[:-1], counts, width=np.diff(bins),
                       alpha=0.6, color=color, label=label, edgecolor='white',
                       align='edge')
        axes[i].set_title(f'{col} by Churn', fontsize=13, fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].legend()
    plt.tight_layout()
    path = os.path.join(output_dir, 'numeric_distributions.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"  [SAVED] {path}")


def categorical_analysis(df, output_dir="outputs/figures"):
    """Bar charts of churn rate by key categorical features."""
    cat_features = ['Contract', 'InternetService', 'PaymentMethod',
                    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                    'PhoneService', 'PaperlessBilling', 'TechSupport']
    n_cols = 2
    n_rows = (len(cat_features) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(cat_features):
        # Compute churn rate per category
        churn_rate = df.groupby(feat)['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).sort_values(ascending=False)
        bars = axes[i].bar(range(len(churn_rate)), churn_rate.values,
                           color=[BLUE if v < 30 else ORANGE for v in churn_rate.values])
        axes[i].set_xticks(range(len(churn_rate)))
        axes[i].set_xticklabels(churn_rate.index, rotation=30, ha='right', fontsize=8)
        axes[i].set_title(f'Churn Rate by {feat}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Churn Rate (%)')
        axes[i].set_ylim(0, max(churn_rate.values) * 1.3)
        # Add value labels
        for bar, val in zip(bars, churn_rate.values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Churn Rate by Categorical Features', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'categorical_analysis.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {path}")


def correlation_heatmap(df, output_dir="outputs/figures"):
    """Correlation heatmap of numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='Blues',
                vmin=-1, vmax=1, center=0, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'correlation_heatmap.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"  [SAVED] {path}")


def tenure_vs_charges(df, output_dir="outputs/figures"):
    """Scatter plot of tenure vs MonthlyCharges colored by Churn."""
    # Ensure TotalCharges is numeric for plotting
    plot_df = df.copy()
    plot_df['TotalCharges'] = pd.to_numeric(plot_df['TotalCharges'], errors='coerce').fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    for churn_val, color, label, marker in [
        ('No', BLUE, 'No Churn', 'o'),
        ('Yes', ORANGE, 'Churn', 'x')
    ]:
        subset = plot_df[plot_df['Churn'] == churn_val]
        ax.scatter(subset['tenure'], subset['MonthlyCharges'],
                   c=color, label=label, marker=marker, alpha=0.4, s=20)
    ax.set_xlabel('Tenure (months)', fontsize=12)
    ax.set_ylabel('Monthly Charges ($)', fontsize=12)
    ax.set_title('Tenure vs Monthly Charges by Churn Status', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, 'tenure_vs_charges.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"  [SAVED] {path}")


def run_eda(df, output_dir="outputs/figures"):
    """Run full EDA: print summaries and generate all charts."""
    basic_info(df)
    check_missing(df)
    class_distribution(df, output_dir)
    numeric_distributions(df, output_dir)
    categorical_analysis(df, output_dir)
    correlation_heatmap(df, output_dir)
    tenure_vs_charges(df, output_dir)
    print("\n[INFO] EDA complete — all charts saved to outputs/figures/")
