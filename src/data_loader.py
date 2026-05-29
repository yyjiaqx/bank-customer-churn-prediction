# -*- coding: utf-8 -*-
"""
电信客户流失预测 — 数据加载与探索性分析（EDA）
加载 Telco 数据集，输出摘要统计，生成 EDA 图表。
"""
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 全局样式设置
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 120
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# 银行蓝 + 警示橙配色
BLUE = (0.122, 0.467, 0.706)
ORANGE = (1.000, 0.498, 0.055)
COLORS = [BLUE, ORANGE]


def load_data(data_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """加载 Telco 客户流失数据集。

    参数
    ----------
    data_path : str
        CSV 文件路径。

    返回
    -------
    pd.DataFrame
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"未找到数据集 '{data_path}'。"
            f"请将 Telco Customer Churn CSV 文件放入 data/ 目录。"
        )
    df = pd.read_csv(data_path, engine='python')
    print(f"[信息] 数据集加载完成: {df.shape[0]} 行 × {df.shape[1]} 列")
    return df


def basic_info(df):
    """打印数据集基本信息（形状、类型、统计摘要）。"""
    print("\n" + "=" * 60)
    print("  数据集概览")
    print("=" * 60)
    print(f"  形状:        {df.shape}")
    print(f"  内存占用:    {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"\n  列数据类型:")
    print(df.dtypes.to_string())
    print(f"\n  描述性统计（数值列）:")
    print(df.describe().to_string())


def check_missing(df):
    """检查并报告缺失值。"""
    print("\n" + "=" * 60)
    print("  缺失值检查")
    print("=" * 60)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("  未检测到缺失值。")
    else:
        print(missing.to_string())


def class_distribution(df, output_dir="outputs/figures"):
    """流失比例饼图。"""
    counts = df['Churn'].value_counts()
    print("\n" + "=" * 60)
    print("  流失分布")
    print("=" * 60)
    for label, cnt in counts.items():
        print(f"  {label}:  {cnt:5d}  ({cnt/len(df)*100:.2f}%)")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
           colors=COLORS, startangle=90, explode=(0, 0.05),
           textprops={'fontsize': 13})
    ax.set_title('客户流失分布', fontsize=15, fontweight='bold')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'churn_distribution.png')
    plt.savefig(path)
    plt.close()
    print(f"  [已保存] {path}")


def numeric_distributions(df, output_dir="outputs/figures"):
    """按流失状态分组的数值特征直方图（用 bar 兼容旧版 matplotlib）。"""
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, col in enumerate(numeric_cols):
        for churn_val, color, label in [('No', BLUE, '未流失'), ('Yes', ORANGE, '已流失')]:
            raw = df[df['Churn'] == churn_val][col]
            # TotalCharges 是 object 类型，需先转数值
            if raw.dtype == object:
                raw = pd.to_numeric(raw, errors='coerce').dropna()
            subset = raw.values.astype(float)
            counts, bins = np.histogram(subset, bins=30)
            axes[i].bar(bins[:-1], counts, width=np.diff(bins),
                       alpha=0.6, color=color, label=label, edgecolor='white',
                       align='edge')
        axes[i].set_title(f'{col} 按流失状态', fontsize=13, fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('数量')
        axes[i].legend()
    plt.tight_layout()
    path = os.path.join(output_dir, 'numeric_distributions.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"  [已保存] {path}")


def categorical_analysis(df, output_dir="outputs/figures"):
    """按关键分类特征展示流失率柱状图。"""
    cat_features = ['Contract', 'InternetService', 'PaymentMethod',
                    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                    'PhoneService', 'PaperlessBilling', 'TechSupport']
    n_cols = 2
    n_rows = (len(cat_features) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(cat_features):
        # 计算每个类别的流失率
        churn_rate = df.groupby(feat)['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).sort_values(ascending=False)
        bars = axes[i].bar(range(len(churn_rate)), churn_rate.values,
                           color=[BLUE if v < 30 else ORANGE for v in churn_rate.values])
        axes[i].set_xticks(range(len(churn_rate)))
        axes[i].set_xticklabels(churn_rate.index, rotation=30, ha='right', fontsize=8)
        axes[i].set_title(f'{feat} 流失率', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('流失率 (%)')
        axes[i].set_ylim(0, max(churn_rate.values) * 1.3)
        # 添加数值标签
        for bar, val in zip(bars, churn_rate.values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('各分类特征的流失率对比', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'categorical_analysis.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [已保存] {path}")


def correlation_heatmap(df, output_dir="outputs/figures"):
    """数值特征相关性热力图。"""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='Blues',
                vmin=-1, vmax=1, center=0, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_title('特征相关性热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'correlation_heatmap.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"  [已保存] {path}")


def tenure_vs_charges(df, output_dir="outputs/figures"):
    """在网时长 vs 月消费散点图（按流失状态着色）。"""
    # 先确保 TotalCharges 转为数值
    plot_df = df.copy()
    plot_df['TotalCharges'] = pd.to_numeric(plot_df['TotalCharges'], errors='coerce').fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    for churn_val, color, label, marker in [
        ('No', BLUE, '未流失', 'o'),
        ('Yes', ORANGE, '已流失', 'x')
    ]:
        subset = plot_df[plot_df['Churn'] == churn_val]
        ax.scatter(subset['tenure'], subset['MonthlyCharges'],
                   c=color, label=label, marker=marker, alpha=0.4, s=20)
    ax.set_xlabel('在网月数', fontsize=12)
    ax.set_ylabel('月消费 (USD)', fontsize=12)
    ax.set_title('在网时长 vs 月消费 — 按流失状态', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, 'tenure_vs_charges.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"  [已保存] {path}")


def run_eda(df, output_dir="outputs/figures"):
    """运行完整 EDA：打印摘要并生成所有图表。"""
    basic_info(df)
    check_missing(df)
    class_distribution(df, output_dir)
    numeric_distributions(df, output_dir)
    categorical_analysis(df, output_dir)
    correlation_heatmap(df, output_dir)
    tenure_vs_charges(df, output_dir)
    print("\n[信息] EDA 完成 — 所有图表已保存至 outputs/figures/")
