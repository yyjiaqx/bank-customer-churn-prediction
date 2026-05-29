# -*- coding: utf-8 -*-
"""
电信客户流失预测 — 预处理管线
处理：ID 删除、TotalCharges 清洗、编码、标准化、训练/测试集划分。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def clean_total_charges(df):
    """将 TotalCharges 从 object 转为 float，空字符串填 0。"""
    df = df.copy()
    # 转数值，空字符串变 NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    nan_count = df['TotalCharges'].isnull().sum()
    if nan_count > 0:
        print(f"[信息] TotalCharges: {nan_count} 个空值（tenure=0 的新客户）→ 填充为 0")
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    return df


def preprocess(df, random_state=42):
    """完整的预处理管线。

    步骤:
    1. 删除 'customerID'
    2. 清洗 'TotalCharges'（string → float，NaN → 0）
    3. 编码目标变量 'Churn'（Yes=1, No=0）
    4. Label-Encode 二分类特征
    5. One-Hot Encode 多分类特征
    6. StandardScaler 标准化数值特征
    7. 训练/测试集 8:2 分层划分

    参数
    ----------
    df : pd.DataFrame
        原始数据集。
    random_state : int
        随机种子。

    返回
    -------
    X_train, X_test : np.ndarray
    y_train, y_test : np.ndarray
    feature_names : list
        预处理后的特征名列表。
    """
    data = df.copy()

    # 步骤 1：删除 ID 列
    data.drop('customerID', axis=1, inplace=True)
    print("[信息] 已删除 'customerID'")

    # 步骤 2：清洗 TotalCharges
    data = clean_total_charges(data)

    # 步骤 3：编码目标变量
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    print(f"[信息] 目标变量 'Churn' 已编码: {data['Churn'].sum()} 个流失客户")

    # 步骤 4：Label-Encode 二分类特征
    # "No internet service" / "No phone service" 等价于 No，统一编码为 0
    binary_mappings = {
        'gender': {'Male': 1, 'Female': 0},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'PaperlessBilling': {'Yes': 1, 'No': 0},
        'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'OnlineBackup': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'StreamingTV': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'StreamingMovies': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': 0},
    }
    encoded_binary = []
    for col, mapping in binary_mappings.items():
        data[col] = data[col].map(mapping)
        encoded_binary.append(col)
    print(f"[信息] 已 Label-Encode {len(encoded_binary)} 个二分类特征")

    # 步骤 5：One-Hot Encode 多分类特征
    multi_cat_cols = ['InternetService', 'Contract', 'PaymentMethod']
    data = pd.get_dummies(data, columns=multi_cat_cols, drop_first=False, dtype=np.float64)
    print(f"[信息] 已 One-Hot Encode: {multi_cat_cols}")

    # 步骤 6：StandardScaler 标准化
    feature_cols = [c for c in data.columns if c != 'Churn']
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    print(f"[信息] StandardScaler 已应用于 {len(feature_cols)} 个特征")

    # 拆分特征与目标
    X = data.drop('Churn', axis=1).values
    y = data['Churn'].values
    final_feature_names = data.drop('Churn', axis=1).columns.tolist()

    # 步骤 7：训练/测试集划分（80/20，分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    churn_train = np.sum(y_train)
    churn_test = np.sum(y_test)
    print(f"[信息] 训练集: {X_train.shape[0]} 样本，流失: {churn_train} ({churn_train/len(y_train)*100:.1f}%)")
    print(f"[信息] 测试集: {X_test.shape[0]} 样本，流失: {churn_test} ({churn_test/len(y_test)*100:.1f}%)")

    return X_train, X_test, y_train, y_test, final_feature_names
