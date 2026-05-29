"""
Telco Customer Churn — Preprocessing Pipeline
Handles: ID removal, TotalCharges cleaning, encoding, scaling, train/test split.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def clean_total_charges(df):
    """Convert TotalCharges from object to float, handling empty strings."""
    df = df.copy()
    # Convert to numeric, empty strings become NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    nan_count = df['TotalCharges'].isnull().sum()
    if nan_count > 0:
        print(f"[INFO] TotalCharges: {nan_count} NaN values (tenure=0 customers) → filled with 0")
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    return df


def preprocess(df, random_state=42):
    """Full preprocessing pipeline for the Telco Churn dataset.

    Steps:
    1. Drop 'customerID'
    2. Clean 'TotalCharges' (string → float, NaN → 0)
    3. Encode target 'Churn' (Yes=1, No=0)
    4. Label-encode binary categorical features
    5. One-Hot encode multi-category features
    6. StandardScaler on numeric features
    7. Train/test split (80/20, stratified)

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    random_state : int
        Random seed.

    Returns
    -------
    X_train, X_test : np.ndarray
    y_train, y_test : np.ndarray
    feature_names : list
    """
    data = df.copy()

    # Step 1: Drop ID
    data.drop('customerID', axis=1, inplace=True)
    print("[INFO] Dropped 'customerID'")

    # Step 2: Clean TotalCharges
    data = clean_total_charges(data)

    # Step 3: Encode target
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    print(f"[INFO] Target 'Churn' encoded: {data['Churn'].sum()} churners")

    # Step 4: Label-encode binary categorical features
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
    print(f"[INFO] Label-encoded {len(encoded_binary)} binary features")

    # Step 5: One-Hot encode multi-category features
    multi_cat_cols = ['InternetService', 'Contract', 'PaymentMethod']
    data = pd.get_dummies(data, columns=multi_cat_cols, drop_first=False, dtype=np.float64)
    print(f"[INFO] One-Hot encoded: {multi_cat_cols}")

    # Identify numeric columns for scaling
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    # Add the binary encoded columns (they are now 0/1 numeric)
    numeric_cols += encoded_binary

    # Step 6: StandardScaler
    feature_cols = [c for c in data.columns if c != 'Churn']
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    print(f"[INFO] StandardScaler applied to {len(feature_cols)} features")

    # Split
    X = data.drop('Churn', axis=1).values
    y = data['Churn'].values
    final_feature_names = data.drop('Churn', axis=1).columns.tolist()

    # Step 7: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    churn_train = np.sum(y_train)
    churn_test = np.sum(y_test)
    print(f"[INFO] Train: {X_train.shape[0]} samples, churn: {churn_train} ({churn_train/len(y_train)*100:.1f}%)")
    print(f"[INFO] Test:  {X_test.shape[0]} samples, churn: {churn_test} ({churn_test/len(y_test)*100:.1f}%)")

    return X_train, X_test, y_train, y_test, final_feature_names
