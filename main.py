"""
Telco Customer Churn Prediction — Main Pipeline
Run: python main.py

Steps:
  1. Load data
  2. EDA (summary stats + charts)
  3. Preprocessing (encode, scale, split)
  4. Model training (Logistic Regression + GridSearchCV)
  5. Evaluation (metrics, confusion matrix, ROC curve, coefficients)
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
    # Paths
    DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    OUTPUT_DIR = "outputs/figures"
    RANDOM_STATE = 42

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============================================================
    # Step 1: Load Data
    # ============================================================
    print("\n" + "#" * 60)
    print("#  STEP 1: Data Loading & EDA")
    print("#" * 60)
    df = load_data(DATA_PATH)
    run_eda(df, OUTPUT_DIR)

    # ============================================================
    # Step 2: Preprocessing
    # ============================================================
    print("\n" + "#" * 60)
    print("#  STEP 2: Preprocessing")
    print("#" * 60)
    X_train, X_test, y_train, y_test, feature_names = preprocess(df, random_state=RANDOM_STATE)
    print(f"\n[INFO] Final feature set ({len(feature_names)}):")
    for i, name in enumerate(feature_names):
        print(f"  {i:2d}. {name}")

    # ============================================================
    # Step 3: Model Training
    # ============================================================
    print("\n" + "#" * 60)
    print("#  STEP 3: Model Training (Logistic Regression)")
    print("#" * 60)
    model, best_params = train_logistic_regression(X_train, y_train, RANDOM_STATE)

    # Coefficient analysis
    coef_df = get_coefficients(model, feature_names)
    print_coefficient_insights(coef_df)
    plot_coefficients(coef_df, OUTPUT_DIR)

    # ============================================================
    # Step 4: Evaluation
    # ============================================================
    print("\n" + "#" * 60)
    print("#  STEP 4: Model Evaluation")
    print("#" * 60)
    metrics = evaluate_model(model, X_test, y_test, OUTPUT_DIR)
    cm = confusion_matrix(y_test, model.predict(X_test))
    plot_confusion_matrix(cm, "Logistic Regression", OUTPUT_DIR)
    plot_roc_curve(model, X_test, y_test, OUTPUT_DIR)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Model:   Logistic Regression")
    print(f"  Config:  {best_params}")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1-Score:  {metrics['F1-Score']:.4f}")
    print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    print(f"\n  Charts and results saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
