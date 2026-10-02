"""
Standalone script: generate the ROC curve for your final selected model
(XGBoost, saved as best_model.joblib) using the untouched test set.

Run this from your project root:
    python generate_xgb_roc_curve.py

Output:
    outputs/best_model_roc_curve.png
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import load
from sklearn.metrics import RocCurveDisplay, roc_auc_score

# Adjust these paths if your folder layout differs
MODEL_PATH = os.path.join("models", "feature_engineered", "best_model.joblib")
X_TEST_PATH = os.path.join("outputs", "feature_engineered", "X_test_encoded.csv")
Y_TEST_PATH = os.path.join("outputs", "feature_engineered", "y_test.csv")

OUTPUT_PATH = os.path.join("outputs", "best_model_roc_curve.png")


def main():
    print("Loading selected model (XGBoost)...")
    model = load(MODEL_PATH)
    print("Model type:", type(model).__name__)

    print("Loading test data...")
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Compute AUC for sanity-check printout
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"Test ROC-AUC: {auc:.4f}")

    print("Generating ROC curve...")
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"XGBoost — Final Test ROC Curve (AUC = {auc:.3f})")

    os.makedirs("outputs", exist_ok=True)
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=300)
    plt.close()

    print("\nSaved ROC curve to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()