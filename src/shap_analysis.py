import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
from joblib import load


MODEL_PATH = os.path.join("models", "best_model.joblib")
X_TEST_PATH = os.path.join("outputs", "X_test_encoded.csv")

SHAP_SUMMARY_PATH = os.path.join("outputs", "shap_summary_plot.png")
SHAP_BAR_PATH = os.path.join("outputs", "shap_bar_plot.png")
SHAP_WATERFALL_PATH = os.path.join("outputs", "shap_waterfall_plot.png")
SHAP_IMPORTANCE_PATH = os.path.join("outputs", "shap_feature_importance.csv")


def run_shap_analysis():
    print("=" * 70)
    print("STEP: SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)

    os.makedirs("outputs", exist_ok=True)

    print("\nLoading best model...")
    model = load(MODEL_PATH)

    print("Loading test data...")
    X_test = pd.read_csv(X_TEST_PATH)

    # Use a smaller sample to make SHAP faster
    X_sample = X_test.sample(
        n=min(1000, len(X_test)),
        random_state=42
    )

    print("SHAP sample shape:", X_sample.shape)

    print("\nCreating SHAP Tree Explainer...")
    explainer = shap.TreeExplainer(model)

    print("Calculating SHAP values...")
    shap_values = explainer(X_sample)

    # For binary classification, sometimes SHAP returns 3D values
    if len(shap_values.values.shape) == 3:
        shap_values_binary = shap_values[:, :, 1]
    else:
        shap_values_binary = shap_values

    print("\nGenerating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(
        shap_values_binary,
        X_sample,
        show=False
    )
    plt.savefig(SHAP_SUMMARY_PATH, bbox_inches="tight", dpi=300)
    plt.close()

    print("Saved:", SHAP_SUMMARY_PATH)

    print("\nGenerating SHAP bar plot...")
    plt.figure()
    shap.plots.bar(
        shap_values_binary,
        show=False
    )
    plt.savefig(SHAP_BAR_PATH, bbox_inches="tight", dpi=300)
    plt.close()

    print("Saved:", SHAP_BAR_PATH)

    print("\nGenerating SHAP waterfall plot...")
    plt.figure()
    shap.plots.waterfall(
        shap_values_binary[0],
        show=False
    )
    plt.savefig(SHAP_WATERFALL_PATH, bbox_inches="tight", dpi=300)
    plt.close()

    print("Saved:", SHAP_WATERFALL_PATH)

    print("\nSaving SHAP feature importance...")

    importance_values = np.abs(shap_values_binary.values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": X_sample.columns,
        "Mean_Absolute_SHAP_Value": importance_values
    }).sort_values(
        by="Mean_Absolute_SHAP_Value",
        ascending=False
    )

    importance_df.to_csv(SHAP_IMPORTANCE_PATH, index=False)

    print("Saved:", SHAP_IMPORTANCE_PATH)

    print("\nTop 10 SHAP Features:")
    print(importance_df.head(10))

    print("\nSHAP analysis completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_shap_analysis()