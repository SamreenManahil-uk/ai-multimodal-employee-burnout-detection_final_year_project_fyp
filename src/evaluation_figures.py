import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import load
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


X_TEST_PATH = os.path.join("outputs", "X_test_encoded.csv")
Y_TEST_PATH = os.path.join("outputs", "y_test.csv")

RF_MODEL_PATH = os.path.join("models", "random_forest_model.joblib")
XGB_MODEL_PATH = os.path.join("models", "xgboost_model.joblib")

OUTPUT_DIR = "outputs"


def generate_confusion_matrix(model, X_test, y_test, model_name, output_path):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Burnout", "Burnout"]
    )

    display.plot(values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def generate_roc_comparison(rf_model, xgb_model, X_test, y_test):
    plt.figure()

    ax = plt.gca()

    RocCurveDisplay.from_estimator(
        rf_model,
        X_test,
        y_test,
        ax=ax,
        name="Random Forest"
    )

    RocCurveDisplay.from_estimator(
        xgb_model,
        X_test,
        y_test,
        ax=ax,
        name="XGBoost"
    )

    plt.title("ROC Curve Comparison: Random Forest vs XGBoost")
    output_path = os.path.join(OUTPUT_DIR, "roc_curve_comparison.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def generate_feature_importance_chart(csv_path, title, output_path, top_n=10):
    df = pd.read_csv(csv_path)

    df = df.sort_values(
        by="Importance",
        ascending=False
    ).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(df["Feature"], df["Importance"])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    print("=" * 70)
    print("GENERATING EVALUATION FIGURES")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    rf_model = load(RF_MODEL_PATH)
    xgb_model = load(XGB_MODEL_PATH)

    generate_confusion_matrix(
        rf_model,
        X_test,
        y_test,
        "Random Forest",
        os.path.join(OUTPUT_DIR, "random_forest_confusion_matrix.png")
    )

    generate_confusion_matrix(
        xgb_model,
        X_test,
        y_test,
        "XGBoost",
        os.path.join(OUTPUT_DIR, "xgboost_confusion_matrix.png")
    )

    generate_roc_comparison(
        rf_model,
        xgb_model,
        X_test,
        y_test
    )

    generate_feature_importance_chart(
        os.path.join(OUTPUT_DIR, "random_forest_feature_importance.csv"),
        "Random Forest Top 10 Feature Importance",
        os.path.join(OUTPUT_DIR, "random_forest_feature_importance_chart.png")
    )

    generate_feature_importance_chart(
        os.path.join(OUTPUT_DIR, "xgboost_feature_importance.csv"),
        "XGBoost Top 10 Feature Importance",
        os.path.join(OUTPUT_DIR, "xgboost_feature_importance_chart.png")
    )

    print("\nEvaluation figures generated successfully.")


if __name__ == "__main__":
    main()