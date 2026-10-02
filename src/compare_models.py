"""
Model comparison and final test evaluation.

This module:

1. Reads Random Forest cross-validation results.
2. Reads XGBoost cross-validation results.
3. Selects the best model using mean CV F1.
4. Copies the selected candidate to best_model.joblib.
5. Evaluates only the selected model on the untouched test set.
6. Saves the final test metrics and ROC curve.

The test set is not used to choose the best model.
"""

import os
import re
import shutil

import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from joblib import load

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)


# ==========================================================
# FILE PATHS
# ==========================================================

RF_RESULTS_PATH = os.path.join(
    "outputs",
    "random_forest_results.txt",
)

XGB_RESULTS_PATH = os.path.join(
    "outputs",
    "xgboost_results.txt",
)

RF_MODEL_PATH = os.path.join(
    "models",
    "random_forest_model.joblib",
)

XGB_MODEL_PATH = os.path.join(
    "models",
    "xgboost_model.joblib",
)

X_TEST_PATH = os.path.join(
    "outputs",
    "X_test_encoded.csv",
)

Y_TEST_PATH = os.path.join(
    "outputs",
    "y_test.csv",
)

COMPARISON_PATH = os.path.join(
    "outputs",
    "model_comparison.csv",
)

BEST_MODEL_PATH = os.path.join(
    "models",
    "best_model.joblib",
)

FINAL_RESULT_PATH = os.path.join(
    "outputs",
    "final_model_test_results.txt",
)

FINAL_ROC_PATH = os.path.join(
    "outputs",
    "best_model_roc_curve.png",
)


# ==========================================================
# RESULT READING
# ==========================================================

def extract_metric(
    text,
    metric_name,
):
    """
    Extract one numerical metric from a result text file.
    """

    pattern = (
        rf"{re.escape(metric_name)}:"
        rf"\s*([-+]?[0-9]*\.?[0-9]+)"
    )

    match = re.search(
        pattern,
        text,
    )

    if match is None:
        raise ValueError(
            f"Could not find '{metric_name}' "
            "inside the model result file."
        )

    return float(
        match.group(1)
    )


def read_cv_results(
    file_path,
):
    """
    Read cross-validation results from a model result file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Model result file was not found:\n"
            f"{file_path}"
        )

    with open(
        file_path,
        mode="r",
        encoding="utf-8",
    ) as result_file:
        text = result_file.read()

    return {
        "CV Accuracy": extract_metric(
            text,
            "CV Accuracy",
        ),
        "CV Precision": extract_metric(
            text,
            "CV Precision",
        ),
        "CV Recall": extract_metric(
            text,
            "CV Recall",
        ),
        "CV F1": extract_metric(
            text,
            "CV F1",
        ),
        "CV ROC-AUC": extract_metric(
            text,
            "CV ROC-AUC",
        ),
        "CV Accuracy Standard Deviation": extract_metric(
            text,
            "CV Accuracy Standard Deviation",
        ),
        "CV Precision Standard Deviation": extract_metric(
            text,
            "CV Precision Standard Deviation",
        ),
        "CV Recall Standard Deviation": extract_metric(
            text,
            "CV Recall Standard Deviation",
        ),
        "CV F1 Standard Deviation": extract_metric(
            text,
            "CV F1 Standard Deviation",
        ),
        "CV ROC-AUC Standard Deviation": extract_metric(
            text,
            "CV ROC-AUC Standard Deviation",
        ),
    }


# ==========================================================
# MODEL SELECTION
# ==========================================================

def create_comparison_table():
    """
    Create a model comparison table using CV metrics only.
    """

    rf_results = read_cv_results(
        RF_RESULTS_PATH
    )

    xgb_results = read_cv_results(
        XGB_RESULTS_PATH
    )

    comparison_df = pd.DataFrame(
        [
            {
                "Model": "Random Forest",
                **rf_results,
            },
            {
                "Model": "XGBoost",
                **xgb_results,
            },
        ]
    )

    return comparison_df


def select_best_model(
    comparison_df,
):
    """
    Select the model with the highest mean CV F1.

    CV ROC-AUC is used as a tie-breaker.
    """

    sorted_comparison = (
        comparison_df.sort_values(
            by=[
                "CV F1",
                "CV ROC-AUC",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    best_model_name = (
        sorted_comparison.loc[
            0,
            "Model",
        ]
    )

    comparison_df = comparison_df.copy()

    comparison_df["Selected"] = (
        comparison_df["Model"]
        == best_model_name
    )

    return (
        best_model_name,
        comparison_df,
    )


def copy_selected_model(
    best_model_name,
):
    """
    Copy the selected candidate model to best_model.joblib.
    """

    if best_model_name == "Random Forest":
        source_model_path = RF_MODEL_PATH

    elif best_model_name == "XGBoost":
        source_model_path = XGB_MODEL_PATH

    else:
        raise ValueError(
            f"Unsupported model name: "
            f"{best_model_name}"
        )

    if not os.path.exists(source_model_path):
        raise FileNotFoundError(
            f"Selected candidate model was not found:\n"
            f"{source_model_path}"
        )

    os.makedirs(
        "models",
        exist_ok=True,
    )

    shutil.copy2(
        source_model_path,
        BEST_MODEL_PATH,
    )

    print(
        "\nSelected model copied from:",
        source_model_path,
    )

    print(
        "Best model saved to:",
        BEST_MODEL_PATH,
    )


# ==========================================================
# FINAL TEST EVALUATION
# ==========================================================

def load_test_data():
    """
    Load the untouched encoded test set.
    """

    if not os.path.exists(X_TEST_PATH):
        raise FileNotFoundError(
            f"Encoded test features were not found:\n"
            f"{X_TEST_PATH}"
        )

    if not os.path.exists(Y_TEST_PATH):
        raise FileNotFoundError(
            f"Test labels were not found:\n"
            f"{Y_TEST_PATH}"
        )

    X_test = pd.read_csv(
        X_TEST_PATH
    )

    y_test = pd.read_csv(
        Y_TEST_PATH
    ).squeeze("columns")

    if len(X_test) != len(y_test):
        raise ValueError(
            "The number of test feature rows does not "
            "match the number of test labels."
        )

    print("\nUntouched test data loaded:")

    print(
        "X_test shape:",
        X_test.shape,
    )

    print(
        "y_test shape:",
        y_test.shape,
    )

    return X_test, y_test


def evaluate_selected_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate the selected model once on the untouched test set.
    """

    print(
        "\nEvaluating selected model on "
        "the untouched test set..."
    )

    y_pred = model.predict(
        X_test
    )

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise AttributeError(
            "The selected model does not support "
            "predict_proba()."
        )

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    test_results = {
        "Test Accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
        "Test Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "Test Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "Test F1": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "Test ROC-AUC": roc_auc_score(
            y_test,
            y_probability,
        ),
    }

    confusion = confusion_matrix(
        y_test,
        y_pred,
    )

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    print("\nFinal test results:")

    for metric, value in test_results.items():
        print(
            f"{metric}: {value:.4f}"
        )

    print("\nConfusion matrix:")
    print(confusion)

    print("\nClassification report:")
    print(report)

    return (
        test_results,
        confusion,
        report,
    )


def save_final_roc_curve(
    model,
    X_test,
    y_test,
    best_model_name,
):
    """
    Save the ROC curve for the selected model only.
    """

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.title(
        f"{best_model_name} – Final Test ROC Curve"
    )

    plt.savefig(
        FINAL_ROC_PATH,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    print(
        "Final ROC curve saved to:",
        FINAL_ROC_PATH,
    )


def save_final_test_results(
    best_model_name,
    selection_cv_f1,
    selection_cv_roc_auc,
    test_results,
    confusion,
    report,
):
    """
    Save the selected model's final test results.
    """

    with open(
        FINAL_RESULT_PATH,
        mode="w",
        encoding="utf-8",
    ) as result_file:

        result_file.write(
            "FINAL SELECTED MODEL RESULTS\n"
        )

        result_file.write(
            "============================\n\n"
        )

        result_file.write(
            f"Selected Model: "
            f"{best_model_name}\n"
        )

        result_file.write(
            "Selection Metric: "
            "Mean 5-Fold Cross-Validation F1\n"
        )

        result_file.write(
            f"Selected Model CV F1: "
            f"{selection_cv_f1:.6f}\n"
        )

        result_file.write(
            f"Selected Model CV ROC-AUC: "
            f"{selection_cv_roc_auc:.6f}\n\n"
        )

        result_file.write(
            "Methodology\n"
        )

        result_file.write(
            "-----------\n"
        )

        result_file.write(
            "SMOTE was applied only within each "
            "cross-validation training fold.\n"
        )

        result_file.write(
            "The model was selected without using "
            "the test set.\n"
        )

        result_file.write(
            "Only the selected model was evaluated "
            "on the untouched test set.\n\n"
        )

        result_file.write(
            "Final Test Set Results\n"
        )

        result_file.write(
            "----------------------\n"
        )

        for metric, value in test_results.items():
            result_file.write(
                f"{metric}: {value:.6f}\n"
            )

        result_file.write(
            "\nConfusion Matrix\n"
        )

        result_file.write(
            "----------------\n"
        )

        result_file.write(
            str(confusion)
        )

        result_file.write(
            "\n\nClassification Report\n"
        )

        result_file.write(
            "---------------------\n"
        )

        result_file.write(
            report
        )

    print(
        "Final test results saved to:",
        FINAL_RESULT_PATH,
    )


# ==========================================================
# MAIN COMPARISON FUNCTION
# ==========================================================

def compare_models():
    """
    Compare candidates using CV F1 and evaluate only the
    selected model on the untouched test set.
    """

    print("\n" + "=" * 70)
    print("STEP: MODEL COMPARISON AND FINAL EVALUATION")
    print("=" * 70)

    os.makedirs(
        "outputs",
        exist_ok=True,
    )

    os.makedirs(
        "models",
        exist_ok=True,
    )

    comparison_df = (
        create_comparison_table()
    )

    (
        best_model_name,
        comparison_df,
    ) = select_best_model(
        comparison_df
    )

    print("\nCROSS-VALIDATION MODEL COMPARISON")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    print(
        "\nComparison saved to:",
        COMPARISON_PATH,
    )

    print(
        "\nBest model selected:",
        best_model_name,
    )

    print(
        "Selection metric: "
        "Mean 5-Fold CV F1"
    )

    copy_selected_model(
        best_model_name
    )

    selected_model = load(
        BEST_MODEL_PATH
    )

    X_test, y_test = (
        load_test_data()
    )

    (
        test_results,
        confusion,
        report,
    ) = evaluate_selected_model(
        selected_model,
        X_test,
        y_test,
    )

    save_final_roc_curve(
        model=selected_model,
        X_test=X_test,
        y_test=y_test,
        best_model_name=best_model_name,
    )

    selected_row = comparison_df.loc[
        comparison_df["Selected"]
    ].iloc[0]

    save_final_test_results(
        best_model_name=best_model_name,
        selection_cv_f1=float(
            selected_row["CV F1"]
        ),
        selection_cv_roc_auc=float(
            selected_row["CV ROC-AUC"]
        ),
        test_results=test_results,
        confusion=confusion,
        report=report,
    )

    print("\n" + "=" * 70)
    print(
        "MODEL COMPARISON AND FINAL TEST "
        "EVALUATION COMPLETED"
    )
    print("=" * 70)

    return {
        "best_model_name": best_model_name,
        "comparison": comparison_df,
        "test_results": test_results,
        "best_model_path": BEST_MODEL_PATH,
    }


if __name__ == "__main__":
    compare_models()