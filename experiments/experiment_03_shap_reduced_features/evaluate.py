from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

# Use a non-interactive Matplotlib backend.
# This saves plots without requiring Tkinter/Tcl.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

def create_directories(*directories: Path) -> None:
    """
    Create all required experiment output directories.
    """

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def make_json_serialisable(value: Any) -> Any:
    """
    Convert NumPy and pandas objects into JSON-compatible values.
    """

    if isinstance(value, dict):
        return {
            str(key): make_json_serialisable(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_serialisable(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            make_json_serialisable(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, pd.Series):
        return value.tolist()

    return value


def save_json(data: dict[str, Any], file_path: Path) -> None:
    """
    Save a dictionary as formatted JSON.
    """

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(
            make_json_serialisable(data),
            file,
            indent=4,
        )


def calculate_test_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, Any]:
    """
    Calculate test-set metrics.
    """

    matrix = confusion_matrix(y_true, y_pred)

    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )

    specificity = (
        true_negative
        / (true_negative + false_positive)
        if true_negative + false_positive > 0
        else 0.0
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "specificity": specificity,
        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),
        "average_precision": average_precision_score(
            y_true,
            y_probability,
        ),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }

    return make_json_serialisable(metrics)


def save_classification_report(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    results_directory: Path,
) -> None:
    """
    Save text and CSV versions of the classification report.
    """

    text_report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "Low Burnout",
            "Moderate/High Burnout",
        ],
        digits=4,
        zero_division=0,
    )

    text_path = (
        results_directory
        / f"{model_name}_classification_report.txt"
    )

    text_path.write_text(
        text_report,
        encoding="utf-8",
    )

    dictionary_report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "Low Burnout",
            "Moderate/High Burnout",
        ],
        output_dict=True,
        zero_division=0,
    )

    report_dataframe = (
        pd.DataFrame(dictionary_report)
        .transpose()
    )

    report_dataframe.to_csv(
        results_directory
        / f"{model_name}_classification_report.csv"
    )


def save_confusion_matrix_plot(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plots_directory: Path,
) -> None:
    """
    Save confusion-matrix plot.
    """

    figure, axis = plt.subplots(figsize=(7, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=[
            "Low Burnout",
            "Moderate/High Burnout",
        ],
        values_format="d",
        ax=axis,
    )

    axis.set_title(
        f"{model_name.replace('_', ' ').title()} Confusion Matrix"
    )

    figure.tight_layout()

    figure.savefig(
        plots_directory
        / f"{model_name}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_roc_curve_plot(
    model_name: str,
    y_true: np.ndarray,
    y_probability: np.ndarray,
    plots_directory: Path,
) -> None:
    """
    Save ROC curve.
    """

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true,
        y_probability,
    )

    roc_auc = roc_auc_score(
        y_true,
        y_probability,
    )

    figure, axis = plt.subplots(figsize=(7, 6))

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {roc_auc:.4f}",
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(
        f"{model_name.replace('_', ' ').title()} ROC Curve"
    )
    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        plots_directory
        / f"{model_name}_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_precision_recall_curve_plot(
    model_name: str,
    y_true: np.ndarray,
    y_probability: np.ndarray,
    plots_directory: Path,
) -> None:
    """
    Save precision-recall curve.
    """

    precision_values, recall_values, _ = precision_recall_curve(
        y_true,
        y_probability,
    )

    average_precision = average_precision_score(
        y_true,
        y_probability,
    )

    figure, axis = plt.subplots(figsize=(7, 6))

    axis.plot(
        recall_values,
        precision_values,
        label=f"Average precision = {average_precision:.4f}",
    )

    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(
        f"{model_name.replace('_', ' ').title()} "
        "Precision-Recall Curve"
    )
    axis.legend(loc="lower left")
    axis.grid(alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        plots_directory
        / f"{model_name}_precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
    outputs_directory: Path,
) -> None:
    """
    Save test predictions.
    """

    predictions_dataframe = pd.DataFrame(
        {
            "actual_target": y_true,
            "predicted_target": y_pred,
            "burnout_probability": y_probability,
        }
    )

    predictions_dataframe.to_csv(
        outputs_directory
        / f"{model_name}_test_predictions.csv",
        index=False,
    )


def save_feature_list(
    feature_names: list[str],
    outputs_directory: Path,
) -> None:
    """
    Save encoded model feature names.
    """

    feature_dataframe = pd.DataFrame(
        {
            "feature_number": range(
                1,
                len(feature_names) + 1,
            ),
            "encoded_feature": feature_names,
        }
    )

    feature_dataframe.to_csv(
        outputs_directory
        / "final_encoded_feature_list.csv",
        index=False,
    )


def save_experiment_metadata(
    metadata: dict[str, Any],
    results_directory: Path,
) -> None:
    """
    Save complete experiment metadata.
    """

    save_json(
        metadata,
        results_directory / "experiment_metadata.json",
    )


def save_model_artifacts(
    fitted_preprocessor,
    fitted_models: dict[str, Any],
    models_directory: Path,
) -> None:
    """
    Save the preprocessor and trained models.

    These files should remain ignored by Git if they exceed GitHub limits.
    """

    joblib.dump(
        fitted_preprocessor,
        models_directory / "preprocessor.joblib",
        compress=3,
    )

    for model_name, model in fitted_models.items():
        joblib.dump(
            model,
            models_directory / f"{model_name}_model.joblib",
            compress=3,
        )


def evaluate_and_save_model(
    model_name: str,
    model,
    x_test_processed: np.ndarray,
    y_test: np.ndarray,
    cv_metrics: dict[str, float],
    results_directory: Path,
    plots_directory: Path,
    outputs_directory: Path,
) -> dict[str, Any]:
    """
    Evaluate one trained model and save all of its results.
    """

    y_pred = model.predict(x_test_processed)

    y_probability = model.predict_proba(
        x_test_processed
    )[:, 1]

    test_metrics = calculate_test_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    complete_metrics = {
        "model": model_name,
        **cv_metrics,
        **{
            f"test_{key}": value
            for key, value in test_metrics.items()
        },
    }

    save_json(
        complete_metrics,
        results_directory / f"{model_name}_metrics.json",
    )

    save_classification_report(
        model_name=model_name,
        y_true=y_test,
        y_pred=y_pred,
        results_directory=results_directory,
    )

    save_confusion_matrix_plot(
        model_name=model_name,
        y_true=y_test,
        y_pred=y_pred,
        plots_directory=plots_directory,
    )

    save_roc_curve_plot(
        model_name=model_name,
        y_true=y_test,
        y_probability=y_probability,
        plots_directory=plots_directory,
    )

    save_precision_recall_curve_plot(
        model_name=model_name,
        y_true=y_test,
        y_probability=y_probability,
        plots_directory=plots_directory,
    )

    save_predictions(
        model_name=model_name,
        y_true=y_test,
        y_pred=y_pred,
        y_probability=y_probability,
        outputs_directory=outputs_directory,
    )

    print(f"\n{model_name.replace('_', ' ').title()} test results")
    print("-" * 50)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-score:  {test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")

    return complete_metrics


def save_model_comparison(
    all_metrics: dict[str, dict[str, Any]],
    results_directory: Path,
) -> None:
    """
    Save a single comparison table for both models.
    """

    comparison_rows = []

    for model_name, metrics in all_metrics.items():
        comparison_rows.append(
            {
                "model": model_name,
                "cv_accuracy_mean": metrics.get(
                    "cv_accuracy_mean"
                ),
                "cv_precision_mean": metrics.get(
                    "cv_precision_mean"
                ),
                "cv_recall_mean": metrics.get(
                    "cv_recall_mean"
                ),
                "cv_f1_mean": metrics.get(
                    "cv_f1_mean"
                ),
                "cv_roc_auc_mean": metrics.get(
                    "cv_roc_auc_mean"
                ),
                "test_accuracy": metrics.get(
                    "test_accuracy"
                ),
                "test_precision": metrics.get(
                    "test_precision"
                ),
                "test_recall": metrics.get(
                    "test_recall"
                ),
                "test_f1_score": metrics.get(
                    "test_f1_score"
                ),
                "test_roc_auc": metrics.get(
                    "test_roc_auc"
                ),
                "test_average_precision": metrics.get(
                    "test_average_precision"
                ),
            }
        )

    comparison_dataframe = pd.DataFrame(
        comparison_rows
    )

    comparison_dataframe.to_csv(
        results_directory / "model_comparison.csv",
        index=False,
    )