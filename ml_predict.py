"""
Runtime Machine Learning prediction for one employee.

This module loads the saved ML model, fitted OneHotEncoder, and preprocessing
metadata, then transforms one employee record exactly like the training data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from joblib import load

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_PIPELINE = "feature_engineered"
VALID_PIPELINES = {"baseline", "feature_engineered"}


def add_engineered_features(employee_df: pd.DataFrame) -> pd.DataFrame:
    """Add the same engineered features used during model training."""
    df = employee_df.copy()

    required_columns = {
        "stress_level",
        "anxiety_score",
        "depression_score",
        "work_hours_per_week",
        "overtime_hours",
        "work_life_balance",
        "manager_support",
        "sleep_hours",
        "physical_activity_days",
        "screen_time_hours",
    }

    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Cannot create engineered features because these source columns "
            f"are missing: {missing_columns}"
        )

    df["mental_health_index"] = (
        df["stress_level"] + df["anxiety_score"] + df["depression_score"]
    ) / 3
    df["workload_index"] = df["work_hours_per_week"] + df["overtime_hours"]
    df["work_life_support_index"] = (
        df["work_life_balance"] + df["manager_support"]
    ) / 2
    df["lifestyle_index"] = (
        df["sleep_hours"] + df["physical_activity_days"]
    ) / 2
    df["screen_workload_interaction"] = (
        df["screen_time_hours"] * df["work_hours_per_week"]
    )
    return df


def normalise_pipeline_name(pipeline_name: str) -> str:
    normalised = str(pipeline_name).strip().lower()
    if normalised not in VALID_PIPELINES:
        raise ValueError(
            f"Invalid pipeline name: {pipeline_name}. "
            f"Expected one of: {sorted(VALID_PIPELINES)}"
        )
    return normalised


def get_artifact_paths(pipeline_name: str) -> dict[str, Path]:
    pipeline_name = normalise_pipeline_name(pipeline_name)
    pipeline_dir = MODELS_DIR / pipeline_name
    return {
        "model": pipeline_dir / "best_model.joblib",
        "encoder": pipeline_dir / "onehot_encoder.joblib",
        "metadata": pipeline_dir / "ml_metadata.joblib",
    }


def validate_artifact_paths(artifact_paths: Mapping[str, Path]) -> None:
    missing = [str(path) for path in artifact_paths.values() if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Required ML runtime artifact(s) were not found:\n"
            f"{formatted}\n\nRun and archive the selected ML pipeline first."
        )


def load_ml_artifacts(
    pipeline_name: str = DEFAULT_PIPELINE,
) -> tuple[Any, Any, dict[str, Any]]:
    artifact_paths = get_artifact_paths(pipeline_name)
    validate_artifact_paths(artifact_paths)

    model = load(artifact_paths["model"])
    encoder = load(artifact_paths["encoder"])
    metadata = load(artifact_paths["metadata"])

    if not isinstance(metadata, dict):
        raise TypeError("ML metadata must be a dictionary.")

    return model, encoder, metadata


def validate_metadata(metadata: Mapping[str, Any]) -> None:
    required_keys = {
        "original_input_columns",
        "categorical_columns",
        "numerical_columns",
        "encoded_feature_columns",
        "feature_engineering_used",
        "prediction_threshold",
    }
    missing_keys = sorted(required_keys - set(metadata))
    if missing_keys:
        raise ValueError(f"ML metadata is missing required key(s): {missing_keys}")


def coerce_numerical_columns(
    employee_df: pd.DataFrame,
    numerical_columns: list[str],
) -> pd.DataFrame:
    df = employee_df.copy()
    for column in numerical_columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.isna().any():
            original_value = df.at[df.index[0], column]
            raise ValueError(
                f"Column '{column}' must contain a valid numeric value. "
                f"Received: {original_value!r}"
            )
        df[column] = converted.astype("float64")
    return df


def prepare_original_employee_frame(
    employee_data: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    if not isinstance(employee_data, Mapping):
        raise TypeError("employee_data must be a dictionary-like mapping.")
    if not employee_data:
        raise ValueError("employee_data cannot be empty.")

    validate_metadata(metadata)
    original_input_columns = list(metadata["original_input_columns"])
    employee_df = pd.DataFrame([dict(employee_data)])

    if bool(metadata["feature_engineering_used"]):
        employee_df = add_engineered_features(employee_df)

    missing_columns = [
        column for column in original_input_columns if column not in employee_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Employee input is missing required field(s): "
            f"{missing_columns}"
        )

    employee_df = employee_df[original_input_columns].copy()

    if employee_df.isna().any().any():
        columns = employee_df.columns[employee_df.isna().any()].tolist()
        raise ValueError(f"Employee input contains missing value(s) in: {columns}")

    employee_df = coerce_numerical_columns(
        employee_df,
        list(metadata["numerical_columns"]),
    )

    for column in list(metadata["categorical_columns"]):
        employee_df[column] = employee_df[column].astype(str)

    return employee_df


def encode_employee_record(
    employee_df: pd.DataFrame,
    encoder: Any,
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    categorical_columns = list(metadata["categorical_columns"])
    numerical_columns = list(metadata["numerical_columns"])
    expected_encoded_columns = list(metadata["encoded_feature_columns"])

    numerical_df = employee_df[numerical_columns].copy().reset_index(drop=True)

    if categorical_columns:
        encoded_array = encoder.transform(employee_df[categorical_columns])
        encoded_column_names = encoder.get_feature_names_out(
            categorical_columns
        ).tolist()
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoded_column_names,
            index=employee_df.index,
        ).reset_index(drop=True)
    else:
        encoded_df = pd.DataFrame(index=employee_df.index).reset_index(drop=True)

    final_df = pd.concat([numerical_df, encoded_df], axis=1)

    unexpected_columns = sorted(
        set(final_df.columns) - set(expected_encoded_columns)
    )
    if unexpected_columns:
        raise ValueError(
            "Runtime preprocessing created unexpected column(s): "
            f"{unexpected_columns}"
        )

    for column in expected_encoded_columns:
        if column not in final_df.columns:
            final_df[column] = 0.0

    final_df = final_df[expected_encoded_columns].astype("float64")

    if final_df.isna().any().any():
        raise ValueError("Encoded employee record contains missing values.")

    return final_df


def predict_employee_burnout(
    employee_data: Mapping[str, Any],
    pipeline_name: str = DEFAULT_PIPELINE,
) -> dict[str, Any]:
    """Predict burnout risk for one employee and return a fusion-ready result."""
    pipeline_name = normalise_pipeline_name(pipeline_name)

    try:
        model, encoder, metadata = load_ml_artifacts(pipeline_name)
        employee_df = prepare_original_employee_frame(employee_data, metadata)
        encoded_employee_df = encode_employee_record(
            employee_df,
            encoder,
            metadata,
        )

        if not hasattr(model, "predict_proba"):
            raise AttributeError("The saved ML model does not support predict_proba().")

        probabilities = model.predict_proba(encoded_employee_df)
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_class = metadata.get("positive_class", 1)

        if positive_class not in classes:
            raise ValueError(
                f"Positive class {positive_class!r} was not found in model classes: {classes}"
            )

        ml_score = float(probabilities[0][classes.index(positive_class)])
        if not np.isfinite(ml_score):
            raise ValueError("The model returned a non-finite probability.")

        ml_score = float(np.clip(ml_score, 0.0, 1.0))
        threshold = float(metadata.get("prediction_threshold", 0.50))
        prediction = int(ml_score >= threshold)

        return {
            "ml_available": True,
            "ml_score": ml_score,
            "ml_percentage": ml_score * 100,
            "prediction": prediction,
            "risk_label": (
                "Elevated burnout risk" if prediction == 1 else "Low burnout risk"
            ),
            "threshold": threshold,
            "pipeline_name": pipeline_name,
            "model_type": type(model).__name__,
            "feature_count": int(encoded_employee_df.shape[1]),
            "error": None,
        }

    except Exception as exc:
        return {
            "ml_available": False,
            "ml_score": None,
            "ml_percentage": None,
            "prediction": None,
            "risk_label": "ML prediction unavailable",
            "threshold": None,
            "pipeline_name": pipeline_name,
            "model_type": None,
            "feature_count": None,
            "error": str(exc),
        }


run_ml_prediction = predict_employee_burnout


if __name__ == "__main__":
    print(
        "Import predict_employee_burnout() and pass one employee dictionary.\n"
        "Example:\n"
        "from src.ml_predict import predict_employee_burnout\n"
        "result = predict_employee_burnout(employee_data={...})\n"
        "print(result)"
    )