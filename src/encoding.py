"""
One-hot encoding and preprocessing metadata saving.

This module:

1. Loads the training and test feature files.
2. Detects categorical and numerical columns.
3. Fits OneHotEncoder only on the training data.
4. Transforms both training and test data.
5. Saves the encoded datasets.
6. Saves the fitted encoder.
7. Saves metadata required for future runtime prediction.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.preprocessing import OneHotEncoder


# ==========================================================
# PROJECT PATHS
# ==========================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

X_TRAIN_PATH = OUTPUTS_DIR / "X_train.csv"
X_TEST_PATH = OUTPUTS_DIR / "X_test.csv"

ENCODED_TRAIN_PATH = OUTPUTS_DIR / "X_train_encoded.csv"
ENCODED_TEST_PATH = OUTPUTS_DIR / "X_test_encoded.csv"

ENCODER_PATH = MODELS_DIR / "onehot_encoder.joblib"

METADATA_JOBLIB_PATH = MODELS_DIR / "ml_metadata.joblib"
METADATA_JSON_PATH = MODELS_DIR / "ml_metadata.json"


# ==========================================================
# SETTINGS
# ==========================================================

TARGET_COLUMN = "target"
POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0
PREDICTION_THRESHOLD = 0.50


# ==========================================================
# VALIDATION
# ==========================================================

def validate_input_files() -> None:
    """
    Confirm that the training and test feature files exist.
    """

    missing_files = []

    if not X_TRAIN_PATH.exists():
        missing_files.append(str(X_TRAIN_PATH))

    if not X_TEST_PATH.exists():
        missing_files.append(str(X_TEST_PATH))

    if missing_files:
        formatted_files = "\n".join(
            f"- {file_path}"
            for file_path in missing_files
        )

        raise FileNotFoundError(
            "The following input files are missing:\n"
            f"{formatted_files}\n\n"
            "Run the data splitting step before encoding."
        )


def validate_train_test_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """
    Confirm that training and test datasets have matching
    original input columns.
    """

    train_columns = list(X_train.columns)
    test_columns = list(X_test.columns)

    if train_columns != test_columns:
        train_only = sorted(
            set(train_columns) - set(test_columns)
        )

        test_only = sorted(
            set(test_columns) - set(train_columns)
        )

        raise ValueError(
            "Training and test columns do not match.\n"
            f"Columns only in training: {train_only}\n"
            f"Columns only in testing: {test_only}"
        )


def validate_missing_values(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """
    Stop encoding if missing values remain.

    Missing-value handling will later be centralised in the
    preprocessing stage.
    """

    train_missing = int(
        X_train.isna().sum().sum()
    )

    test_missing = int(
        X_test.isna().sum().sum()
    )

    if train_missing > 0 or test_missing > 0:
        raise ValueError(
            "Missing values were found before encoding.\n"
            f"Training missing cells: {train_missing}\n"
            f"Testing missing cells: {test_missing}\n\n"
            "Clean or impute missing values before encoding."
        )


# ==========================================================
# COLUMN DETECTION
# ==========================================================

def detect_column_types(
    X_train: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Detect categorical and numerical columns.
    """

    categorical_columns = (
        X_train.select_dtypes(
            include=[
                "object",
                "string",
                "category",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    numerical_columns = [
        column
        for column in X_train.columns
        if column not in categorical_columns
    ]

    return (
        categorical_columns,
        numerical_columns,
    )


# ==========================================================
# ENCODER
# ==========================================================

def build_encoder() -> OneHotEncoder:
    """
    Build the one-hot encoder.
    """

    return OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype="float64",
    )


def encode_categorical_columns(
    encoder: OneHotEncoder,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Fit the encoder on training data and transform both sets.
    """

    if not categorical_columns:
        return (
            pd.DataFrame(index=X_train.index),
            pd.DataFrame(index=X_test.index),
            [],
        )

    train_encoded_array = encoder.fit_transform(
        X_train[categorical_columns]
    )

    test_encoded_array = encoder.transform(
        X_test[categorical_columns]
    )

    encoded_column_names = (
        encoder.get_feature_names_out(
            categorical_columns
        )
        .tolist()
    )

    train_encoded_df = pd.DataFrame(
        train_encoded_array,
        columns=encoded_column_names,
        index=X_train.index,
    )

    test_encoded_df = pd.DataFrame(
        test_encoded_array,
        columns=encoded_column_names,
        index=X_test.index,
    )

    return (
        train_encoded_df,
        test_encoded_df,
        encoded_column_names,
    )


def combine_encoded_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_columns: list[str],
    train_encoded_df: pd.DataFrame,
    test_encoded_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine numerical columns with encoded categorical data.
    """

    train_numerical_df = (
        X_train[numerical_columns]
        .copy()
        .reset_index(drop=True)
    )

    test_numerical_df = (
        X_test[numerical_columns]
        .copy()
        .reset_index(drop=True)
    )

    train_encoded_df = (
        train_encoded_df
        .copy()
        .reset_index(drop=True)
    )

    test_encoded_df = (
        test_encoded_df
        .copy()
        .reset_index(drop=True)
    )

    X_train_final = pd.concat(
        [
            train_numerical_df,
            train_encoded_df,
        ],
        axis=1,
    )

    X_test_final = pd.concat(
        [
            test_numerical_df,
            test_encoded_df,
        ],
        axis=1,
    )

    if list(X_train_final.columns) != list(
        X_test_final.columns
    ):
        raise ValueError(
            "Encoded training and test columns do not match."
        )

    return (
        X_train_final,
        X_test_final,
    )


# ==========================================================
# PIPELINE DETECTION
# ==========================================================

def detect_pipeline_type(
    original_columns: list[str],
) -> tuple[str, bool]:
    """
    Determine whether engineered features are present.
    """

    engineered_features = {
        "mental_health_index",
        "workload_index",
        "work_life_support_index",
        "lifestyle_index",
        "screen_workload_interaction",
    }

    feature_engineering_used = (
        engineered_features.issubset(
            set(original_columns)
        )
    )

    if feature_engineering_used:
        pipeline_type = "feature_engineered"
    else:
        pipeline_type = "baseline"

    return (
        pipeline_type,
        feature_engineering_used,
    )


# ==========================================================
# METADATA
# ==========================================================

def build_metadata(
    X_train: pd.DataFrame,
    X_train_final: pd.DataFrame,
    categorical_columns: list[str],
    numerical_columns: list[str],
    encoded_column_names: list[str],
    encoder: OneHotEncoder,
) -> dict:
    """
    Build preprocessing metadata for runtime prediction.
    """

    original_input_columns = list(
        X_train.columns
    )

    encoded_feature_columns = list(
        X_train_final.columns
    )

    (
        pipeline_type,
        feature_engineering_used,
    ) = detect_pipeline_type(
        original_input_columns
    )

    categorical_values = {}

    if categorical_columns:
        for column, categories in zip(
            categorical_columns,
            encoder.categories_,
        ):
            categorical_values[column] = [
                str(value)
                for value in categories.tolist()
            ]

    metadata = {
        "metadata_version": "1.0",
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "pipeline_type": pipeline_type,
        "feature_engineering_used": (
            feature_engineering_used
        ),
        "target_column": TARGET_COLUMN,
        "negative_class": NEGATIVE_CLASS,
        "positive_class": POSITIVE_CLASS,
        "prediction_threshold": (
            PREDICTION_THRESHOLD
        ),
        "encoder_type": "OneHotEncoder",
        "encoder_settings": {
            "handle_unknown": "ignore",
            "sparse_output": False,
            "dtype": "float64",
        },
        "original_input_columns": (
            original_input_columns
        ),
        "categorical_columns": (
            categorical_columns
        ),
        "numerical_columns": (
            numerical_columns
        ),
        "encoded_categorical_columns": (
            encoded_column_names
        ),
        "encoded_feature_columns": (
            encoded_feature_columns
        ),
        "categorical_values_seen_during_training": (
            categorical_values
        ),
        "original_feature_count": len(
            original_input_columns
        ),
        "categorical_feature_count": len(
            categorical_columns
        ),
        "numerical_feature_count": len(
            numerical_columns
        ),
        "encoded_feature_count": len(
            encoded_feature_columns
        ),
    }

    return metadata


def save_metadata(
    metadata: dict,
) -> None:
    """
    Save metadata in joblib and readable JSON formats.
    """

    dump(
        metadata,
        METADATA_JOBLIB_PATH,
    )

    with METADATA_JSON_PATH.open(
        mode="w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "Metadata joblib saved to:",
        METADATA_JOBLIB_PATH,
    )

    print(
        "Metadata JSON saved to:",
        METADATA_JSON_PATH,
    )


# ==========================================================
# MAIN ENCODING FUNCTION
# ==========================================================

def encode_data():
    """
    Run one-hot encoding and save preprocessing metadata.
    """

    print("\n" + "=" * 70)
    print("STARTING ONE-HOT ENCODING")
    print("=" * 70)

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_input_files()

    X_train = pd.read_csv(
        X_TRAIN_PATH
    )

    X_test = pd.read_csv(
        X_TEST_PATH
    )

    print(
        "\nTraining shape:",
        X_train.shape,
    )

    print(
        "Testing shape:",
        X_test.shape,
    )

    validate_train_test_columns(
        X_train,
        X_test,
    )

    validate_missing_values(
        X_train,
        X_test,
    )

    (
        categorical_columns,
        numerical_columns,
    ) = detect_column_types(
        X_train
    )

    print("\nCategorical columns:")
    print(categorical_columns)

    print("\nNumerical columns:")
    print(numerical_columns)

    encoder = build_encoder()

    (
        train_encoded_df,
        test_encoded_df,
        encoded_column_names,
    ) = encode_categorical_columns(
        encoder=encoder,
        X_train=X_train,
        X_test=X_test,
        categorical_columns=categorical_columns,
    )

    (
        X_train_final,
        X_test_final,
    ) = combine_encoded_features(
        X_train=X_train,
        X_test=X_test,
        numerical_columns=numerical_columns,
        train_encoded_df=train_encoded_df,
        test_encoded_df=test_encoded_df,
    )

    print(
        "\nEncoded training shape:",
        X_train_final.shape,
    )

    print(
        "Encoded testing shape:",
        X_test_final.shape,
    )

    X_train_final.to_csv(
        ENCODED_TRAIN_PATH,
        index=False,
    )

    X_test_final.to_csv(
        ENCODED_TEST_PATH,
        index=False,
    )

    dump(
        encoder,
        ENCODER_PATH,
    )

    metadata = build_metadata(
        X_train=X_train,
        X_train_final=X_train_final,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        encoded_column_names=encoded_column_names,
        encoder=encoder,
    )

    save_metadata(
        metadata
    )

    print("\nFiles saved successfully:")

    print(
        ENCODED_TRAIN_PATH
    )

    print(
        ENCODED_TEST_PATH
    )

    print(
        ENCODER_PATH
    )

    print(
        METADATA_JOBLIB_PATH
    )

    print(
        METADATA_JSON_PATH
    )

    print("\nPipeline type detected:")
    print(
        metadata["pipeline_type"]
    )

    print("=" * 70)

    return (
        X_train_final,
        X_test_final,
        metadata,
    )


if __name__ == "__main__":
    encode_data()