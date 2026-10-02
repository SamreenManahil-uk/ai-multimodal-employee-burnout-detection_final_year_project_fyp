"""
Pipeline artifact management.

This module prevents the baseline and feature-engineering
pipelines from overwriting each other's saved models and outputs.

The existing ML scripts can continue writing temporary files to:

    models/
    outputs/

After a pipeline finishes, its files are copied into:

    models/baseline/
    outputs/baseline/

or:

    models/feature_engineered/
    outputs/feature_engineered/
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable


# ==========================================================
# PROJECT DIRECTORIES
# ==========================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ==========================================================
# VALID PIPELINE NAMES
# ==========================================================

BASELINE_PIPELINE = "baseline"
FEATURE_ENGINEERED_PIPELINE = "feature_engineered"

VALID_PIPELINES = {
    BASELINE_PIPELINE,
    FEATURE_ENGINEERED_PIPELINE,
}


# ==========================================================
# ML MODEL FILES
# ==========================================================

MODEL_FILES = [
    "random_forest_model.joblib",
    "xgboost_model.joblib",
    "best_model.joblib",
    "onehot_encoder.joblib",
    "ml_metadata.joblib",
    "ml_metadata.json",
]


# ==========================================================
# ML OUTPUT FILES
# ==========================================================

OUTPUT_FILES = [
    # Dataset stages
    "cleaned_dataset.csv",
    "resized_50k_dataset.csv",
    "feature_engineered_50k_dataset.csv",

    # Train/test split
    "X_train.csv",
    "X_test.csv",
    "y_train.csv",
    "y_test.csv",

    # Encoded data
    "X_train_encoded.csv",
    "X_test_encoded.csv",

    # SMOTE data
    "X_train_smote.csv",
    "y_train_smote.csv",

    # Random Forest outputs
    "random_forest_results.txt",
    "random_forest_roc_curve.png",
    "random_forest_feature_importance.csv",

    # XGBoost outputs
    "xgboost_results.txt",
    "xgboost_roc_curve.png",
    "xgboost_feature_importance.csv",

    # Model comparison
    "model_comparison.csv",

    # SHAP outputs
    "shap_summary_plot.png",
    "shap_bar_plot.png",
    "shap_waterfall_plot.png",
    "shap_feature_importance.csv",
]


def validate_pipeline_name(
    pipeline_name: str,
) -> str:
    """
    Validate and normalise a pipeline name.
    """

    normalised_name = str(
        pipeline_name
    ).strip().lower()

    if normalised_name not in VALID_PIPELINES:
        raise ValueError(
            f"Invalid pipeline name: {pipeline_name}. "
            f"Expected one of: "
            f"{sorted(VALID_PIPELINES)}"
        )

    return normalised_name


def create_project_directories() -> None:
    """
    Create the main models and outputs folders.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def remove_files(
    directory: Path,
    filenames: Iterable[str],
) -> None:
    """
    Remove specified files from a directory.

    Missing files are ignored.
    """

    for filename in filenames:
        file_path = directory / filename

        if file_path.exists() and file_path.is_file():
            file_path.unlink()


def clear_temporary_ml_files() -> None:
    """
    Remove temporary root-level ML files before a new pipeline run.

    This does not remove:

    - models/baseline
    - models/feature_engineered
    - outputs/baseline
    - outputs/feature_engineered
    - Computer Vision outputs
    - NLP outputs not listed in OUTPUT_FILES
    """

    create_project_directories()

    remove_files(
        directory=MODELS_DIR,
        filenames=MODEL_FILES,
    )

    remove_files(
        directory=OUTPUTS_DIR,
        filenames=OUTPUT_FILES,
    )

    print("\nTemporary ML files cleared.")
    print(
        "Previously archived pipeline folders "
        "were not deleted."
    )


def recreate_directory(
    directory: Path,
) -> None:
    """
    Delete and recreate one pipeline archive directory.
    """

    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def copy_existing_files(
    source_directory: Path,
    destination_directory: Path,
    filenames: Iterable[str],
) -> list[str]:
    """
    Copy existing files and return their filenames.
    """

    copied_files: list[str] = []

    for filename in filenames:
        source_path = source_directory / filename

        if not source_path.exists():
            continue

        if not source_path.is_file():
            continue

        destination_path = (
            destination_directory / filename
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        copied_files.append(filename)

    return copied_files


def build_manifest(
    pipeline_name: str,
    copied_model_files: list[str],
    copied_output_files: list[str],
) -> dict:
    """
    Build a record describing the archived pipeline.
    """

    return {
        "pipeline_name": pipeline_name,
        "archived_at": (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "model_directory": str(
            MODELS_DIR / pipeline_name
        ),
        "output_directory": str(
            OUTPUTS_DIR / pipeline_name
        ),
        "model_files": copied_model_files,
        "output_files": copied_output_files,
        "best_model_path": str(
            MODELS_DIR
            / pipeline_name
            / "best_model.joblib"
        ),
        "encoder_path": str(
            MODELS_DIR
            / pipeline_name
            / "onehot_encoder.joblib"
        ),
        "metadata_joblib_path": str(
            MODELS_DIR
            / pipeline_name
            / "ml_metadata.joblib"
        ),
        "metadata_json_path": str(
            MODELS_DIR
            / pipeline_name
            / "ml_metadata.json"
        ),
    }


def save_manifest(
    manifest: dict,
    model_destination: Path,
    output_destination: Path,
) -> None:
    """
    Save the pipeline manifest in both archive folders.
    """

    for destination in (
        model_destination,
        output_destination,
    ):
        manifest_path = (
            destination / "pipeline_manifest.json"
        )

        with manifest_path.open(
            mode="w",
            encoding="utf-8",
        ) as manifest_file:
            json.dump(
                manifest,
                manifest_file,
                indent=4,
                ensure_ascii=False,
            )


def archive_pipeline_artifacts(
    pipeline_name: str,
) -> dict:
    """
    Copy the completed pipeline files into a permanent
    pipeline-specific directory.

    Parameters
    ----------
    pipeline_name:
        Either:

        - baseline
        - feature_engineered

    Returns
    -------
    dict
        Archive manifest.
    """

    pipeline_name = validate_pipeline_name(
        pipeline_name
    )

    create_project_directories()

    model_destination = (
        MODELS_DIR / pipeline_name
    )

    output_destination = (
        OUTPUTS_DIR / pipeline_name
    )

    # Remove the previous archive for this same pipeline.
    # The other pipeline archive is not affected.
    recreate_directory(
        model_destination
    )

    recreate_directory(
        output_destination
    )

    copied_model_files = copy_existing_files(
        source_directory=MODELS_DIR,
        destination_directory=model_destination,
        filenames=MODEL_FILES,
    )

    copied_output_files = copy_existing_files(
        source_directory=OUTPUTS_DIR,
        destination_directory=output_destination,
        filenames=OUTPUT_FILES,
    )

    manifest = build_manifest(
        pipeline_name=pipeline_name,
        copied_model_files=copied_model_files,
        copied_output_files=copied_output_files,
    )

    save_manifest(
        manifest=manifest,
        model_destination=model_destination,
        output_destination=output_destination,
    )

    print("\n" + "=" * 70)
    print("PIPELINE FILES ARCHIVED")
    print("=" * 70)

    print(
        f"Pipeline: {pipeline_name}"
    )

    print(
        f"Models saved in: "
        f"{model_destination}"
    )

    print(
        f"Outputs saved in: "
        f"{output_destination}"
    )

    print(
        f"Model files copied: "
        f"{len(copied_model_files)}"
    )

    print(
        f"Output files copied: "
        f"{len(copied_output_files)}"
    )

    if "best_model.joblib" not in copied_model_files:
        print(
            "\nWarning: best_model.joblib "
            "was not found."
        )

    if "onehot_encoder.joblib" not in copied_model_files:
        print(
            "\nWarning: onehot_encoder.joblib "
            "was not found."
        )

    if "ml_metadata.joblib" not in copied_model_files:
        print(
            "\nWarning: ml_metadata.joblib "
            "was not found."
        )

    if "ml_metadata.json" not in copied_model_files:
        print(
            "\nWarning: ml_metadata.json "
            "was not found."
        )

    print("=" * 70)

    return manifest


def print_saved_pipeline_locations() -> None:
    """
    Print the expected permanent pipeline artifact locations.
    """

    print("\nPermanent pipeline locations:")

    print("\nBaseline pipeline:")

    print(
        "Best model: "
        f"{MODELS_DIR / BASELINE_PIPELINE / 'best_model.joblib'}"
    )

    print(
        "Encoder: "
        f"{MODELS_DIR / BASELINE_PIPELINE / 'onehot_encoder.joblib'}"
    )

    print(
        "Metadata joblib: "
        f"{MODELS_DIR / BASELINE_PIPELINE / 'ml_metadata.joblib'}"
    )

    print(
        "Metadata JSON: "
        f"{MODELS_DIR / BASELINE_PIPELINE / 'ml_metadata.json'}"
    )

    print("\nFeature-engineered pipeline:")

    print(
        "Best model: "
        f"{MODELS_DIR / FEATURE_ENGINEERED_PIPELINE / 'best_model.joblib'}"
    )

    print(
        "Encoder: "
        f"{MODELS_DIR / FEATURE_ENGINEERED_PIPELINE / 'onehot_encoder.joblib'}"
    )

    print(
        "Metadata joblib: "
        f"{MODELS_DIR / FEATURE_ENGINEERED_PIPELINE / 'ml_metadata.joblib'}"
    )

    print(
        "Metadata JSON: "
        f"{MODELS_DIR / FEATURE_ENGINEERED_PIPELINE / 'ml_metadata.json'}"
    )


if __name__ == "__main__":
    print_saved_pipeline_locations()