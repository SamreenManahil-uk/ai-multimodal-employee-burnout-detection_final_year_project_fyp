from __future__ import annotations

import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    EXPERIMENT_NAME,
    FEATURES_TO_REMOVE,
    MODELS_DIR,
    OUTPUTS_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from evaluate import (
    create_directories,
    evaluate_and_save_model,
    save_experiment_metadata,
    save_feature_list,
    save_model_artifacts,
    save_model_comparison,
)
from feature_selection import (
    prepare_feature_engineered_reduced_dataset,
)
from train_models import (
    build_models,
    create_preprocessor,
    fit_preprocessor_and_resample,
    identify_feature_types,
    run_cross_validation,
    train_final_models,
)


def print_heading(title: str) -> None:
    """
    Print a formatted console heading.
    """

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    """
    Run Experiment 03 from beginning to end.
    """

    experiment_start = datetime.now()

    print_heading(EXPERIMENT_NAME)

    create_directories(
        MODELS_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        OUTPUTS_DIR,
    )

    # ======================================================
    # 1. Prepare reduced feature-engineered dataset
    # ======================================================

    print_heading("1. DATA PREPARATION")

    (
        dataframe,
        removed_features,
        missing_removal_features,
    ) = prepare_feature_engineered_reduced_dataset(
        DATA_PATH
    )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not created."
        )

    x = dataframe.drop(columns=[TARGET_COLUMN])
    y = dataframe[TARGET_COLUMN].astype(int)

    original_input_feature_count = x.shape[1]

    print(f"\nInput feature count before encoding: {x.shape[1]}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")

    # ======================================================
    # 2. Train-test split
    # ======================================================

    print_heading("2. STRATIFIED TRAIN-TEST SPLIT")

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"X_train shape: {x_train.shape}")
    print(f"X_test shape:  {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting target distribution:")
    print(y_test.value_counts().sort_index())

    # ======================================================
    # 3. Preprocessing
    # ======================================================

    print_heading("3. PREPROCESSING")

    (
        categorical_columns,
        numerical_columns,
    ) = identify_feature_types(x_train)

    print(
        f"Categorical variables ({len(categorical_columns)}): "
        f"{categorical_columns}"
    )

    print(
        f"\nNumerical variables ({len(numerical_columns)}): "
        f"{numerical_columns}"
    )

    preprocessor = create_preprocessor(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
    )

    # ======================================================
    # 4. Models and cross-validation
    # ======================================================

    print_heading("4. FIVE-FOLD CROSS-VALIDATION")

    models = build_models()
    cross_validation_results = {}

    for model_name, model in models.items():
        cross_validation_results[model_name] = (
            run_cross_validation(
                model_name=model_name,
                model=model,
                preprocessor=preprocessor,
                x_train=x_train,
                y_train=y_train,
            )
        )

    # ======================================================
    # 5. Fit preprocessor and apply SMOTE
    # ======================================================

    print_heading("5. FINAL PREPROCESSING AND SMOTE")

    (
        fitted_preprocessor,
        x_train_resampled,
        x_test_processed,
        y_train_resampled,
        _,
        encoded_feature_names,
    ) = fit_preprocessor_and_resample(
        preprocessor=preprocessor,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
    )

    print(
        f"\nEncoded feature count: "
        f"{len(encoded_feature_names)}"
    )

    print(
        f"Resampled training shape: "
        f"{x_train_resampled.shape}"
    )

    print(
        f"Processed test shape: "
        f"{x_test_processed.shape}"
    )

    save_feature_list(
        feature_names=encoded_feature_names,
        outputs_directory=OUTPUTS_DIR,
    )

    # Save the reduced unencoded dataset used by this experiment
    dataframe.to_csv(
        OUTPUTS_DIR
        / "shap_reduced_feature_engineered_dataset.csv",
        index=False,
    )

    # Save train/test row-level inputs separately
    x_train.assign(
        target=y_train.values
    ).to_csv(
        OUTPUTS_DIR / "training_data_before_encoding.csv",
        index=False,
    )

    x_test.assign(
        target=y_test.values
    ).to_csv(
        OUTPUTS_DIR / "testing_data_before_encoding.csv",
        index=False,
    )

    # ======================================================
    # 6. Train final models
    # ======================================================

    print_heading("6. FINAL MODEL TRAINING")

    fitted_models = train_final_models(
        models=models,
        x_train_resampled=x_train_resampled,
        y_train_resampled=y_train_resampled,
    )

    # ======================================================
    # 7. Evaluation
    # ======================================================

    print_heading("7. TEST-SET EVALUATION")

    all_model_metrics = {}

    for model_name, fitted_model in fitted_models.items():
        all_model_metrics[model_name] = (
            evaluate_and_save_model(
                model_name=model_name,
                model=fitted_model,
                x_test_processed=x_test_processed,
                y_test=np.asarray(y_test),
                cv_metrics=cross_validation_results[
                    model_name
                ],
                results_directory=RESULTS_DIR,
                plots_directory=PLOTS_DIR,
                outputs_directory=OUTPUTS_DIR,
            )
        )

    save_model_comparison(
        all_metrics=all_model_metrics,
        results_directory=RESULTS_DIR,
    )

    # ======================================================
    # 8. Save models and metadata
    # ======================================================

    print_heading("8. SAVING EXPERIMENT ARTIFACTS")

    save_model_artifacts(
        fitted_preprocessor=fitted_preprocessor,
        fitted_models=fitted_models,
        models_directory=MODELS_DIR,
    )

    experiment_end = datetime.now()
    duration_seconds = (
        experiment_end - experiment_start
    ).total_seconds()

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "started_at": experiment_start.isoformat(),
        "completed_at": experiment_end.isoformat(),
        "duration_seconds": duration_seconds,
        "dataset_path": str(DATA_PATH),
        "dataset_rows_used": int(len(dataframe)),
        "input_features_before_encoding": int(
            original_input_feature_count
        ),
        "encoded_features_after_preprocessing": int(
            len(encoded_feature_names)
        ),
        "training_rows_before_smote": int(len(x_train)),
        "testing_rows": int(len(x_test)),
        "training_rows_after_smote": int(
            len(x_train_resampled)
        ),
        "target_column": TARGET_COLUMN,
        "requested_shap_features_to_remove": (
            FEATURES_TO_REMOVE
        ),
        "successfully_removed_features": (
            removed_features
        ),
        "requested_features_not_found": (
            missing_removal_features
        ),
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "encoded_feature_names": encoded_feature_names,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    save_experiment_metadata(
        metadata=metadata,
        results_directory=RESULTS_DIR,
    )

    print("\nAll Experiment 03 files were saved inside:")
    print(f"  Models:  {MODELS_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Plots:   {PLOTS_DIR}")
    print(f"  Outputs: {OUTPUTS_DIR}")

    print_heading("EXPERIMENT 03 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\nExperiment failed.")
        print(f"Error: {error}")
        print("\nFull error details:")
        traceback.print_exc()
        sys.exit(1)