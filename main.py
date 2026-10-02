"""
Main Machine Learning pipeline runner.

This file provides:

1. Baseline pipeline
2. Feature-engineering pipeline
3. Separate permanent storage for both pipelines
"""
from src.split_data import (
    load_resized_dataset,
    split_dataset,
)
from src.feature_engineering import (
    add_engineered_features,
)

from src.split_data import (
    split_dataset,
)

from src.encoding import (
    encode_data,
)

from src.random_forest import (
    train_random_forest,
)

from src.xgboost_model import (
    train_xgboost,
)

from src.compare_models import (
    compare_models,
)

from src.shap_analysis import (
    run_shap_analysis,
)

from src.pipeline_storage import (
    BASELINE_PIPELINE,
    FEATURE_ENGINEERED_PIPELINE,
    archive_pipeline_artifacts,
    clear_temporary_ml_files,
    print_saved_pipeline_locations,
)

def run_baseline_pipeline():
    """
    Run the baseline ML pipeline.
    """

    print("\n" + "=" * 80)
    print("RUNNING BASELINE MACHINE LEARNING PIPELINE")
    print("=" * 80)

    clear_temporary_ml_files()

    df = load_dataset()

    df_cleaned = clean_data(
        df
    )

    df_50k = resize_dataset_to_50k(
        df_cleaned
    )

    split_dataset(
        df_50k
    )

    encode_data()

    train_random_forest()

    train_xgboost()

    compare_models()

    run_shap_analysis()

    archive_pipeline_artifacts(
        BASELINE_PIPELINE
    )

    print("\n" + "=" * 80)
    print("BASELINE PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print(
        "\nBaseline best model:"
    )

    print(
        "models/baseline/"
        "best_model.joblib"
    )


def run_feature_engineering_pipeline():
    """
    Run the feature-engineered ML pipeline.
    """

    print("\n" + "=" * 80)
    print("RUNNING FEATURE ENGINEERING PIPELINE")
    print("=" * 80)

    clear_temporary_ml_files()

    df = load_dataset()

    df_cleaned = clean_data(
        df
    )

    df_50k = resize_dataset_to_50k(
        df_cleaned
    )

    df_feature_engineered = (
        add_engineered_features(
            df_50k
        )
    )

    split_dataset(
        df_feature_engineered
    )

    encode_data()

    train_random_forest()

    train_xgboost()

    compare_models()

    run_shap_analysis()

    archive_pipeline_artifacts(
        FEATURE_ENGINEERED_PIPELINE
    )

    print("\n" + "=" * 80)
    print(
        "FEATURE ENGINEERING PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 80)

    print(
        "\nFeature-engineered best model:"
    )

    print(
        "models/feature_engineered/"
        "best_model.joblib"
    )


def display_menu():
    """
    Display pipeline choices.
    """

    print("\n" + "=" * 80)
    print("MACHINE LEARNING PIPELINE MENU")
    print("=" * 80)

    print("\nChoose which pipeline to run:")

    print(
        "1. Baseline Pipeline"
    )

    print(
        "2. Feature Engineering Pipeline"
    )

    print(
        "3. Show saved pipeline locations"
    )

    print(
        "4. Run both pipelines"
    )


def run_both_pipelines():
    """
    Run the baseline and feature-engineering pipelines.
    """

    print(
        "\nThe baseline pipeline will run first."
    )

    run_baseline_pipeline()

    print(
        "\nThe feature-engineering pipeline "
        "will now run."
    )

    run_feature_engineering_pipeline()

    print("\n" + "=" * 80)
    print("BOTH PIPELINES COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print_saved_pipeline_locations()


def main():
    """
    Main program entry point.
    """

    display_menu()

    choice = input(
        "\nEnter 1, 2, 3 or 4: "
    ).strip()

    if choice == "1":
        run_baseline_pipeline()

    elif choice == "2":
        run_feature_engineering_pipeline()

    elif choice == "3":
        print_saved_pipeline_locations()

    elif choice == "4":
        run_both_pipelines()

    else:
        print(
            "\nInvalid choice. Please run again "
            "and enter 1, 2, 3 or 4."
        )


if __name__ == "__main__":
    main()