from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from config import (
    ENGINEERED_FEATURES,
    FEATURES_TO_REMOVE,
    LEAKAGE_COLUMNS,
    RANDOM_STATE,
    SAMPLE_SIZE,
    TARGET_COLUMN,
)


def load_dataset(file_path) -> pd.DataFrame:
    """
    Load the original workplace burnout dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{file_path}\n\n"
            "Place tech_mental_health_burnout.csv inside data/raw/."
        )

    dataframe = pd.read_csv(file_path)

    print("\nDataset loaded successfully.")
    print(f"Original shape: {dataframe.shape}")

    return dataframe


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """
    Confirm that all columns required for target creation and feature
    engineering are available.
    """

    required_columns = {
        "burnout_level",
        "burnout_score",
        "stress_level",
        "anxiety_score",
        "depression_score",
        "work_hours_per_week",
        "overtime_hours",
        "meetings_per_day",
        "deadlines_missed",
        "job_satisfaction",
        "manager_support",
        "work_life_balance",
        "social_support_score",
        "sleep_hours",
        "physical_activity_days",
        "screen_time_hours",
        "caffeine_intake",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "The following required columns are missing from the dataset:\n"
            f"{sorted(missing_columns)}"
        )


def clean_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names.
    """

    dataframe = dataframe.copy()

    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return dataframe


def create_binary_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert burnout_level into a binary target.

    Low      -> 0
    Moderate -> 1
    High     -> 1
    """

    dataframe = dataframe.copy()

    normalised_levels = (
        dataframe["burnout_level"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    target_mapping = {
        "low": 0,
        "moderate": 1,
        "high": 1,
    }

    dataframe[TARGET_COLUMN] = normalised_levels.map(target_mapping)

    invalid_count = int(dataframe[TARGET_COLUMN].isna().sum())

    if invalid_count > 0:
        invalid_values = sorted(
            dataframe.loc[
                dataframe[TARGET_COLUMN].isna(),
                "burnout_level",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{invalid_count} rows have an invalid burnout_level value.\n"
            f"Unexpected values: {invalid_values}"
        )

    dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].astype(int)

    return dataframe


def remove_duplicate_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows and report the number removed.
    """

    dataframe = dataframe.copy()

    duplicate_count = int(dataframe.duplicated().sum())

    if duplicate_count > 0:
        dataframe = dataframe.drop_duplicates().reset_index(drop=True)

    print(f"Duplicate rows removed: {duplicate_count}")

    return dataframe


def convert_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert known numerical columns to numeric data types.
    Invalid values become NaN and are handled later by the preprocessor.
    """

    dataframe = dataframe.copy()

    numeric_columns = [
        "age",
        "experience_years",
        "work_hours_per_week",
        "overtime_hours",
        "meetings_per_day",
        "deadlines_missed",
        "job_satisfaction",
        "manager_support",
        "work_life_balance",
        "sleep_hours",
        "physical_activity_days",
        "screen_time_hours",
        "caffeine_intake",
        "social_support_score",
        "stress_level",
        "anxiety_score",
        "depression_score",
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


def min_max_scale_series(series: pd.Series) -> pd.Series:
    """
    Scale a numerical series to the range 0–1.

    This is used only for constructing engineered indices.
    The model's main numerical preprocessing is performed separately.
    """

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return series

    if np.isclose(maximum, minimum):
        return pd.Series(
            np.zeros(len(series)),
            index=series.index,
            dtype=float,
        )

    return (series - minimum) / (maximum - minimum)


def create_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create the same five domain-informed features used in the complete
    feature-engineering experiment.

    IMPORTANT:
    These formulas must match Experiment 02 exactly. If your previous
    experiment used different formulas, replace this function with the
    exact function from Experiment 02.
    """

    dataframe = dataframe.copy()

    # ------------------------------------------------------
    # Scale the component variables for index construction
    # ------------------------------------------------------

    stress_scaled = min_max_scale_series(dataframe["stress_level"])
    anxiety_scaled = min_max_scale_series(dataframe["anxiety_score"])
    depression_scaled = min_max_scale_series(dataframe["depression_score"])

    work_hours_scaled = min_max_scale_series(
        dataframe["work_hours_per_week"]
    )
    overtime_scaled = min_max_scale_series(dataframe["overtime_hours"])
    meetings_scaled = min_max_scale_series(dataframe["meetings_per_day"])
    deadlines_scaled = min_max_scale_series(dataframe["deadlines_missed"])

    job_satisfaction_scaled = min_max_scale_series(
        dataframe["job_satisfaction"]
    )
    manager_support_scaled = min_max_scale_series(
        dataframe["manager_support"]
    )
    work_life_balance_scaled = min_max_scale_series(
        dataframe["work_life_balance"]
    )
    social_support_scaled = min_max_scale_series(
        dataframe["social_support_score"]
    )

    sleep_scaled = min_max_scale_series(dataframe["sleep_hours"])
    activity_scaled = min_max_scale_series(
        dataframe["physical_activity_days"]
    )
    caffeine_scaled = min_max_scale_series(dataframe["caffeine_intake"])
    screen_time_scaled = min_max_scale_series(
        dataframe["screen_time_hours"]
    )

    # ------------------------------------------------------
    # 1. Mental-health index
    # Higher value = greater mental-health difficulty
    # ------------------------------------------------------

    dataframe["mental_health_index"] = (
        stress_scaled
        + anxiety_scaled
        + depression_scaled
    ) / 3

    # ------------------------------------------------------
    # 2. Workload index
    # Higher value = greater workload pressure
    # ------------------------------------------------------

    dataframe["workload_index"] = (
        work_hours_scaled
        + overtime_scaled
        + meetings_scaled
        + deadlines_scaled
    ) / 4

    # ------------------------------------------------------
    # 3. Work-life support index
    # Higher value = stronger workplace and social support
    # ------------------------------------------------------

    dataframe["work_life_support_index"] = (
        job_satisfaction_scaled
        + manager_support_scaled
        + work_life_balance_scaled
        + social_support_scaled
    ) / 4

    # ------------------------------------------------------
    # 4. Lifestyle index
    # Higher value = healthier lifestyle
    # ------------------------------------------------------

    dataframe["lifestyle_index"] = (
        sleep_scaled
        + activity_scaled
        + (1 - caffeine_scaled)
    ) / 3

    # ------------------------------------------------------
    # 5. Screen/workload interaction
    # ------------------------------------------------------

    dataframe["screen_workload_interaction"] = (
        screen_time_scaled * work_hours_scaled
    )

    print("\nEngineered features created:")

    for feature in ENGINEERED_FEATURES:
        print(f"  - {feature}")

    return dataframe


def remove_leakage_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the original burnout target variables from model predictors.
    """

    dataframe = dataframe.copy()

    existing_leakage_columns = [
        column
        for column in LEAKAGE_COLUMNS
        if column in dataframe.columns
    ]

    dataframe = dataframe.drop(
        columns=existing_leakage_columns,
        errors="ignore",
    )

    print("\nLeakage columns removed:")
    for column in existing_leakage_columns:
        print(f"  - {column}")

    return dataframe


def remove_low_contribution_features(
    dataframe: pd.DataFrame,
    features_to_remove: Iterable[str] = FEATURES_TO_REMOVE,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Remove the selected low-contributing features identified by SHAP.

    Returns:
        reduced dataframe,
        successfully removed features,
        requested features not present in the dataframe.
    """

    dataframe = dataframe.copy()

    requested_features = list(features_to_remove)

    removed_features = [
        feature
        for feature in requested_features
        if feature in dataframe.columns
    ]

    missing_features = [
        feature
        for feature in requested_features
        if feature not in dataframe.columns
    ]

    dataframe = dataframe.drop(
        columns=removed_features,
        errors="ignore",
    )

    print("\nSHAP-based features removed:")
    for feature in removed_features:
        print(f"  - {feature}")

    if missing_features:
        print("\nRequested removal features not found:")
        for feature in missing_features:
            print(f"  - {feature}")

    return dataframe, removed_features, missing_features


def create_stratified_sample(
    dataframe: pd.DataFrame,
    sample_size: int | None = SAMPLE_SIZE,
) -> pd.DataFrame:
    """
    Create a reproducible stratified sample.

    If sample_size is None or greater than the dataset size,
    the complete dataset is returned.
    """

    if sample_size is None or sample_size >= len(dataframe):
        print("\nUsing the complete dataset.")
        return dataframe.reset_index(drop=True)

    sampled_parts = []

    for _, group in dataframe.groupby(TARGET_COLUMN):
        group_fraction = len(group) / len(dataframe)
        group_sample_size = round(sample_size * group_fraction)

        sampled_group = group.sample(
            n=min(group_sample_size, len(group)),
            random_state=RANDOM_STATE,
        )

        sampled_parts.append(sampled_group)

    sampled_dataframe = pd.concat(
        sampled_parts,
        axis=0,
        ignore_index=True,
    )

    # Correct possible rounding differences
    if len(sampled_dataframe) > sample_size:
        sampled_dataframe = sampled_dataframe.sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )

    elif len(sampled_dataframe) < sample_size:
        remaining = dataframe.drop(sampled_dataframe.index, errors="ignore")

        required_rows = sample_size - len(sampled_dataframe)

        if required_rows > 0 and len(remaining) >= required_rows:
            additional_rows = remaining.sample(
                n=required_rows,
                random_state=RANDOM_STATE,
            )

            sampled_dataframe = pd.concat(
                [sampled_dataframe, additional_rows],
                ignore_index=True,
            )

    sampled_dataframe = sampled_dataframe.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print(f"\nStratified sample shape: {sampled_dataframe.shape}")
    print("Sample target distribution:")
    print(sampled_dataframe[TARGET_COLUMN].value_counts().sort_index())

    return sampled_dataframe


def prepare_feature_engineered_reduced_dataset(
    file_path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Complete data preparation workflow for Experiment 03.
    """

    dataframe = load_dataset(file_path)
    dataframe = clean_column_names(dataframe)

    validate_required_columns(dataframe)

    dataframe = remove_duplicate_rows(dataframe)
    dataframe = convert_numeric_columns(dataframe)
    dataframe = create_binary_target(dataframe)

    # Create the engineered dataset first
    dataframe = create_engineered_features(dataframe)

    # Remove target leakage
    dataframe = remove_leakage_columns(dataframe)

    # Remove low-contributing features after feature engineering
    dataframe, removed_features, missing_features = (
        remove_low_contribution_features(dataframe)
    )

    # Take the same stratified sample size used previously
    dataframe = create_stratified_sample(dataframe)

    print(f"\nFinal reduced dataset shape: {dataframe.shape}")

    return dataframe, removed_features, missing_features