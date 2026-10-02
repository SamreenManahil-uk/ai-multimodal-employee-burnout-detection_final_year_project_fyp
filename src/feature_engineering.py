import os
import pandas as pd


INPUT_PATH = os.path.join("outputs", "resized_50k_dataset.csv")
OUTPUT_PATH = os.path.join("outputs", "feature_engineered_50k_dataset.csv")


def add_engineered_features(df):
    """
    Add evidence-based engineered features based on SHAP results.
    No leakage columns are used.
    """

    df = df.copy()

    print("=" * 70)
    print("STEP: FEATURE ENGINEERING")
    print("=" * 70)

    print("\nOriginal shape:", df.shape)

    df["mental_health_index"] = (
        df["stress_level"] +
        df["anxiety_score"] +
        df["depression_score"]
    ) / 3

    df["workload_index"] = (
        df["work_hours_per_week"] +
        df["overtime_hours"]
    )

    df["work_life_support_index"] = (
        df["work_life_balance"] +
        df["manager_support"]
    ) / 2

    df["lifestyle_index"] = (
        df["sleep_hours"] +
        df["physical_activity_days"]
    ) / 2

    df["screen_workload_interaction"] = (
        df["screen_time_hours"] *
        df["work_hours_per_week"]
    )

    print("\nEngineered features added:")
    print([
        "mental_health_index",
        "workload_index",
        "work_life_support_index",
        "lifestyle_index",
        "screen_workload_interaction"
    ])

    print("\nNew shape:", df.shape)

    os.makedirs("outputs", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature engineered dataset saved to:")
    print(OUTPUT_PATH)

    return df


if __name__ == "__main__":

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}\n"
            "Please run balance_data.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    engineered_df = add_engineered_features(df)

    print("\nFirst 5 rows after feature engineering:")
    print(engineered_df.head())