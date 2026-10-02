import pandas as pd
import os


DATASET_PATH = os.path.join("data", "tech_mental_health_burnout.csv")

LEAKAGE_COLUMNS = [
    "burnout_level",
    "burnout_score"
]


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}\n"
            "Make sure you are running this file from the main project folder."
        )

    df = pd.read_csv(DATASET_PATH)

    print("=" * 70)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 70)
    print("Shape:", df.shape)

    return df


def create_binary_target(df):
    df = df.copy()

    df["target"] = df["burnout_level"].apply(
        lambda x: 0 if x == "Low" else 1
    )

    return df


def remove_leakage_columns(df):
    df = df.copy()

    existing_leakage_cols = [
        col for col in LEAKAGE_COLUMNS if col in df.columns
    ]

    df = df.drop(columns=existing_leakage_cols)

    print("\nRemoved leakage columns:")
    print(existing_leakage_cols)

    return df


def clean_data(df):
    print("\nStarting preprocessing...")
    print("Original shape:", df.shape)

    df = df.drop_duplicates()
    print("After removing duplicates:", df.shape)

    df = create_binary_target(df)

    print("\nBinary target distribution:")
    print(df["target"].value_counts())

    df = remove_leakage_columns(df)

    print("\nFinal shape after preprocessing:", df.shape)

    print("\nRemaining columns:")
    print(df.columns.tolist())

    print("\nPreprocessing completed successfully.")

    return df


if __name__ == "__main__":
    df = load_dataset()
    df = clean_data(df)

    print("\nFirst 5 rows after preprocessing:")
    print(df.head())