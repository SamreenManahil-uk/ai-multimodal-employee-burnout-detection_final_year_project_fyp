import os
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = os.path.join("outputs", "cleaned_dataset.csv")
OUTPUT_PATH = os.path.join("outputs", "resized_50k_dataset.csv")


def resize_dataset_to_50k(df):
    """
    Resize dataset to 50,000 rows using stratified sampling.
    This keeps the original class ratio.
    SMOTE will be applied later only on training data.
    """

    print("\nStarting stratified resize to 50,000 rows...")

    print("\nOriginal dataset shape:")
    print(df.shape)

    print("\nOriginal target distribution:")
    print(df["target"].value_counts())

    df_50k, _ = train_test_split(
        df,
        train_size=50000,
        stratify=df["target"],
        random_state=42
    )

    df_50k = df_50k.reset_index(drop=True)

    print("\nResized dataset shape:")
    print(df_50k.shape)

    print("\nResized target distribution:")
    print(df_50k["target"].value_counts())

    os.makedirs("outputs", exist_ok=True)

    df_50k.to_csv(OUTPUT_PATH, index=False)

    print(f"\nResized 50k dataset saved to: {OUTPUT_PATH}")

    return df_50k


if __name__ == "__main__":

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Cleaned dataset not found at {INPUT_PATH}. "
            "Please run preprocess.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    resized_df = resize_dataset_to_50k(df)

    print("\nFirst 5 rows of resized dataset:")
    print(resized_df.head())