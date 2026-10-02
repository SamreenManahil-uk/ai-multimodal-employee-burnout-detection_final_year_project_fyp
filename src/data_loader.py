import pandas as pd
import os


def load_dataset():
    """
    Load and inspect the dataset.
    """

    # -----------------------------
    # Load Dataset
    # -----------------------------
    file_path = os.path.join("data", "tech_mental_health_burnout.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    print("=" * 70)
    print("DATASET SUCCESSFULLY LOADED")
    print("=" * 70)

    # -----------------------------
    # Basic Information
    # -----------------------------
    print("\nDataset Shape")
    print(df.shape)

    print("\nNumber of Rows:", df.shape[0])
    print("Number of Columns:", df.shape[1])

    # -----------------------------
    # First & Last Rows
    # -----------------------------
    print("\nFirst 5 Rows")
    print(df.head())

    print("\nLast 5 Rows")
    print(df.tail())

    # -----------------------------
    # Column Names
    # -----------------------------
    print("\nColumn Names")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i}. {col}")

    # -----------------------------
    # Data Types
    # -----------------------------
    print("\nData Types")
    print(df.dtypes)

    # -----------------------------
    # Dataset Info
    # -----------------------------
    print("\nDataset Information")
    df.info()

    # -----------------------------
    # Missing Values
    # -----------------------------
    print("\nMissing Values")
    print(df.isnull().sum())

    # -----------------------------
    # Duplicate Rows
    # -----------------------------
    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    # -----------------------------
    # Numerical Summary
    # -----------------------------
    print("\nNumerical Summary")
    print(df.describe())

    # -----------------------------
    # Categorical Summary
    # -----------------------------
    print("\nCategorical Summary")
    print(df.describe(include="object"))

    # -----------------------------
    # Unique Values
    # -----------------------------
    print("\nUnique Values Per Column")
    print(df.nunique())

    # -----------------------------
    # Target Column Check
    # -----------------------------
    print("\nPossible Target Columns")

    possible_targets = [
        "burnout_level",
        "burnout_score",
        "label",
        "target",
        "class",
    ]

    found = False

    for col in possible_targets:
        if col in df.columns:
            found = True
            print(f"\nTarget Column Found: {col}")
            print(df[col].value_counts(dropna=False))

    if not found:
        print("No common target column found automatically.")

    print("\nFinished Dataset Inspection")
    print("=" * 70)

    return df


if __name__ == "__main__":
    load_dataset()