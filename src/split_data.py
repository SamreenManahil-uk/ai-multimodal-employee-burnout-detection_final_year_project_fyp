import os
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = os.path.join("outputs", "resized_50k_dataset.csv")

X_TRAIN_PATH = os.path.join("outputs", "X_train.csv")
X_TEST_PATH = os.path.join("outputs", "X_test.csv")
Y_TRAIN_PATH = os.path.join("outputs", "y_train.csv")
Y_TEST_PATH = os.path.join("outputs", "y_test.csv")


def split_dataset(df):
    """
    Split dataset into training and testing sets.

    Target column:
    target

    Split:
    80% training
    20% testing

    Stratify:
    Keeps the same class ratio in train and test.
    """

    print("\nStarting train-test split...")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nX_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    os.makedirs("outputs", exist_ok=True)

    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)

    print("\nTrain-test split files saved:")
    print(X_TRAIN_PATH)
    print(X_TEST_PATH)
    print(Y_TRAIN_PATH)
    print(Y_TEST_PATH)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Resized dataset not found at: {INPUT_PATH}\n"
            "Please run balance_data.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    split_dataset(df)