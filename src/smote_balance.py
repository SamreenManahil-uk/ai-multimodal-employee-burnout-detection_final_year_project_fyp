import os
import pandas as pd
from imblearn.over_sampling import SMOTE


X_TRAIN_PATH = os.path.join("outputs", "X_train_encoded.csv")
Y_TRAIN_PATH = os.path.join("outputs", "y_train.csv")

X_TRAIN_SMOTE_PATH = os.path.join("outputs", "X_train_smote.csv")
Y_TRAIN_SMOTE_PATH = os.path.join("outputs", "y_train_smote.csv")


def apply_smote():
    print("=" * 70)
    print("STARTING SMOTE BALANCING ON TRAINING DATA ONLY")
    print("=" * 70)

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

    print("\nBefore SMOTE:")
    print(y_train.value_counts())

    smote = SMOTE(random_state=42)

    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_smote).value_counts())

    print("\nX_train before SMOTE:", X_train.shape)
    print("X_train after SMOTE :", X_train_smote.shape)

    os.makedirs("outputs", exist_ok=True)

    X_train_smote.to_csv(X_TRAIN_SMOTE_PATH, index=False)
    pd.Series(y_train_smote, name="target").to_csv(Y_TRAIN_SMOTE_PATH, index=False)

    print("\nSMOTE files saved:")
    print(X_TRAIN_SMOTE_PATH)
    print(Y_TRAIN_SMOTE_PATH)

    return X_train_smote, y_train_smote


if __name__ == "__main__":
    apply_smote()