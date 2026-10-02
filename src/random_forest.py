"""
Random Forest training module.

Methodology
-----------
1. Load the original encoded training data.
2. Apply SMOTE inside each cross-validation training fold.
3. Calculate 5-fold cross-validation metrics.
4. Apply SMOTE to the complete training set.
5. Train a final Random Forest candidate model.
6. Save the candidate model.

The test set is not used in this file.

The final test set is evaluated only after model selection
inside compare_models.py.
"""

import os

import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from joblib import dump

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold,
)


# ==========================================================
# FILE PATHS
# ==========================================================

X_TRAIN_PATH = os.path.join(
    "outputs",
    "X_train_encoded.csv",
)

Y_TRAIN_PATH = os.path.join(
    "outputs",
    "y_train.csv",
)

MODEL_PATH = os.path.join(
    "models",
    "random_forest_model.joblib",
)

RESULT_PATH = os.path.join(
    "outputs",
    "random_forest_results.txt",
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    "outputs",
    "random_forest_feature_importance.csv",
)


# ==========================================================
# LOAD TRAINING DATA
# ==========================================================

def load_training_data():
    """
    Load the original encoded training data.

    Important:
    This data has not been processed by SMOTE.
    """

    print("\nLoading original encoded training data...")

    if not os.path.exists(X_TRAIN_PATH):
        raise FileNotFoundError(
            f"Training features were not found at:\n"
            f"{X_TRAIN_PATH}\n\n"
            "Please run split_data.py and encoding.py first."
        )

    if not os.path.exists(Y_TRAIN_PATH):
        raise FileNotFoundError(
            f"Training labels were not found at:\n"
            f"{Y_TRAIN_PATH}\n\n"
            "Please run split_data.py first."
        )

    X_train = pd.read_csv(
        X_TRAIN_PATH
    )

    y_train = pd.read_csv(
        Y_TRAIN_PATH
    ).squeeze("columns")

    if len(X_train) != len(y_train):
        raise ValueError(
            "The number of training feature rows does not "
            "match the number of training labels."
        )

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    print("\nOriginal training target distribution:")
    print(y_train.value_counts())

    return X_train, y_train


# ==========================================================
# MODEL
# ==========================================================

def build_model():
    """
    Create a new Random Forest classifier.
    """

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )


def build_cross_validation_pipeline():
    """
    Create a pipeline that applies SMOTE inside each
    cross-validation training fold.

    The validation portion of each fold is never processed
    by SMOTE.
    """

    return Pipeline(
        steps=[
            (
                "smote",
                SMOTE(
                    random_state=42,
                ),
            ),
            (
                "model",
                build_model(),
            ),
        ]
    )


# ==========================================================
# CROSS-VALIDATION
# ==========================================================

def run_cross_validation(
    X_train,
    y_train,
):
    """
    Run leakage-safe stratified 5-fold cross-validation.
    """

    print("\nStarting leakage-safe 5-fold cross-validation...")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    precision_scorer = make_scorer(
        precision_score,
        zero_division=0,
    )

    recall_scorer = make_scorer(
        recall_score,
        zero_division=0,
    )

    f1_scorer = make_scorer(
        f1_score,
        zero_division=0,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": precision_scorer,
        "recall": recall_scorer,
        "f1": f1_scorer,
        "roc_auc": "roc_auc",
    }

    cv_pipeline = (
        build_cross_validation_pipeline()
    )

    scores = cross_validate(
        estimator=cv_pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

    cv_results = {
        "CV Accuracy": float(
            scores["test_accuracy"].mean()
        ),
        "CV Precision": float(
            scores["test_precision"].mean()
        ),
        "CV Recall": float(
            scores["test_recall"].mean()
        ),
        "CV F1": float(
            scores["test_f1"].mean()
        ),
        "CV ROC-AUC": float(
            scores["test_roc_auc"].mean()
        ),
        "CV Accuracy Standard Deviation": float(
            scores["test_accuracy"].std()
        ),
        "CV Precision Standard Deviation": float(
            scores["test_precision"].std()
        ),
        "CV Recall Standard Deviation": float(
            scores["test_recall"].std()
        ),
        "CV F1 Standard Deviation": float(
            scores["test_f1"].std()
        ),
        "CV ROC-AUC Standard Deviation": float(
            scores["test_roc_auc"].std()
        ),
    }

    print("\nCross-validation results:")

    for metric, value in cv_results.items():
        print(
            f"{metric}: {value:.4f}"
        )

    return cv_results


# ==========================================================
# FINAL CANDIDATE TRAINING
# ==========================================================

def train_final_candidate(
    X_train,
    y_train,
):
    """
    Apply SMOTE to the complete training set and train the
    final Random Forest candidate.

    The test set remains untouched.
    """

    print(
        "\nApplying SMOTE to the complete training set "
        "for final candidate training..."
    )

    smote = SMOTE(
        random_state=42,
    )

    X_train_smote, y_train_smote = (
        smote.fit_resample(
            X_train,
            y_train,
        )
    )

    print("\nTarget distribution after final-training SMOTE:")
    print(
        pd.Series(
            y_train_smote
        ).value_counts()
    )

    print(
        "\nTraining final Random Forest candidate model..."
    )

    model = build_model()

    model.fit(
        X_train_smote,
        y_train_smote,
    )

    return model


# ==========================================================
# SAVE OUTPUTS
# ==========================================================

def save_feature_importance(
    model,
    feature_names,
):
    """
    Save Random Forest feature importance values.
    """

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": (
                model.feature_importances_
            ),
        }
    ).sort_values(
        by="Importance",
        ascending=False,
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    print(
        "Feature importance saved to:",
        FEATURE_IMPORTANCE_PATH,
    )


def save_results(
    cv_results,
):
    """
    Save cross-validation results.

    No test metrics are saved here because the model has not
    yet been selected.
    """

    with open(
        RESULT_PATH,
        mode="w",
        encoding="utf-8",
    ) as result_file:

        result_file.write(
            "RANDOM FOREST MODEL RESULTS\n"
        )

        result_file.write(
            "===========================\n\n"
        )

        result_file.write(
            "Model-selection methodology\n"
        )

        result_file.write(
            "---------------------------\n"
        )

        result_file.write(
            "SMOTE was applied inside each "
            "cross-validation training fold.\n"
        )

        result_file.write(
            "The test set was not used during "
            "cross-validation or model selection.\n\n"
        )

        result_file.write(
            "5-Fold Cross-Validation Results\n"
        )

        result_file.write(
            "--------------------------------\n"
        )

        for metric, value in cv_results.items():
            result_file.write(
                f"{metric}: {value:.6f}\n"
            )

        result_file.write(
            "\nFinal test-set results are produced only "
            "for the selected model in compare_models.py.\n"
        )

    print(
        "Cross-validation results saved to:",
        RESULT_PATH,
    )


# ==========================================================
# MAIN TRAINING FUNCTION
# ==========================================================

def train_random_forest():
    """
    Run Random Forest cross-validation and train the final
    candidate model.
    """

    print("\n" + "=" * 70)
    print("STEP: RANDOM FOREST TRAINING")
    print("=" * 70)

    os.makedirs(
        "models",
        exist_ok=True,
    )

    os.makedirs(
        "outputs",
        exist_ok=True,
    )

    X_train, y_train = (
        load_training_data()
    )

    cv_results = run_cross_validation(
        X_train,
        y_train,
    )

    model = train_final_candidate(
        X_train,
        y_train,
    )

    dump(
        model,
        MODEL_PATH,
    )

    print(
        "\nRandom Forest candidate saved to:",
        MODEL_PATH,
    )

    save_feature_importance(
        model=model,
        feature_names=X_train.columns,
    )

    save_results(
        cv_results=cv_results,
    )

    print(
        "\nRandom Forest candidate training "
        "completed successfully."
    )

    print("=" * 70)

    return {
        "model_name": "Random Forest",
        "model": model,
        "cv_results": cv_results,
        "model_path": MODEL_PATH,
    }


if __name__ == "__main__":
    train_random_forest()