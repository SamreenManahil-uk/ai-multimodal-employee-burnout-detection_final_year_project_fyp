from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import (
    CV_FOLDS,
    RANDOM_FOREST_PARAMS,
    RANDOM_STATE,
    XGBOOST_PARAMS,
)


def identify_feature_types(
    dataframe: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Identify categorical and numerical feature columns.
    """

    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_columns = [
        column
        for column in dataframe.columns
        if column not in categorical_columns
    ]

    return categorical_columns, numerical_columns


def create_preprocessor(
    categorical_columns: list[str],
    numerical_columns: list[str],
) -> ColumnTransformer:
    """
    Create preprocessing pipelines for categorical and numerical data.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_models() -> dict[str, Any]:
    """
    Create the two machine-learning models.
    """

    return {
        "random_forest": RandomForestClassifier(
            **RANDOM_FOREST_PARAMS
        ),
        "xgboost": XGBClassifier(
            **XGBOOST_PARAMS
        ),
    }


def create_cross_validation_pipeline(
    preprocessor: ColumnTransformer,
    model: Any,
) -> ImbalancedPipeline:
    """
    Create a leakage-safe cross-validation pipeline.

    During every cross-validation fold:
        1. preprocessing is fitted only on that fold's training data;
        2. SMOTE is applied only to that fold's training data;
        3. the model is fitted on the resampled fold.
    """

    return ImbalancedPipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor),
            ),
            (
                "smote",
                SMOTE(random_state=RANDOM_STATE),
            ),
            (
                "model",
                clone(model),
            ),
        ]
    )


def run_cross_validation(
    model_name: str,
    model: Any,
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, float]:
    """
    Run five-fold stratified cross-validation using multiple metrics.
    """

    print(f"\nRunning {CV_FOLDS}-fold CV for {model_name}...")

    pipeline = create_cross_validation_pipeline(
        preprocessor=preprocessor,
        model=model,
    )

    cross_validator = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(
        estimator=pipeline,
        X=x_train,
        y=y_train,
        cv=cross_validator,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

    cv_results = {}

    for metric_name in scoring:
        values = scores[f"test_{metric_name}"]

        cv_results[f"cv_{metric_name}_mean"] = float(
            np.mean(values)
        )
        cv_results[f"cv_{metric_name}_std"] = float(
            np.std(values)
        )

    print(
        f"CV accuracy: "
        f"{cv_results['cv_accuracy_mean']:.4f}"
    )
    print(
        f"CV F1-score: "
        f"{cv_results['cv_f1_mean']:.4f}"
    )
    print(
        f"CV ROC-AUC: "
        f"{cv_results['cv_roc_auc_mean']:.4f}"
    )

    return cv_results


def fit_preprocessor_and_resample(
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[
    ColumnTransformer,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """
    Fit preprocessing on training data only and apply SMOTE only
    to the transformed training set.
    """

    fitted_preprocessor = clone(preprocessor)

    x_train_processed = fitted_preprocessor.fit_transform(x_train)
    x_test_processed = fitted_preprocessor.transform(x_test)

    feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
        .tolist()
    )

    smote = SMOTE(random_state=RANDOM_STATE)

    x_train_resampled, y_train_resampled = smote.fit_resample(
        x_train_processed,
        y_train,
    )

    print("\nTraining distribution before SMOTE:")
    print(y_train.value_counts().sort_index())

    print("\nTraining distribution after SMOTE:")
    unique_values, counts = np.unique(
        y_train_resampled,
        return_counts=True,
    )

    for class_value, count in zip(unique_values, counts):
        print(f"{class_value}: {count}")

    return (
        fitted_preprocessor,
        x_train_resampled,
        x_test_processed,
        np.asarray(y_train_resampled),
        np.asarray(y_train),
        feature_names,
    )


def train_final_models(
    models: dict[str, Any],
    x_train_resampled: np.ndarray,
    y_train_resampled: np.ndarray,
) -> dict[str, Any]:
    """
    Train final Random Forest and XGBoost models on the
    SMOTE-resampled training set.
    """

    fitted_models = {}

    for model_name, model in models.items():
        print(f"\nTraining final {model_name} model...")

        fitted_model = clone(model)

        fitted_model.fit(
            x_train_resampled,
            y_train_resampled,
        )

        fitted_models[model_name] = fitted_model

        print(f"{model_name} training complete.")

    return fitted_models