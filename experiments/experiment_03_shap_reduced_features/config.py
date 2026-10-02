from pathlib import Path

# ==========================================================
# Experiment information
# ==========================================================

EXPERIMENT_NAME = "Experiment 03 - SHAP Reduced Feature Engineering"

# ==========================================================
# Reproducibility and evaluation settings
# ==========================================================

RANDOM_STATE = 42
SAMPLE_SIZE = 50_000
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET_COLUMN = "target"

# ==========================================================
# Dataset columns
# ==========================================================

LEAKAGE_COLUMNS = [
    "burnout_level",
    "burnout_score",
]

ENGINEERED_FEATURES = [
    "mental_health_index",
    "workload_index",
    "work_life_support_index",
    "lifestyle_index",
    "screen_workload_interaction",
]

# Low-contribution features selected from the SHAP analysis
FEATURES_TO_REMOVE = [
    "age",
    "deadlines_missed",
    "caffeine_intake",
    "screen_time_hours",
    "lifestyle_index",
]

# ==========================================================
# Random Forest parameters
# ==========================================================

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ==========================================================
# XGBoost parameters
# ==========================================================

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ==========================================================
# Project and experiment paths
# ==========================================================

# Current file:
# My MSc Project/
# └── experiments/
#     └── experiment_03_shap_reduced_features/
#         └── config.py
#
# parents[0] = experiment_03_shap_reduced_features
# parents[1] = experiments
# parents[2] = My MSc Project

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset is directly inside the data folder:
#
# My MSc Project/
# └── data/
#     └── tech_mental_health_burnout.csv

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "tech_mental_health_burnout.csv"
)

# Experiment-specific output folders

MODELS_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PLOTS_DIR = EXPERIMENT_DIR / "plots"
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"