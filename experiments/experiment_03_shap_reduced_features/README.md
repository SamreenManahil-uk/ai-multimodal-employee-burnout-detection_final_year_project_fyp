# Experiment 03: SHAP-Reduced Feature-Engineered Models

## Objective

This experiment evaluates whether a reduced feature-engineered dataset can
maintain or improve predictive performance after removing features with low
mean absolute SHAP contribution.

## Starting point

The experiment begins with the same original dataset and the same five
engineered features used in the complete feature-engineering experiment:

1. mental_health_index
2. workload_index
3. work_life_support_index
4. lifestyle_index
5. screen_workload_interaction

## Features removed

The following features were selected conservatively using the SHAP ranking:

- age
- deadlines_missed
- caffeine_intake
- screen_time_hours
- lifestyle_index

## Models

- Random Forest
- XGBoost

## Method

1. Load and validate the original dataset.
2. Create a binary burnout target.
3. Recreate all engineered features.
4. Remove burnout_level and burnout_score to prevent leakage.
5. Remove the selected low-SHAP features.
6. Create a stratified sample of 50,000 records.
7. Perform an 80/20 stratified train-test split.
8. Impute missing values.
9. Standardise numerical features.
10. One-hot encode categorical features.
11. Apply SMOTE only to training data.
12. Perform five-fold stratified cross-validation.
13. Train and evaluate Random Forest and XGBoost.
14. Save every output inside this experiment directory.

## Run

From the project root:

```powershell
python experiments\experiment_03_shap_reduced_features\main.py