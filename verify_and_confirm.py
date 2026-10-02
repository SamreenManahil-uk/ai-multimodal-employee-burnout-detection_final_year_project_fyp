"""
SCRIPT 0b: Verify & Confirm — Run This SECOND (after discover_my_files.py)
=============================================================================
Based on your discover_my_files.py output, this script tries the most likely
real files for your FEATURE-ENGINEERED pipeline, loads your real XGBoost
model, and checks whether the resulting accuracy/ROC-AUC match your IEEE
paper's reported numbers (92.8% accuracy, 0.958 ROC-AUC).

If the numbers match -> you've found the right files. The script prints the
exact strings to copy into the other 4 scripts.
If they don't match -> it tells you so clearly, and you paste the output
here so I can help find the right files instead of guessing.

HOW TO USE: python verify_and_confirm.py
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

print("="*70)
print("Trying: models/feature_engineered/xgboost_model.joblib")
print("="*70)

try:
    model = joblib.load("models/feature_engineered/xgboost_model.joblib")
    print("Model loaded OK.")
    print(f"Model expects {model.n_features_in_} features.")
except Exception as e:
    print(f"COULD NOT LOAD MODEL: {e}")
    model = None

# Try both candidate test-feature files
X_test = None
X_test_path_used = None
for candidate in ["outputs/X_test_encoded.csv", "outputs/X_test.csv"]:
    try:
        df = pd.read_csv(candidate)
        print(f"\nTried {candidate}: shape = {df.shape}")
        if model is not None and df.shape[1] == model.n_features_in_:
            print(f"  -> Column count MATCHES model's expected {model.n_features_in_} features. Likely correct file.")
            X_test = df
            X_test_path_used = candidate
        else:
            print(f"  -> Column count does NOT match model's expected features.")
    except Exception as e:
        print(f"\nTried {candidate}: could not load ({e})")

# Try to find y_test automatically
y_test = None
y_test_path_used = None
for candidate in ["outputs/y_test.csv", "outputs/Y_test.csv", "outputs/y_test_encoded.csv"]:
    try:
        y_test = pd.read_csv(candidate).squeeze()
        y_test_path_used = candidate
        print(f"\nFound y_test at: {candidate} (shape: {y_test.shape})")
        break
    except Exception:
        continue

if y_test is None:
    print("\nCould NOT automatically find a y_test.csv file in outputs/.")
    print("Please check your outputs/ folder manually for the test-labels file,")
    print("or tell me and I'll help you regenerate it from your original split code.")

# If we have everything, run the real check
if model is not None and X_test is not None and y_test is not None:
    print("\n" + "="*70)
    print("RUNNING REAL VERIFICATION AGAINST YOUR PAPER'S REPORTED NUMBERS")
    print("="*70)
    try:
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_scores)
        print(f"Real accuracy on this test set: {acc:.4f}  (paper reports 0.9277)")
        print(f"Real ROC-AUC on this test set:  {auc:.4f}  (paper reports 0.9580)")
        if abs(acc - 0.9277) < 0.01 and abs(auc - 0.958) < 0.01:
            print("\n*** MATCH CONFIRMED — these are the correct files. ***")
            print("\nCopy these exact lines into the other 4 scripts:")
            print(f'  MODEL_PATH / XGB_MODEL_PATH = "models/feature_engineered/xgboost_model.joblib"')
            print(f'  RF_MODEL_PATH = "models/feature_engineered/random_forest_model.joblib"')
            print(f'  X_TEST_PATH = "{X_test_path_used}"')
            print(f'  Y_TEST_PATH = "{y_test_path_used}"')
        else:
            print("\nNumbers do NOT closely match your paper. This might still be the")
            print("right pipeline (small differences can be normal), but please paste")
            print("this full output back so we can double check together.")
    except Exception as e:
        print(f"Prediction failed: {e}")
        print("This usually means X_test columns don't exactly match what the model expects.")
else:
    print("\nCould not run full verification — see messages above for what's missing.")

# Also inspect the full engineered dataset for the feature-distribution script
print("\n" + "="*70)
print("Checking outputs/feature_engineered_50k_dataset.csv (for distribution plots)")
print("="*70)
try:
    full_df = pd.read_csv("outputs/feature_engineered_50k_dataset.csv", nrows=5)
    print(f"Shape (preview): {full_df.shape}")
    print(f"Columns: {list(full_df.columns)}")
except Exception as e:
    print(f"Could not load: {e}")