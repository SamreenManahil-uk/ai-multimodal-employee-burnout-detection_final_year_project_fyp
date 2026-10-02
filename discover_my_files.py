"""
SCRIPT 0: Project Discovery — Run This FIRST
==============================================
This script scans your current folder and subfolders to find your saved
model files, data files, and shows you their exact column names — so you
know exactly what to type into the "EDIT THIS" sections of the other
4 scripts, instead of guessing.

HOW TO USE:
1. Copy this file into your MAIN project folder (the one containing your
   burnout detection code — app.py, fusion.py, etc.)
2. Run: python discover_my_files.py
3. Read the output — it tells you exactly what to paste into the other scripts.
"""

import os
import glob

print("="*70)
print("STEP 1: Looking for saved MODEL files (.joblib, .pkl, .sav)")
print("="*70)
model_files = glob.glob("**/*.joblib", recursive=True) + \
              glob.glob("**/*.pkl", recursive=True) + \
              glob.glob("**/*.sav", recursive=True)
if model_files:
    for f in model_files:
        size_kb = os.path.getsize(f) / 1024
        print(f"  FOUND: {f}  ({size_kb:.0f} KB)")
else:
    print("  None found in this folder. Try running this script from your")
    print("  main project folder, or search manually for where your model")
    print("  was saved (look for a line like joblib.dump(model, ...) in your code).")

print()
print("="*70)
print("STEP 2: Looking for saved DATA files (.csv)")
print("="*70)
csv_files = glob.glob("**/*.csv", recursive=True)
if csv_files:
    for f in csv_files[:20]:  # limit output if there are many
        size_kb = os.path.getsize(f) / 1024
        print(f"  FOUND: {f}  ({size_kb:.0f} KB)")
    if len(csv_files) > 20:
        print(f"  ... and {len(csv_files) - 20} more CSV files")
else:
    print("  No CSV files found in this folder.")

print()
print("="*70)
print("STEP 3: Inspecting column names in each CSV found (first 5 rows)")
print("="*70)
try:
    import pandas as pd
    for f in csv_files[:10]:
        try:
            df = pd.read_csv(f, nrows=5)
            print(f"\n  --- {f} ---")
            print(f"  Shape (first 5 rows preview): {df.shape}")
            print(f"  Columns: {list(df.columns)}")
        except Exception as e:
            print(f"\n  --- {f} --- (could not read: {e})")
except ImportError:
    print("  pandas not installed — run: pip install pandas")

print()
print("="*70)
print("NEXT STEPS")
print("="*70)
print("""
1. From the MODEL files listed above, pick your XGBoost model file and your
   Random Forest model file (if saved separately) — copy their exact paths
   into MODEL_PATH / XGB_MODEL_PATH / RF_MODEL_PATH in the other scripts.

2. From the CSV files listed above, find:
   - Your test features file (likely named something like X_test.csv,
     or it might be a single combined file you split in code)
   - Your test labels file (y_test.csv, or a 'label'/'target' column
     inside a combined file)
   - Your full processed dataset (for the feature-distribution script)

3. From the "Columns" printed above, copy the EXACT column names for:
   - Your target/label column (e.g. 'burnout_risk', 'is_at_risk', 'label')
   - Your top SHAP features (e.g. 'stress_level', 'work_hours_per_week')

4. Paste these exact paths and names into the "EDIT THIS" sections of:
   generate_pr_curve.py, generate_feature_distributions.py,
   generate_combined_roc.py, generate_threshold_table.py

If you don't have X_test/y_test saved as separate files (common if you
split the data in-memory during training), that's fine — just tell me,
and I'll give you a modified script that re-does the same train/test split
your original code used, so the test set matches exactly.
""")