"""
SCRIPT 1: Precision-Recall Curve
================================
Generates a real Precision-Recall curve from your actual trained XGBoost model
and actual test set. This is NOT in your IEEE paper (which only has the ROC curve),
so it's genuine new content for Supporting Material.

HOW TO USE:
1. Place this file in your project folder (same level as your saved model / data files)
2. Edit the 3 lines marked "EDIT THIS" below to match YOUR actual file paths/variable names
3. Run: python generate_pr_curve.py
4. It will save "precision_recall_curve.png" in the same folder
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend — avoids the Tkinter/Tcl error, saves PNG directly
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# IMPORTANT: Use the FEATURE-ENGINEERED pipeline (42 features), NOT the baseline
# pipeline — this must match the model reported in your IEEE paper (92.8% acc).
# ============ EDIT THIS: paths to your real saved files ============
MODEL_PATH = "models/feature_engineered/xgboost_model.joblib"
X_TEST_PATH = "outputs/X_test_encoded.csv"
Y_TEST_PATH = "outputs/y_test.csv"
# =====================================================================

model = joblib.load(MODEL_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()  # squeeze() turns single-column df into a Series

# Get real predicted probabilities from your real model
y_scores = model.predict_proba(X_test)[:, 1]  # probability of the positive (at-risk) class

# Compute real precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
ap_score = average_precision_score(y_test, y_scores)

# Plot
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(recall, precision, color='#7B4FE0', linewidth=2, label=f'XGBoost (AP = {ap_score:.3f})')
ax.axhline(y=y_test.mean(), color='gray', linestyle='--', linewidth=1,
           label=f'Baseline (positive rate = {y_test.mean():.3f})')
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Precision-Recall Curve — XGBoost (Test Set)', fontsize=12)
ax.legend(loc='lower left', fontsize=9.5)
ax.grid(alpha=0.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved precision_recall_curve.png — Average Precision = {ap_score:.4f}")
print("This is a REAL result from your actual model. Add this AP score to your Supporting Material text.")