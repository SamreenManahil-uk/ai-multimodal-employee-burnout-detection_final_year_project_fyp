"""
SCRIPT 4: Decision-Threshold Sensitivity Table
================================================
Your paper uses a fixed 0.50 decision threshold throughout. This script shows
how accuracy, precision, recall and F1 would each change if you moved that
threshold — directly relevant evidence for your Discussion point about
recall-vs-F1 selection, and genuinely NOT shown anywhere in your IEEE paper.

HOW TO USE:
1. Edit the paths below
2. Run: python generate_threshold_table.py
3. Prints a real table you can paste directly into Supporting Material,
   and saves "threshold_sensitivity.png" (a line chart version)
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend — avoids the Tkinter/Tcl error, saves PNG directly
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# IMPORTANT: Use the FEATURE-ENGINEERED pipeline (42 features), NOT the baseline
# pipeline — this must match the model reported in your IEEE paper (92.8% acc).
# ============ EDIT THIS ============
MODEL_PATH = "models/feature_engineered/xgboost_model.joblib"
X_TEST_PATH = "outputs/X_test_encoded.csv"
Y_TEST_PATH = "outputs/y_test.csv"
# =====================================

model = joblib.load(MODEL_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()
y_scores = model.predict_proba(X_test)[:, 1]

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
rows = []
for t in thresholds:
    y_pred = (y_scores >= t).astype(int)
    rows.append({
        "Threshold": t,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    })

results_df = pd.DataFrame(rows)
print("\n=== REAL Threshold Sensitivity Table (paste into Supporting Material) ===\n")
print(results_df.round(4).to_string(index=False))
results_df.to_csv("threshold_sensitivity_table.csv", index=False)

# Plot
fig, ax = plt.subplots(figsize=(7, 4.5))
for col, color in zip(["Accuracy", "Precision", "Recall", "F1"], ['#4C9AFF', '#7B4FE0', '#E0975B', '#3B9B6E']):
    ax.plot(results_df["Threshold"], results_df[col], marker='o', label=col, color=color)
ax.axvline(x=0.50, color='gray', linestyle='--', linewidth=1, label='Current threshold (0.50)')
ax.set_xlabel('Decision Threshold', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Metric Sensitivity Across Decision Thresholds (Real Test Set)', fontsize=11.5)
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('threshold_sensitivity.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\nSaved threshold_sensitivity_table.csv and threshold_sensitivity.png")
print("This directly shows what would happen if you'd selected by recall instead of F1 — real evidence for your Discussion.")