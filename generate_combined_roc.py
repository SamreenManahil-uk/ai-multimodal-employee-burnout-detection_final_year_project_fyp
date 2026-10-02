"""
SCRIPT 3: Combined ROC Curve — Random Forest vs XGBoost on ONE plot
====================================================================
Your IEEE paper shows the XGBoost ROC curve alone (Fig. 4) and the two
confusion matrices side-by-side (Fig. 3), but never overlays BOTH models'
ROC curves on the same axes for direct visual comparison. This is genuinely
new content for Supporting Material.

HOW TO USE:
1. Edit the paths below to your two real saved models and real test data
2. Run: python generate_combined_roc.py
3. Saves "combined_roc_comparison.png"
"""

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend — avoids the Tkinter/Tcl error, saves PNG directly
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# IMPORTANT: Use the FEATURE-ENGINEERED pipeline (42 features), NOT the baseline
# pipeline — this must match the model reported in your IEEE paper (92.8% acc).
# ============ EDIT THIS ============
RF_MODEL_PATH = "models/feature_engineered/random_forest_model.joblib"
XGB_MODEL_PATH = "models/feature_engineered/xgboost_model.joblib"
X_TEST_PATH = "outputs/X_test_encoded.csv"
Y_TEST_PATH = "outputs/y_test.csv"
# =====================================

rf_model = joblib.load(RF_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()

rf_scores = rf_model.predict_proba(X_test)[:, 1]
xgb_scores = xgb_model.predict_proba(X_test)[:, 1]

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_scores)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_scores)
rf_auc = roc_auc_score(y_test, rf_scores)
xgb_auc = roc_auc_score(y_test, xgb_scores)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(rf_fpr, rf_tpr, color='#5B8FE0', linewidth=2, label=f'Random Forest (AUC = {rf_auc:.3f})')
ax.plot(xgb_fpr, xgb_tpr, color='#E0975B', linewidth=2, label=f'XGBoost (AUC = {xgb_auc:.3f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random guess')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curve Comparison — Random Forest vs XGBoost (Test Set)', fontsize=11.5)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('combined_roc_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved combined_roc_comparison.png")
print(f"RF test-set AUC = {rf_auc:.4f}, XGBoost test-set AUC = {xgb_auc:.4f}")
print("These are REAL test-set (not cross-validation) AUC values for both models on the same axes.")