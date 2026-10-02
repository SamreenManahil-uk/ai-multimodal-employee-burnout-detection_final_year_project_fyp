"""
SCRIPT 2: Feature Distribution Comparison (Box Plots)
======================================================
Shows WHY your top SHAP features (stress_level, work_hours_per_week,
work_life_balance) are predictive, by comparing their distributions between
At-Risk and Not-At-Risk employees. This is NOT in your IEEE paper and is
genuinely useful supporting evidence for your SHAP findings.

HOW TO USE:
1. Edit the paths below to your real dataset file
2. Edit FEATURE_COLS if your column names are different
3. Run: python generate_feature_distributions.py
4. Saves "feature_distributions.png"
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend — avoids the Tkinter/Tcl error, saves PNG directly
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORTANT: Use the FEATURE-ENGINEERED pipeline (42 features), NOT the baseline
# pipeline — this must match the model reported in your IEEE paper (92.8% acc).
# ============ EDIT THIS ============
DATA_PATH = "outputs/feature_engineered_50k_dataset.csv"
TARGET_COL = "target"
FEATURE_COLS = ["mental_health_index", "stress_level", "work_life_balance"]
# =====================================

df = pd.read_csv(DATA_PATH)

fig, axes = plt.subplots(1, len(FEATURE_COLS), figsize=(4 * len(FEATURE_COLS), 4.5))
colors = ['#A8C6E8', '#E88B8B']

for ax, feature in zip(axes, FEATURE_COLS):
    sns.boxplot(data=df, x=TARGET_COL, y=feature, ax=ax, palette=colors)
    ax.set_title(feature, fontsize=11)
    ax.set_xlabel('')
    ax.set_xticklabels(['Not At-Risk', 'At-Risk'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Distribution of Top SHAP Features by Risk Class (Real Dataset)', fontsize=12, y=1.03)
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved feature_distributions.png")

# Also print real summary statistics for your write-up
print("\nReal median values by class (paste these into your text):")
for feature in FEATURE_COLS:
    medians = df.groupby(TARGET_COL)[feature].median()
    print(f"  {feature}: Not-At-Risk median = {medians[0]:.2f}, At-Risk median = {medians[1]:.2f}")