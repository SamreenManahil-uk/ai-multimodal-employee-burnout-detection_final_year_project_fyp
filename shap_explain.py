"""
Live LOCAL SHAP explanation for one employee's ML prediction.

This module deliberately reuses the SAME preprocessing functions as
ml_predict.py (prepare_original_employee_frame, encode_employee_record,
load_ml_artifacts) instead of re-implementing feature engineering /
one-hot encoding independently. That matters: SHAP values are only valid
if they're computed on the EXACT same encoded feature vector the model
actually saw for its prediction. Any drift between "what SHAP explains"
and "what the model was actually given" would make the explanation
technically wrong — by importing ml_predict's functions directly, that
can't happen; if ml_predict.py's preprocessing ever changes, this module
automatically stays in sync.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")  # Streamlit's server has no display; render to
                        # an in-memory figure only.
import matplotlib.pyplot as plt
import numpy as np
import shap

from ml_predict import (
    DEFAULT_PIPELINE,
    PROJECT_ROOT,
    encode_employee_record,
    load_ml_artifacts,
    normalise_pipeline_name,
    prepare_original_employee_frame,
)

# ---- Plot colours per theme, matching app.py's own LIGHT/DARK palette
# exactly (card_bg, text_primary, card_border), so the SAME waterfall
# plot always reads clearly whichever theme the person currently has
# selected. SHAP's own bar colours (red = raises risk, blue = lowers
# risk) are left as SHAP's defaults on purpose — both are bright,
# saturated colours that stay visible against either background below.
PLOT_THEMES = {
    "light": {
        "fig_facecolor": "#FBFBFF",
        "text_color": "#141414",
        "axis_edge": "#CBD8EE",
    },
    "dark": {
        "fig_facecolor": "#1B2153",
        "text_color": "#EDEFFC",
        "axis_edge": "#2C356F",
    },
}

# Where every generated plot gets saved, so you have a persistent file
# per assessment to cite as evidence for your report — not just an
# in-memory image that disappears when the Streamlit session ends.
SHAP_LOCAL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shap_local"


def explain_employee_prediction(
    employee_data: Mapping[str, Any],
    pipeline_name: str = DEFAULT_PIPELINE,
    max_display: int = 12,
    theme: str = "light",
) -> dict[str, Any]:
    """
    Computes a LOCAL SHAP explanation for one employee's prediction —
    i.e. "which features pushed THIS specific prediction up or down,
    and by how much" — as opposed to a global/dataset-wide explanation.

    theme: "light" or "dark" — matches the Streamlit app's own theme
    toggle, so the plot's background/text colours are always readable
    against whichever theme is currently active. Falls back to "light"
    for any unrecognised value.

    The plot is ALSO saved to disk automatically (see
    SHAP_LOCAL_OUTPUT_DIR), so every explanation you generate leaves a
    permanent PNG file you can reference in your report — the returned
    dict's "saved_plot_path" tells you exactly where.

    Returns a dict:
        {
            "shap_available": bool,
            "waterfall_figure": matplotlib Figure | None,
            "saved_plot_path": str | None,        # where the PNG was saved
            "base_value": float | None,           # model's average output
            "predicted_value": float | None,      # this record's raw model output
            "top_features": list[dict] | None,    # [{feature, feature_value, shap_value}, ...]
            "error": str | None,
        }

    Never raises — on any failure (missing shap package, artefact issues,
    unexpected model type, etc.) it returns shap_available=False with the
    error message, so a SHAP problem never breaks the rest of the app.
    """
    pipeline_name = normalise_pipeline_name(pipeline_name)
    colours = PLOT_THEMES.get(str(theme).lower(), PLOT_THEMES["light"])

    try:
        model, encoder, metadata = load_ml_artifacts(pipeline_name)
        employee_df = prepare_original_employee_frame(employee_data, metadata)
        encoded_employee_df = encode_employee_record(employee_df, encoder, metadata)

        explainer = shap.TreeExplainer(model)
        shap_explanation = explainer(encoded_employee_df)

        # XGBClassifier binary output: some shap/xgboost version
        # combinations return a 3D array (rows, features, classes).
        # Always select the POSITIVE class (the one predict_employee_burnout
        # itself reports as "ml_score"), so the waterfall plot and the
        # displayed probability always agree with each other.
        if len(shap_explanation.values.shape) == 3:
            classes = list(getattr(model, "classes_", [0, 1]))
            positive_class = metadata.get("positive_class", 1)
            class_index = classes.index(positive_class) if positive_class in classes else 1
            single = shap_explanation[0, :, class_index]
        else:
            single = shap_explanation[0]

        # ---- Styling for a nicer, theme-aware waterfall plot ----------
        # SHAP's default plot is functional but plain (small, grey
        # background, tiny fonts, and NOT theme-aware). This section
        # makes it bigger, cleaner, and switches its background/text
        # colours based on the active app theme — without touching
        # SHAP's own red/blue bar colour coding (kept intentionally,
        # since it's intuitive and consistent with the risk banner's
        # colour language in both themes).
        plt.rcParams.update({
            "font.size": 12,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": colours["axis_edge"],
            "axes.labelcolor": colours["text_color"],
            "text.color": colours["text_color"],
            "xtick.color": colours["text_color"],
            "ytick.color": colours["text_color"],
        })

        plt.figure(figsize=(9, 6), dpi=150)
        shap.plots.waterfall(single, max_display=max_display, show=False)

        fig = plt.gcf()
        fig.set_size_inches(9, 6)
        fig.patch.set_facecolor(colours["fig_facecolor"])
        for ax in fig.axes:
            ax.set_facecolor(colours["fig_facecolor"])
            ax.tick_params(colors=colours["text_color"])
            for spine in ax.spines.values():
                spine.set_color(colours["axis_edge"])

        fig.suptitle(
            "How each factor moved this prediction",
            fontsize=14,
            fontweight="bold",
            color=colours["text_color"],
            y=1.02,
        )
        plt.tight_layout()

        # ---- Save the plot to disk (theme baked into the filename, so
        # a Light-theme and a Dark-theme render of the same assessment
        # don't overwrite each other). ----------------------------------
        saved_plot_path: str | None = None
        try:
            SHAP_LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shap_local_{timestamp}_{theme}.png"
            full_path = SHAP_LOCAL_OUTPUT_DIR / filename
            fig.savefig(
                full_path,
                dpi=150,
                bbox_inches="tight",
                facecolor=colours["fig_facecolor"],
            )
            saved_plot_path = str(full_path)
        except Exception:
            # Saving is a bonus, not a requirement — if the folder isn't
            # writable for some reason, the in-memory plot still displays
            # fine in the app; we just won't have a saved copy that time.
            saved_plot_path = None

        feature_names = list(encoded_employee_df.columns)
        feature_values = encoded_employee_df.iloc[0].values
        shap_values = np.asarray(single.values)

        top_features = sorted(
            (
                {
                    "feature": name,
                    "feature_value": float(value),
                    "shap_value": float(contribution),
                }
                for name, value, contribution in zip(
                    feature_names, feature_values, shap_values
                )
            ),
            key=lambda row: abs(row["shap_value"]),
            reverse=True,
        )

        base_value = single.base_values
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]

        return {
            "shap_available": True,
            "waterfall_figure": fig,
            "saved_plot_path": saved_plot_path,
            "base_value": float(base_value),
            "predicted_value": float(base_value + shap_values.sum()),
            "top_features": top_features,
            "error": None,
        }

    except Exception as exc:
        return {
            "shap_available": False,
            "waterfall_figure": None,
            "saved_plot_path": None,
            "base_value": None,
            "predicted_value": None,
            "top_features": None,
            "error": str(exc),
        }


if __name__ == "__main__":
    print(
        "Import explain_employee_prediction() and pass one employee "
        "dictionary — same shape as predict_employee_burnout().\n"
        "Example:\n"
        "from shap_explain import explain_employee_prediction\n"
        "result = explain_employee_prediction(employee_data={...}, theme='light')\n"
        "print(result['top_features'])\n"
        "print(result['saved_plot_path'])"
    )