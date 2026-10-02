"""
cv_predict.py

Adapter between Streamlit camera/upload input and computer_vision.cv_pipeline.

This version:
- sends the Streamlit image directly to the CV pipeline;
- never asks the CV pipeline to open another camera;
- safely handles None scores;
- prints a complete traceback in the terminal when CV fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import traceback

import numpy as np
from PIL import Image

from computer_vision.cv_pipeline import run_cv_pipeline


def _safe_float(value: Any) -> float | None:
    """Return a finite float, or None when conversion is impossible."""
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    return number


def _normalise_existing_result(result: Any) -> dict[str, Any]:
    """Convert the CV pipeline output into the format expected by app.py."""
    if isinstance(result, (int, float, np.floating)):
        score = _safe_float(result)

        if score is None:
            raise ValueError("The CV pipeline returned an invalid numeric score.")

        if 1.0 < score <= 100.0:
            score /= 100.0

        score = max(0.0, min(1.0, score))

        return {
            "available": True,
            "cv_available": True,
            "cv_score": score,
            "prediction": int(score >= 0.50),
        }

    if not isinstance(result, dict):
        raise TypeError(
            "The CV pipeline must return a dictionary or numeric score."
        )

    if result.get("cv_available") is False:
        reason = (
            result.get("reason")
            or result.get("error")
            or "The CV analysis was unsuccessful."
        )
        raise RuntimeError(str(reason))

    score_keys = (
        "cv_score",
        "visual_negative_affect_score",
        "visual_stress_score",
        "stress_score",
        "burnout_score",
        "probability",
        "score",
    )

    score: float | None = None

    for key in score_keys:
        if key not in result:
            continue

        candidate = _safe_float(result.get(key))

        if candidate is not None:
            score = candidate
            break

    if score is None:
        raise KeyError(
            "CV result does not contain a usable numeric score. "
            f"Expected one of: {list(score_keys)}"
        )

    if 1.0 < score <= 100.0:
        score /= 100.0

    score = max(0.0, min(1.0, score))

    normalised = dict(result)
    normalised.update(
        {
            "available": True,
            "cv_available": True,
            "cv_score": score,
            "cv_score_percentage": round(score * 100.0, 2),
            "prediction": int(score >= 0.50),
        }
    )

    return normalised


def _read_streamlit_image(uploaded_image: Any) -> Image.Image:
    """Read a Streamlit CameraInput/UploadedFile as an independent PIL image."""
    try:
        if hasattr(uploaded_image, "seek"):
            uploaded_image.seek(0)

        image = Image.open(uploaded_image).convert("RGB")
        image.load()
        return image.copy()

    except Exception as exc:
        raise ValueError(
            "The captured image could not be read. Please take the photo again."
        ) from exc


def analyse_face_image(
    uploaded_image: Any,
    project_root: str | Path,
) -> dict[str, Any]:
    """
    Analyse the photo captured inside Streamlit.

    project_root is retained because app.py already supplies it.
    """
    del project_root

    if uploaded_image is None:
        return {
            "available": False,
            "cv_available": False,
            "error": "No face image was supplied.",
        }

    try:
        image = _read_streamlit_image(uploaded_image)

        # Pass the PIL image directly. run_cv_pipeline will not open a webcam.
        raw_result = run_cv_pipeline(
            image=image,
            display_result=False,
        )

        return _normalise_existing_result(raw_result)

    except Exception as exc:
        print("\n" + "=" * 72)
        print("COMPUTER VISION MODULE ERROR")
        print("=" * 72)
        traceback.print_exc()
        print("=" * 72 + "\n")

        return {
            "available": False,
            "cv_available": False,
            "error": f"CV analysis failed: {exc}",
        }