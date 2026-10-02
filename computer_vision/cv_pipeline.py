"""
Streamlit-compatible Computer Vision pipeline.

This module analyses one supplied image. It never opens a webcam and never
calls cv2.VideoCapture(), cv2.imshow(), or cv2.waitKey().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image

from computer_vision.emotion_detection import analyse_single_frame
from computer_vision.visual_negative_affect_score import build_cv_fusion_output


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    return number if np.isfinite(number) else default


def _load_frame(
    image_path: str | Path | None = None,
    image: Any | None = None,
    frame: np.ndarray | None = None,
) -> np.ndarray:
    supplied = sum(value is not None for value in (image_path, image, frame))

    if supplied == 0:
        raise ValueError(
            "No image was supplied. Capture or upload an image in Streamlit."
        )

    # Prefer frame, then image, then path when more than one is supplied.
    if frame is not None:
        array = np.asarray(frame).copy()

        if array.size == 0:
            raise ValueError("The supplied image frame is empty.")

        if array.ndim == 2:
            return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)

        if array.ndim != 3:
            raise ValueError("The supplied frame has an unsupported shape.")

        if array.shape[2] == 4:
            return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)

        if array.shape[2] != 3:
            raise ValueError("The supplied frame must have 3 or 4 channels.")

        # NumPy arrays passed by this project originate from PIL and are RGB.
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    if image is not None:
        if isinstance(image, np.ndarray):
            return _load_frame(frame=image)

        try:
            pil_image = (
                image.copy()
                if isinstance(image, Image.Image)
                else Image.open(image)
            ).convert("RGB")
        except Exception as exc:
            raise ValueError("The supplied image could not be opened.") from exc

        return cv2.cvtColor(
            np.asarray(pil_image),
            cv2.COLOR_RGB2BGR,
        )

    path = Path(image_path)  # type: ignore[arg-type]
    if not path.exists():
        raise FileNotFoundError(f"CV image was not found: {path}")

    loaded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if loaded is None or loaded.size == 0:
        raise ValueError("OpenCV could not read the supplied image.")

    return loaded


def _build_single_image_analysis(
    single_result: Mapping[str, Any],
) -> dict[str, Any]:
    emotion_scores = single_result.get("emotion_scores")

    if not isinstance(emotion_scores, Mapping) or not emotion_scores:
        raise RuntimeError(
            "The emotion model did not return usable emotion probabilities."
        )

    cleaned = {
        str(name): max(0.0, _safe_float(value))
        for name, value in emotion_scores.items()
    }

    total = sum(cleaned.values())
    if total <= 0.0:
        raise RuntimeError("All returned emotion probabilities were zero.")

    cleaned = {
        name: value / total
        for name, value in cleaned.items()
    }

    dominant_emotion = str(
        single_result.get("dominant_emotion")
        or max(cleaned, key=cleaned.get)
    )

    dominant_confidence = _safe_float(
        single_result.get("dominant_confidence"),
        cleaned.get(dominant_emotion, 0.0),
    )

    if dominant_confidence > 1.0:
        dominant_confidence /= 100.0

    dominant_confidence = max(0.0, min(1.0, dominant_confidence))

    return {
        "status": "success",
        "model_name": "DeepFace pretrained facial-expression model",
        "detector_backend": "opencv",
        "frames_requested": 1,
        "frames_analysed": 1,
        "frames_rejected": 0,
        "valid_frame_ratio": 1.0,
        "averaged_emotions": cleaned,
        "dominant_emotion": dominant_emotion,
        "dominant_confidence": dominant_confidence,
        "face_region": single_result.get("face_region", {}),
        "failed_frame_details": [],
    }


def run_cv_pipeline(
    image_path: str | Path | None = None,
    image: Any | None = None,
    frame: np.ndarray | None = None,
    display_result: bool = False,
) -> dict[str, Any]:
    del display_result

    cv_frame = _load_frame(
        image_path=image_path,
        image=image,
        frame=frame,
    )

    single_result = analyse_single_frame(cv_frame)
    emotion_analysis = _build_single_image_analysis(single_result)
    fusion_output = build_cv_fusion_output(emotion_analysis)

    cv_score = _safe_float(
        fusion_output.get("cv_score"),
        _safe_float(
            fusion_output.get("visual_negative_affect_score"),
            0.0,
        ),
    )
    cv_score = max(0.0, min(1.0, cv_score))

    result = dict(fusion_output)
    result.update(
        {
            "available": True,
            "cv_available": True,
            "cv_score": cv_score,
            "cv_score_percentage": round(cv_score * 100.0, 2),
            "prediction": int(cv_score >= 0.50),
            "emotion_analysis": emotion_analysis,
        }
    )
    return result


if __name__ == "__main__":
    raise SystemExit(
        "Run this module through app.py; it does not provide webcam capture."
    )
