"""
emotion_detection.py

This module analyses facial emotions using the pretrained
DeepFace emotion recognition model.

Pipeline
--------
Input Frames
      ↓
DeepFace Emotion Recognition
      ↓
Emotion Probabilities
      ↓
Normalisation
      ↓
Return Clean Emotion Scores
"""

from typing import Any, Dict, List

import numpy as np
from deepface import DeepFace

from computer_vision.config import (
    ALIGN_FACE,
    DEEPFACE_ACTIONS,
    DETECTOR_BACKEND,
    ENFORCE_DETECTION,
    SUPPORTED_EMOTIONS,
)


# ==========================================================
# Helper Function
# ==========================================================

def normalise_emotion_scores(
    raw_emotions: Dict[str, Any],
) -> Dict[str, float]:
    """
    Convert DeepFace emotion percentages into probabilities
    whose total equals 1.0.

    Missing, None, invalid, negative, NaN and infinite values
    are safely treated as zero.
    """

    if not isinstance(raw_emotions, dict):
        raise TypeError(
            "DeepFace emotion scores must be returned as a dictionary."
        )

    cleaned_scores: Dict[str, float] = {}

    for emotion in SUPPORTED_EMOTIONS:
        raw_value = raw_emotions.get(emotion, 0.0)

        if raw_value is None:
            numeric_value = 0.0
        else:
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                numeric_value = 0.0

        if not np.isfinite(numeric_value):
            numeric_value = 0.0

        if numeric_value < 0.0:
            numeric_value = 0.0

        cleaned_scores[emotion] = numeric_value

    total = sum(cleaned_scores.values())

    if total <= 0.0:
        raise RuntimeError(
            "DeepFace did not return usable facial-emotion scores. "
            "Please use a clear, front-facing image with suitable lighting."
        )

    return {
        emotion: value / total
        for emotion, value in cleaned_scores.items()
    }



# ==========================================================
# Analyse One Frame
# ==========================================================

def analyse_single_frame(
    frame: np.ndarray,
) -> Dict[str, Any]:
    """
    Analyse one supplied image frame using the pretrained
    DeepFace emotion model.

    Parameters
    ----------
    frame : numpy.ndarray

    Returns
    -------
    dict
        Emotion probabilities
        Dominant emotion
        Confidence
        Face region
    """

    if frame is None:
        raise ValueError(
            "Input frame is None."
        )

    if not isinstance(frame, np.ndarray):
        raise TypeError(
            "Frame must be a NumPy array."
        )

    if frame.size == 0:
        raise ValueError(
            "Input frame is empty."
        )

    analysis = DeepFace.analyze(

        img_path=frame,

        actions=DEEPFACE_ACTIONS,

        detector_backend=DETECTOR_BACKEND,

        enforce_detection=ENFORCE_DETECTION,

        align=ALIGN_FACE,

        silent=True,

    )

    if isinstance(analysis, list):

        if len(analysis) == 0:

            raise RuntimeError(
                "No face was detected."
            )

        result = analysis[0]

    elif isinstance(analysis, dict):

        result = analysis

    else:

        raise RuntimeError(
            "Unexpected DeepFace response."
        )

    raw_emotions = result.get("emotion")

    if not isinstance(raw_emotions, dict):

        raise RuntimeError(
            "DeepFace did not return emotion scores."
        )

    emotion_scores = normalise_emotion_scores(
        raw_emotions
    )

    # Calculate dominant emotion ourselves
    dominant_emotion = max(
        emotion_scores,
        key=emotion_scores.get,
    )

    dominant_confidence = emotion_scores[
        dominant_emotion
    ]

    return {

        "emotion_scores": emotion_scores,

        "dominant_emotion": dominant_emotion,

        "dominant_confidence": dominant_confidence,

        "face_region": result.get(
            "region",
            {},
        ),

    }


# ==========================================================
# Analyse Multiple Frames
# (Continues in Part 2)
# ==========================================================
def analyse_frame_sequence(
    frames: List[np.ndarray],
) -> Dict[str, Any]:
    """
    Analyse multiple supplied frames and calculate one stable
    averaged facial-emotion result.

    Invalid frames are rejected when:
    - no face is detected,
    - DeepFace fails,
    - emotion scores are missing,
    - or the frame is otherwise unusable.

    Parameters
    ----------
    frames : list[numpy.ndarray]
        Webcam frames captured over several seconds.

    Returns
    -------
    dict
        Averaged emotion probabilities,
        recalculated dominant emotion,
        frame-quality information,
        and model metadata.
    """

    if not isinstance(frames, list):
        raise TypeError(
            "Frames must be provided as a list."
        )

    if len(frames) == 0:
        raise ValueError(
            "No frames were provided for analysis."
        )

    # Store the total probability for each emotion.
    accumulated_emotions: Dict[str, float] = {
        emotion: 0.0
        for emotion in SUPPORTED_EMOTIONS
    }

    valid_results: List[Dict[str, Any]] = []

    failed_frames: List[Dict[str, Any]] = []

    # ======================================================
    # ANALYSE EACH FRAME
    # ======================================================

    for frame_number, frame in enumerate(
        frames,
        start=1,
    ):

        try:

            frame_result = analyse_single_frame(
                frame
            )

            valid_results.append(
                frame_result
            )

            for emotion in SUPPORTED_EMOTIONS:

                accumulated_emotions[
                    emotion
                ] += frame_result[
                    "emotion_scores"
                ][emotion]

        except Exception as error:

            failed_frames.append(
                {
                    "frame_number": frame_number,
                    "reason": str(error),
                }
            )

    # ======================================================
    # FRAME-QUALITY INFORMATION
    # ======================================================

    frames_requested = len(frames)

    valid_frame_count = len(
        valid_results
    )

    rejected_frame_count = len(
        failed_frames
    )

    valid_frame_ratio = (
        valid_frame_count
        / frames_requested
    )

    # Import here to keep all thresholds controlled
    # by config.py.
    from computer_vision.config import (
        MIN_VALID_FRAME_RATIO,
        MIN_VALID_FRAMES,
    )

    # Reject the analysis when there is insufficient
    # reliable facial information.
    if (
        valid_frame_count < MIN_VALID_FRAMES
        or valid_frame_ratio
        < MIN_VALID_FRAME_RATIO
    ):

        raise ValueError(
            "Insufficient valid facial data. "
            f"Only {valid_frame_count} out of "
            f"{frames_requested} frames were valid. "
            "Please face the camera clearly, use suitable "
            "lighting and repeat the analysis."
        )

    # ======================================================
    # AVERAGE EMOTION PROBABILITIES
    # ======================================================

    averaged_emotions: Dict[str, float] = {

        emotion: (
            accumulated_emotions[emotion]
            / valid_frame_count
        )

        for emotion in SUPPORTED_EMOTIONS
    }

    # Normalise again to prevent tiny floating-point
    # differences after averaging.
    averaged_total = sum(
        averaged_emotions.values()
    )

    if averaged_total <= 0:

        raise RuntimeError(
            "The averaged emotion probabilities "
            "were invalid."
        )

    averaged_emotions = {

        emotion: value / averaged_total

        for emotion, value
        in averaged_emotions.items()
    }

    # ======================================================
    # CALCULATE FINAL DOMINANT EMOTION
    # ======================================================

    final_dominant_emotion = max(
        averaged_emotions,
        key=averaged_emotions.get,
    )

    final_dominant_confidence = (
        averaged_emotions[
            final_dominant_emotion
        ]
    )

    # ======================================================
    # RETURN STABLE MULTI-FRAME RESULT
    # ======================================================

    return {

        "model_name": (
            "DeepFace pretrained "
            "facial-expression model"
        ),

        "detector_backend": (
            DETECTOR_BACKEND
        ),

        "frames_requested": (
            frames_requested
        ),

        "frames_analysed": (
            valid_frame_count
        ),

        "frames_rejected": (
            rejected_frame_count
        ),

        "valid_frame_ratio": round(
            valid_frame_ratio,
            4,
        ),

        "averaged_emotions": (
            averaged_emotions
        ),

        "dominant_emotion": (
            final_dominant_emotion
        ),

        "dominant_confidence": round(
            final_dominant_confidence,
            4,
        ),

        "failed_frame_details": (
            failed_frames
        ),

    }
