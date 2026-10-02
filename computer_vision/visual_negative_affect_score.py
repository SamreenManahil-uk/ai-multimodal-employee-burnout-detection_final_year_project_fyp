"""
Computer Vision Visual Negative-Affect Scoring Module.

This module converts averaged facial-emotion probabilities produced by
the DeepFace multi-frame emotion analysis into a continuous visual
negative-affect score between 0 and 1.

Important:
    This score does not diagnose burnout, stress, depression, or anxiety.

    It is only a supporting Computer Vision feature that will later be
    combined with Machine Learning and NLP outputs in the multimodal
    fusion stage.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from computer_vision.config import (
    HAPPINESS_REDUCTION_WEIGHT,
    NEGATIVE_AFFECT_WEIGHTS,
    SUPPORTED_EMOTIONS,
)


def clamp_score(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """
    Restrict a numeric value to a specified range.

    Parameters
    ----------
    value:
        Value to restrict.

    minimum:
        Minimum permitted value.

    maximum:
        Maximum permitted value.

    Returns
    -------
    float
        Value restricted to the range [minimum, maximum].
    """

    return max(minimum, min(float(value), maximum))


def normalise_emotion_probabilities(
    emotion_scores: Mapping[str, float],
) -> Dict[str, float]:
    """
    Validate and normalise facial-emotion probabilities.

    The function accepts probabilities in either:

    - decimal format, such as 0.45
    - percentage format, such as 45.0

    The returned values always sum approximately to 1.0.

    Parameters
    ----------
    emotion_scores:
        Mapping containing facial-emotion names and their probabilities.

    Returns
    -------
    Dict[str, float]
        Normalised probabilities for all supported emotions.

    Raises
    ------
    TypeError
        If emotion_scores is not a mapping.

    ValueError
        If the probabilities are missing, invalid, or sum to zero.
    """

    if not isinstance(emotion_scores, Mapping):
        raise TypeError(
            "emotion_scores must be a dictionary or mapping of "
            "emotion names to probabilities."
        )

    cleaned_scores: Dict[str, float] = {}

    for emotion in SUPPORTED_EMOTIONS:
        raw_value = emotion_scores.get(emotion, 0.0)

        if raw_value is None:
            numeric_value = 0.0
        else:
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                numeric_value = 0.0

        if numeric_value < 0:
            raise ValueError(
                f"Emotion probability cannot be negative: "
                f"{emotion}={numeric_value}"
            )

        cleaned_scores[emotion] = numeric_value

    total_score = sum(cleaned_scores.values())

    if total_score <= 0:
        raise ValueError(
            "Emotion probabilities sum to zero. "
            "A visual score cannot be calculated."
        )

    # DeepFace may return probabilities as percentages, while our averaged
    # output normally uses decimal probabilities. Dividing by the total
    # safely handles both representations.
    normalised_scores = {
        emotion: value / total_score
        for emotion, value in cleaned_scores.items()
    }

    return normalised_scores


def calculate_negative_emotion_mass(
    emotion_scores: Mapping[str, float],
) -> float:
    """
    Calculate the combined probability of explicitly negative emotions.

    Negative emotions included:

    - angry
    - disgust
    - fear
    - sad

    Neutral is not included because neutral expression is not itself a
    negative emotion.

    Parameters
    ----------
    emotion_scores:
        Facial-emotion probabilities.

    Returns
    -------
    float
        Combined negative-emotion probability between 0 and 1.
    """

    probabilities = normalise_emotion_probabilities(emotion_scores)

    negative_emotions = (
        "angry",
        "disgust",
        "fear",
        "sad",
    )

    negative_mass = sum(
        probabilities.get(emotion, 0.0)
        for emotion in negative_emotions
    )

    return clamp_score(negative_mass)


def calculate_weighted_negative_affect(
    emotion_scores: Mapping[str, float],
) -> float:
    """
    Calculate the weighted negative-affect component.

    Each emotion contributes according to NEGATIVE_AFFECT_WEIGHTS
    configured in config.py.

    Parameters
    ----------
    emotion_scores:
        Facial-emotion probabilities.

    Returns
    -------
    float
        Weighted negative-affect value.
    """

    probabilities = normalise_emotion_probabilities(emotion_scores)

    weighted_score = 0.0

    for emotion, weight in NEGATIVE_AFFECT_WEIGHTS.items():
        probability = probabilities.get(emotion, 0.0)
        weighted_score += probability * float(weight)

    return weighted_score


def calculate_visual_negative_affect_score(
    emotion_scores: Mapping[str, float],
) -> Dict[str, Any]:
    """
    Calculate a continuous visual negative-affect score.

    Formula
    -------
    weighted negative affect
    minus
    happy-expression reduction

    The score is restricted to the range 0 to 1.

    The function intentionally does not classify the result as Low or
    Elevated. Binary burnout classification will happen later in the
    multimodal fusion module.

    Parameters
    ----------
    emotion_scores:
        Averaged facial-emotion probabilities produced across valid frames.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:

        - visual_negative_affect_score
        - visual_negative_affect_percentage
        - weighted_negative_component
        - happiness_reduction
        - negative_emotion_mass
        - dominant_emotion
        - dominant_confidence
        - normalised_emotions
    """

    probabilities = normalise_emotion_probabilities(emotion_scores)

    weighted_negative_component = calculate_weighted_negative_affect(
        probabilities
    )

    happy_probability = probabilities.get("happy", 0.0)

    happiness_reduction = (
        happy_probability * float(HAPPINESS_REDUCTION_WEIGHT)
    )

    raw_score = weighted_negative_component - happiness_reduction

    final_score = clamp_score(raw_score)

    negative_emotion_mass = calculate_negative_emotion_mass(
        probabilities
    )

    dominant_emotion = max(
        probabilities,
        key=probabilities.get,
    )

    dominant_confidence = probabilities[dominant_emotion]

    return {
        "visual_negative_affect_score": round(final_score, 4),
        "visual_negative_affect_percentage": round(final_score * 100, 2),
        "weighted_negative_component": round(
            weighted_negative_component,
            4,
        ),
        "happiness_reduction": round(happiness_reduction, 4),
        "negative_emotion_mass": round(negative_emotion_mass, 4),
        "dominant_emotion": dominant_emotion,
        "dominant_confidence": round(dominant_confidence, 4),
        "normalised_emotions": {
            emotion: round(probability, 4)
            for emotion, probability in probabilities.items()
        },
    }


def build_cv_fusion_output(
    emotion_analysis: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build the final Computer Vision output for multimodal fusion.

    A result is treated as valid when at least one frame was analysed and
    averaged emotion probabilities are available. This avoids rejecting a
    successful analysis only because the status field uses a different word
    or is missing.
    """

    if not isinstance(emotion_analysis, Mapping):
        raise TypeError(
            "emotion_analysis must be a dictionary or mapping."
        )

    frames_requested = int(
        emotion_analysis.get("frames_requested", 0) or 0
    )
    frames_analysed = int(
        emotion_analysis.get("frames_analysed", 0) or 0
    )
    frames_rejected = int(
        emotion_analysis.get("frames_rejected", 0) or 0
    )
    valid_frame_ratio = float(
        emotion_analysis.get("valid_frame_ratio", 0.0) or 0.0
    )

    averaged_emotions = (
        emotion_analysis.get("averaged_emotions")
        or emotion_analysis.get("emotion_scores")
        or emotion_analysis.get("emotions")
    )

    if (
        frames_analysed <= 0
        or not isinstance(averaged_emotions, Mapping)
        or not averaged_emotions
    ):
        return {
            "cv_available": False,
            "cv_score": None,
            "cv_score_percentage": None,
            "negative_emotion_mass": None,
            "dominant_emotion": None,
            "dominant_confidence": None,
            "valid_frame_ratio": round(valid_frame_ratio, 4),
            "frames_requested": frames_requested,
            "frames_analysed": frames_analysed,
            "frames_rejected": frames_rejected,
            "reason": (
                emotion_analysis.get("reason")
                or emotion_analysis.get("message")
                or emotion_analysis.get("error")
                or "Computer Vision emotion analysis was unsuccessful."
            ),
        }

    score_result = calculate_visual_negative_affect_score(
        averaged_emotions
    )

    dominant_emotion = (
        emotion_analysis.get("dominant_emotion")
        or score_result["dominant_emotion"]
    )

    dominant_confidence = emotion_analysis.get(
        "dominant_confidence"
    )

    if dominant_confidence is None:
        dominant_confidence = score_result[
            "dominant_confidence"
        ]

    dominant_confidence = float(dominant_confidence)

    if dominant_confidence > 1.0:
        dominant_confidence /= 100.0

    return {
        "cv_available": True,
        "cv_score": score_result[
            "visual_negative_affect_score"
        ],
        "cv_score_percentage": score_result[
            "visual_negative_affect_percentage"
        ],
        "negative_emotion_mass": score_result[
            "negative_emotion_mass"
        ],
        "dominant_emotion": dominant_emotion,
        "dominant_confidence": round(
            dominant_confidence,
            4,
        ),
        "valid_frame_ratio": round(
            valid_frame_ratio,
            4,
        ),
        "frames_requested": frames_requested,
        "frames_analysed": frames_analysed,
        "frames_rejected": frames_rejected,
        "model_name": emotion_analysis.get(
            "model_name",
            "DeepFace pretrained facial-expression model",
        ),
        "detector_backend": emotion_analysis.get(
            "detector_backend",
            "opencv",
        ),
        "score_details": score_result,
        "disclaimer": (
            "This score represents visual facial-expression evidence only. "
            "It does not independently diagnose burnout, stress, depression, "
            "or anxiety."
        ),
    }


def print_visual_score_result(result: Mapping[str, Any]) -> None:
    """
    Print a readable Computer Vision score report.

    Parameters
    ----------
    result:
        Output returned by build_cv_fusion_output().
    """

    print("\n" + "=" * 60)
    print("VISUAL NEGATIVE-AFFECT SCORE")
    print("=" * 60)

    if not result.get("cv_available", False):
        print("Status: Unavailable")
        print(f"Reason: {result.get('reason', 'Unknown error')}")
        return

    print("Status: Successful")
    print(f"CV score: {result['cv_score']:.4f}")
    print(
        f"CV score percentage: "
        f"{result['cv_score_percentage']:.2f}%"
    )
    print(
        f"Negative-emotion mass: "
        f"{result['negative_emotion_mass']:.4f}"
    )
    print(
        f"Dominant emotion: "
        f"{result['dominant_emotion']}"
    )
    print(
        f"Dominant confidence: "
        f"{result['dominant_confidence']:.2%}"
    )
    print(
        f"Valid-frame ratio: "
        f"{result['valid_frame_ratio']:.2%}"
    )

    print("\nScore details")
    print("-" * 60)

    score_details = result["score_details"]

    print(
        "Weighted negative component: "
        f"{score_details['weighted_negative_component']:.4f}"
    )
    print(
        "Happiness reduction: "
        f"{score_details['happiness_reduction']:.4f}"
    )

    print("\nInterpretation:")
    print(
        "This continuous score will be used as a supporting feature "
        "in the multimodal fusion model."
    )
    print(
        "No Low or Elevated burnout classification is made by the "
        "Computer Vision module alone."
    )

    print("\nImportant:")
    print(result["disclaimer"])


def test_visual_negative_affect_score() -> None:
    """
    Test the scoring calculation with sample averaged emotions.

    This test does not open the camera.
    """

    sample_emotions = {
        "angry": 0.0251,
        "disgust": 0.0000,
        "fear": 0.1011,
        "happy": 0.0002,
        "sad": 0.3309,
        "surprise": 0.0000,
        "neutral": 0.5428,
    }

    sample_analysis = {
        "status": "Successful",
        "model_name": "DeepFace pretrained facial-expression model",
        "detector_backend": "opencv",
        "frames_requested": 15,
        "frames_analysed": 15,
        "frames_rejected": 0,
        "valid_frame_ratio": 1.0,
        "averaged_emotions": sample_emotions,
    }

    result = build_cv_fusion_output(sample_analysis)

    print_visual_score_result(result)


if __name__ == "__main__":
    test_visual_negative_affect_score()