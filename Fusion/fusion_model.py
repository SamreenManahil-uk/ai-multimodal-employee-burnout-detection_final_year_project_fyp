"""
Multimodal Burnout Fusion Module.

This module combines continuous outputs from:

1. Machine Learning:
   Structured employee questionnaire probability.

2. Natural Language Processing:
   Burnout-related signal from written employee feedback.

3. Computer Vision:
   Supporting visual negative-affect signal.

Default formula:

    final_score = (
        0.50 * ml_score
        + 0.30 * nlp_score
        + 0.20 * cv_score
    )

The Computer Vision result is supporting evidence only.
It does not independently diagnose burnout or any mental-health condition.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FUSION_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
)

DEFAULT_RESULT_PATH = (
    FUSION_OUTPUT_DIRECTORY
    / "latest_fusion_result.json"
)


# ==========================================================
# FUSION CONFIGURATION
# ==========================================================

DEFAULT_WEIGHTS = {
    "ml": 0.50,
    "nlp": 0.30,
    "cv": 0.20,
}

DEFAULT_DECISION_THRESHOLD = 0.50

LOW_RISK_UPPER_BOUND = 0.40
HIGH_RISK_LOWER_BOUND = 0.70

SUPPORTED_MODALITIES = (
    "ml",
    "nlp",
    "cv",
)


# ==========================================================
# GENERAL VALIDATION
# ==========================================================

def validate_score(
    value: Any,
    score_name: str,
) -> float:
    """
    Validate that a modality score is a finite number
    between 0 and 1.
    """

    if isinstance(value, bool):
        raise TypeError(
            f"{score_name} must be a numeric probability, "
            "not a Boolean value."
        )

    try:
        score = float(value)

    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{score_name} must be numeric. "
            f"Received: {value!r}"
        ) from exc

    if not math.isfinite(score):
        raise ValueError(
            f"{score_name} must be finite. "
            f"Received: {score}"
        )

    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"{score_name} must be between 0 and 1. "
            f"Received: {score}"
        )

    return score


def validate_weights(
    weights: Mapping[str, float],
) -> dict[str, float]:
    """
    Validate the configured fusion weights.

    The supplied weights do not have to sum exactly to 1,
    because they are normalised before fusion.
    """

    if not isinstance(weights, Mapping):
        raise TypeError(
            "weights must be a dictionary or mapping."
        )

    expected_keys = set(
        SUPPORTED_MODALITIES
    )

    supplied_keys = set(
        weights.keys()
    )

    missing_keys = sorted(
        expected_keys - supplied_keys
    )

    unexpected_keys = sorted(
        supplied_keys - expected_keys
    )

    if missing_keys:
        raise ValueError(
            "Missing fusion weight(s): "
            + ", ".join(missing_keys)
        )

    if unexpected_keys:
        raise ValueError(
            "Unexpected fusion weight(s): "
            + ", ".join(unexpected_keys)
        )

    validated_weights = {}

    for modality in SUPPORTED_MODALITIES:

        raw_weight = weights[modality]

        if isinstance(raw_weight, bool):
            raise TypeError(
                f"Weight for '{modality}' must be numeric."
            )

        try:
            weight = float(raw_weight)

        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Weight for '{modality}' must be numeric."
            ) from exc

        if not math.isfinite(weight):
            raise ValueError(
                f"Weight for '{modality}' must be finite."
            )

        if weight < 0:
            raise ValueError(
                f"Weight for '{modality}' cannot be negative."
            )

        validated_weights[modality] = weight

    if sum(validated_weights.values()) <= 0:
        raise ValueError(
            "At least one fusion weight must be greater than zero."
        )

    return validated_weights


def extract_availability(
    result: Mapping[str, Any],
    availability_key: str,
    default: bool = True,
) -> bool:
    """
    Extract a modality availability flag.
    """

    available = result.get(
        availability_key,
        default,
    )

    if not isinstance(available, bool):
        raise TypeError(
            f"'{availability_key}' must be True or False."
        )

    return available


# ==========================================================
# MACHINE LEARNING SCORE
# ==========================================================

def extract_ml_score(
    ml_result: Mapping[str, Any] | None,
) -> tuple[bool, float | None, str | None]:
    """
    Extract the structured-data ML probability.

    Expected key:
        ml_score
    """

    if ml_result is None:
        return (
            False,
            None,
            "Machine Learning result was not supplied.",
        )

    if not isinstance(ml_result, Mapping):
        raise TypeError(
            "ml_result must be a dictionary, mapping, or None."
        )

    available = extract_availability(
        result=ml_result,
        availability_key="ml_available",
        default=True,
    )

    if not available:
        return (
            False,
            None,
            str(
                ml_result.get(
                    "reason",
                    "Machine Learning result is unavailable.",
                )
            ),
        )

    if "ml_score" not in ml_result:
        raise KeyError(
            "The available Machine Learning result "
            "does not contain 'ml_score'."
        )

    score = validate_score(
        ml_result["ml_score"],
        "ml_score",
    )

    return (
        True,
        score,
        None,
    )


# ==========================================================
# NLP SCORE
# ==========================================================

def extract_nlp_score(
    nlp_result: Mapping[str, Any] | None,
) -> tuple[bool, float | None, str | None]:
    """
    Extract the NLP burnout signal.

    Preferred key:
        fusion_score

    Compatibility keys:
        nlp_score
        nlp_burnout_score
    """

    if nlp_result is None:
        return (
            False,
            None,
            "NLP result was not supplied.",
        )

    if not isinstance(nlp_result, Mapping):
        raise TypeError(
            "nlp_result must be a dictionary, mapping, or None."
        )

    available = extract_availability(
        result=nlp_result,
        availability_key="nlp_available",
        default=True,
    )

    if not available:
        return (
            False,
            None,
            str(
                nlp_result.get(
                    "reason",
                    "NLP result is unavailable.",
                )
            ),
        )

    supported_score_keys = (
        "fusion_score",
        "nlp_score",
        "nlp_burnout_score",
    )

    score_key = next(
        (
            key
            for key in supported_score_keys
            if key in nlp_result
        ),
        None,
    )

    if score_key is None:
        raise KeyError(
            "The NLP result must contain one of: "
            + ", ".join(supported_score_keys)
        )

    score = validate_score(
        nlp_result[score_key],
        score_key,
    )

    return (
        True,
        score,
        None,
    )


# ==========================================================
# COMPUTER VISION SCORE
# ==========================================================

def extract_cv_score(
    cv_result: Mapping[str, Any] | None,
) -> tuple[bool, float | None, str | None]:
    """
    Extract the supporting CV negative-affect score.

    Expected key:
        cv_score
    """

    if cv_result is None:
        return (
            False,
            None,
            "Computer Vision result was not supplied.",
        )

    if not isinstance(cv_result, Mapping):
        raise TypeError(
            "cv_result must be a dictionary, mapping, or None."
        )

    available = extract_availability(
        result=cv_result,
        availability_key="cv_available",
        default=True,
    )

    if not available:
        return (
            False,
            None,
            str(
                cv_result.get(
                    "reason",
                    "Computer Vision result is unavailable.",
                )
            ),
        )

    if "cv_score" not in cv_result:
        raise KeyError(
            "The available Computer Vision result "
            "does not contain 'cv_score'."
        )

    score = validate_score(
        cv_result["cv_score"],
        "cv_score",
    )

    return (
        True,
        score,
        None,
    )


# ==========================================================
# WEIGHT NORMALISATION
# ==========================================================

def normalise_available_weights(
    configured_weights: Mapping[str, float],
    availability: Mapping[str, bool],
) -> dict[str, float]:
    """
    Re-normalise weights across available modalities.

    Example:

        ML weight  = 0.50
        NLP weight = 0.30
        CV missing

        Available weight total = 0.80

        Effective ML weight  = 0.50 / 0.80
        Effective NLP weight = 0.30 / 0.80
        Effective CV weight  = 0
    """

    validated_weights = validate_weights(
        configured_weights
    )

    available_weight_total = sum(
        validated_weights[modality]
        for modality in SUPPORTED_MODALITIES
        if availability.get(
            modality,
            False,
        )
    )

    if available_weight_total <= 0:
        raise ValueError(
            "No positively weighted modality is "
            "available for fusion."
        )

    effective_weights = {}

    for modality in SUPPORTED_MODALITIES:

        if availability.get(
            modality,
            False,
        ):
            effective_weights[modality] = (
                validated_weights[modality]
                / available_weight_total
            )

        else:
            effective_weights[modality] = 0.0

    return effective_weights


# ==========================================================
# RISK CLASSIFICATION
# ==========================================================

def assign_risk_level(
    final_score: float,
) -> str:
    """
    Convert the fusion score into a descriptive risk level.

    Low:
        score < 0.40

    Moderate:
        0.40 <= score < 0.70

    High:
        score >= 0.70
    """

    score = validate_score(
        final_score,
        "final_score",
    )

    if score >= HIGH_RISK_LOWER_BOUND:
        return "High"

    if score >= LOW_RISK_UPPER_BOUND:
        return "Moderate"

    return "Low"


def create_interpretation(
    risk_level: str,
    modality_scores: Mapping[str, float | None],
    effective_weights: Mapping[str, float],
) -> str:
    """
    Create a cautious and non-diagnostic interpretation.
    """

    available_scores = {
        modality: score
        for modality, score in modality_scores.items()
        if score is not None
        and effective_weights.get(
            modality,
            0.0,
        ) > 0
    }

    strongest_modality = max(
        available_scores,
        key=available_scores.get,
    )

    modality_names = {
        "ml": "structured questionnaire data",
        "nlp": "written employee feedback",
        "cv": "visual facial-expression evidence",
    }

    if risk_level == "High":
        first_sentence = (
            "The combined system indicates a high "
            "burnout-risk signal."
        )

    elif risk_level == "Moderate":
        first_sentence = (
            "The combined system indicates a moderate "
            "burnout-risk signal."
        )

    else:
        first_sentence = (
            "The combined system indicates a low "
            "burnout-risk signal."
        )

    return (
        f"{first_sentence} The strongest available signal "
        f"came from {modality_names[strongest_modality]}. "
        "This result is intended for early risk screening "
        "and decision support, not clinical diagnosis."
    )


# ==========================================================
# MAIN FUSION FUNCTION
# ==========================================================

def fuse_burnout_scores(
    ml_result: Mapping[str, Any] | None,
    nlp_result: Mapping[str, Any] | None,
    cv_result: Mapping[str, Any] | None,
    weights: Mapping[str, float] | None = None,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    require_ml: bool = True,
) -> dict[str, Any]:
    """
    Combine ML, NLP, and CV outputs.

    Parameters
    ----------
    ml_result:
        Output from predict_ml_burnout().

    nlp_result:
        Output from the NLP employee-feedback analyser.

    cv_result:
        Output from build_cv_fusion_output().

    weights:
        Optional fusion weights.

    decision_threshold:
        Binary classification threshold.

    require_ml:
        When True, the primary ML modality must be available.

    Returns
    -------
    dict
        Final multimodal burnout result.
    """

    configured_weights = validate_weights(
        weights or DEFAULT_WEIGHTS
    )

    threshold = validate_score(
        decision_threshold,
        "decision_threshold",
    )

    (
        ml_available,
        ml_score,
        ml_reason,
    ) = extract_ml_score(
        ml_result
    )

    (
        nlp_available,
        nlp_score,
        nlp_reason,
    ) = extract_nlp_score(
        nlp_result
    )

    (
        cv_available,
        cv_score,
        cv_reason,
    ) = extract_cv_score(
        cv_result
    )

    if require_ml and not ml_available:
        raise ValueError(
            "Machine Learning is the primary prediction "
            "modality and is required for fusion. "
            f"Reason: {ml_reason}"
        )

    availability = {
        "ml": ml_available,
        "nlp": nlp_available,
        "cv": cv_available,
    }

    modality_scores = {
        "ml": ml_score,
        "nlp": nlp_score,
        "cv": cv_score,
    }

    if not any(
        availability.values()
    ):
        raise ValueError(
            "No modality is available for fusion."
        )

    effective_weights = normalise_available_weights(
        configured_weights=configured_weights,
        availability=availability,
    )

    contributions = {}

    for modality in SUPPORTED_MODALITIES:

        score = modality_scores[modality]

        if score is None:
            contributions[modality] = 0.0

        else:
            contributions[modality] = (
                score
                * effective_weights[modality]
            )

    final_score = sum(
        contributions.values()
    )

    final_score = max(
        0.0,
        min(final_score, 1.0),
    )

    binary_prediction = int(
        final_score >= threshold
    )

    risk_level = assign_risk_level(
        final_score
    )

    available_modalities = [
        modality
        for modality in SUPPORTED_MODALITIES
        if availability[modality]
    ]

    unavailable_modalities = [
        modality
        for modality in SUPPORTED_MODALITIES
        if not availability[modality]
    ]

    unavailable_reasons = {
        "ml": ml_reason,
        "nlp": nlp_reason,
        "cv": cv_reason,
    }

    unavailable_reasons = {
        modality: reason
        for modality, reason
        in unavailable_reasons.items()
        if reason is not None
    }

    interpretation = create_interpretation(
        risk_level=risk_level,
        modality_scores=modality_scores,
        effective_weights=effective_weights,
    )

    return {
        "fusion_available": True,

        "fusion_method": (
            "weighted_score_level_fusion"
        ),

        "final_score": round(
            final_score,
            4,
        ),

        "final_score_percentage": round(
            final_score * 100,
            2,
        ),

        "binary_prediction": (
            binary_prediction
        ),

        "binary_label": (
            "Elevated burnout risk"
            if binary_prediction == 1
            else "Low burnout risk"
        ),

        "decision_threshold": round(
            threshold,
            4,
        ),

        "risk_level": risk_level,

        "risk_boundaries": {
            "low": (
                f"score < "
                f"{LOW_RISK_UPPER_BOUND:.2f}"
            ),

            "moderate": (
                f"{LOW_RISK_UPPER_BOUND:.2f} "
                f"<= score < "
                f"{HIGH_RISK_LOWER_BOUND:.2f}"
            ),

            "high": (
                f"score >= "
                f"{HIGH_RISK_LOWER_BOUND:.2f}"
            ),
        },

        "scores": {
            "ml_score": (
                round(ml_score, 4)
                if ml_score is not None
                else None
            ),

            "nlp_score": (
                round(nlp_score, 4)
                if nlp_score is not None
                else None
            ),

            "cv_score": (
                round(cv_score, 4)
                if cv_score is not None
                else None
            ),
        },

        "configured_weights": {
            modality: round(
                configured_weights[modality],
                4,
            )
            for modality in SUPPORTED_MODALITIES
        },

        "effective_weights": {
            modality: round(
                effective_weights[modality],
                4,
            )
            for modality in SUPPORTED_MODALITIES
        },

        "weighted_contributions": {
            "ml_contribution": round(
                contributions["ml"],
                4,
            ),

            "nlp_contribution": round(
                contributions["nlp"],
                4,
            ),

            "cv_contribution": round(
                contributions["cv"],
                4,
            ),
        },

        "available_modalities": (
            available_modalities
        ),

        "unavailable_modalities": (
            unavailable_modalities
        ),

        "unavailable_reasons": (
            unavailable_reasons
        ),

        "all_modalities_available": (
            len(available_modalities)
            == len(SUPPORTED_MODALITIES)
        ),

        "interpretation": (
            interpretation
        ),

        "disclaimer": (
            "This system estimates workplace burnout "
            "risk for research and decision-support "
            "purposes. It is not a medical or "
            "psychological diagnosis. The Computer "
            "Vision component represents supporting "
            "facial-expression evidence only."
        ),

        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }


# ==========================================================
# SAVE RESULT
# ==========================================================

def save_fusion_result(
    result: Mapping[str, Any],
    output_path: str | Path = DEFAULT_RESULT_PATH,
) -> Path:
    """
    Save a fusion result as JSON.
    """

    if not isinstance(
        result,
        Mapping,
    ):
        raise TypeError(
            "result must be a dictionary or mapping."
        )

    path = Path(
        output_path
    )

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            dict(result),
            file,
            indent=4,
            ensure_ascii=False,
        )

    return path


# ==========================================================
# DISPLAY RESULT
# ==========================================================

def print_fusion_result(
    result: Mapping[str, Any],
) -> None:
    """
    Print a readable fusion report.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MULTIMODAL BURNOUT FUSION RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"Final score: "
        f"{result['final_score']:.4f}"
    )

    print(
        f"Final percentage: "
        f"{result['final_score_percentage']:.2f}%"
    )

    print(
        f"Risk level: "
        f"{result['risk_level']}"
    )

    print(
        f"Binary result: "
        f"{result['binary_label']}"
    )

    print(
        f"Decision threshold: "
        f"{result['decision_threshold']:.2f}"
    )

    print(
        "\nModality scores"
    )

    print(
        "-" * 70
    )

    for name, score in result[
        "scores"
    ].items():

        if score is None:
            displayed_score = "Unavailable"

        else:
            displayed_score = (
                f"{score:.4f}"
            )

        print(
            f"{name}: "
            f"{displayed_score}"
        )

    print(
        "\nEffective weights"
    )

    print(
        "-" * 70
    )

    for name, weight in result[
        "effective_weights"
    ].items():

        print(
            f"{name}: "
            f"{weight:.4f}"
        )

    print(
        "\nWeighted contributions"
    )

    print(
        "-" * 70
    )

    for name, contribution in result[
        "weighted_contributions"
    ].items():

        print(
            f"{name}: "
            f"{contribution:.4f}"
        )

    print(
        "\nInterpretation"
    )

    print(
        "-" * 70
    )

    print(
        result["interpretation"]
    )

    print(
        "\nImportant"
    )

    print(
        "-" * 70
    )

    print(
        result["disclaimer"]
    )


# ==========================================================
# CALCULATION-ONLY TEST
# ==========================================================

def test_fusion_model() -> None:
    """
    Test the fusion calculation without loading any model.
    """

    sample_ml_result = {
        "ml_available": True,
        "ml_score": 0.72,
        "prediction": 1,
    }

    sample_nlp_result = {
        "nlp_available": True,
        "fusion_score": 0.65,
    }

    sample_cv_result = {
        "cv_available": True,
        "cv_score": 0.30,
        "dominant_emotion": "neutral",
    }

    result = fuse_burnout_scores(
        ml_result=sample_ml_result,
        nlp_result=sample_nlp_result,
        cv_result=sample_cv_result,
    )

    print_fusion_result(
        result
    )

    saved_path = save_fusion_result(
        result
    )

    print(
        "\nFusion test result saved to:"
    )

    print(
        saved_path
    )


if __name__ == "__main__":
    test_fusion_model()