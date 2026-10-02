from __future__ import annotations

from typing import Any


def _extract_score(
    result: dict[str, Any],
    score_key: str,
    availability_keys: tuple[str, ...],
) -> float | None:
    available = any(bool(result.get(key)) for key in availability_keys)

    if not available:
        return None

    if score_key not in result:
        return None

    score = float(result[score_key])
    return max(0.0, min(1.0, score))


def fuse_burnout_scores(
    ml_result: dict[str, Any],
    nlp_result: dict[str, Any],
    cv_result: dict[str, Any],
    weights: dict[str, float] | None = None,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Fuse available ML, NLP and CV scores using a weighted average.

    Missing modalities are excluded and the remaining weights are
    automatically re-normalised.
    """

    if weights is None:
        weights = {
            "ml": 0.50,
            "nlp": 0.30,
            "cv": 0.20,
        }

    raw_scores = {
        "ml": _extract_score(
            ml_result,
            "ml_score",
            ("available", "ml_available"),
        ),
        "nlp": _extract_score(
            nlp_result,
            "nlp_score",
            ("available",),
        ),
        "cv": _extract_score(
            cv_result,
            "cv_score",
            ("available",),
        ),
    }

    available_scores = {
        name: score
        for name, score in raw_scores.items()
        if score is not None
    }

    if not available_scores:
        raise ValueError(
            "Fusion cannot run because ML, NLP and CV are all unavailable."
        )

    selected_weights = {
        name: max(0.0, float(weights.get(name, 0.0)))
        for name in available_scores
    }

    weight_total = sum(selected_weights.values())

    if weight_total <= 0:
        raise ValueError(
            "At least one available modality must have a positive weight."
        )

    normalised_weights = {
        name: weight / weight_total
        for name, weight in selected_weights.items()
    }

    contributions = {
        name: available_scores[name] * normalised_weights[name]
        for name in available_scores
    }

    fusion_score = sum(contributions.values())
    threshold = max(0.0, min(1.0, float(threshold)))
    prediction = int(fusion_score >= threshold)

    if fusion_score < 0.35:
        risk_band = "Low"
    elif fusion_score < 0.65:
        risk_band = "Moderate"
    else:
        risk_band = "High"

    return {
        "available": True,
        "fusion_score": fusion_score,
        "prediction": prediction,
        "risk_band": risk_band,
        "threshold": threshold,
        "scores_used": available_scores,
        "weights_used": normalised_weights,
        "weighted_contributions": contributions,
        "missing_modalities": [
            name for name, score in raw_scores.items()
            if score is None
        ],
        "method": "normalised weighted average",
    }