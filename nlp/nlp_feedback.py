"""
DistilBERT Workplace Feedback Analysis Module

This module uses a DistilBERT model fine-tuned on MNLI for zero-shot
classification. It analyses employee feedback against workplace-related
categories without requiring a separate labelled text dataset.

The module produces:

- Workplace category scores
- Dominant feedback category
- Numerical NLP burnout signal
- Fusion-ready score between 0 and 1
- Binary burnout risk:
    Low Burnout Risk
    Elevated Burnout Risk

Important:
The output is a prototype textual burnout-risk signal.
It is not a clinical or medical diagnosis.
"""

from __future__ import annotations

from typing import Any
from nlp.save_results import save_nlp_result

from transformers import pipeline

# from save_results import save_nlp_result


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_NAME = "typeform/distilbert-base-uncased-mnli"

MODEL_METHOD = "DistilBERT MNLI Zero-Shot Classification"

RISK_THRESHOLD = 0.50

_classifier = None


CANDIDATE_LABELS = [
    "emotional exhaustion and burnout",
    "workplace stress and overload",
    "job dissatisfaction",
    "positive workplace wellbeing",
    "neutral workplace experience",
]


# =========================================================
# MODEL LOADING
# =========================================================

def load_distilbert_model():
    """
    Load the DistilBERT MNLI zero-shot classification model.

    The model is downloaded during the first run.
    Later executions use the locally cached model.
    """

    global _classifier

    if _classifier is None:
        print("=" * 70)
        print("LOADING DISTILBERT WORKPLACE FEEDBACK MODEL")
        print("=" * 70)

        _classifier = pipeline(
            task="zero-shot-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=-1,
        )

        print("DistilBERT model loaded successfully.")

    return _classifier


# =========================================================
# TEXT VALIDATION
# =========================================================

def clean_feedback(feedback: str) -> str:
    """
    Validate and clean employee feedback text.
    """

    if not isinstance(feedback, str):
        raise TypeError("Feedback must be provided as text.")

    cleaned_feedback = " ".join(
        feedback.strip().split()
    )

    if not cleaned_feedback:
        raise ValueError("Feedback cannot be empty.")

    if len(cleaned_feedback) < 10:
        raise ValueError(
            "Feedback is too short. "
            "Please enter a meaningful workplace comment."
        )

    return cleaned_feedback


def normalise_score(score: float) -> float:
    """
    Keep a numerical score between 0 and 1.
    """

    return max(
        0.0,
        min(1.0, float(score)),
    )


# =========================================================
# BURNOUT SIGNAL CALCULATION
# =========================================================

def calculate_burnout_signal(
    scores: dict[str, float],
) -> float:
    """
    Calculate the prototype NLP burnout signal.

    Weighting:

    Emotional exhaustion and burnout = 50%
    Workplace stress and overload    = 30%
    Job dissatisfaction              = 10%
    Positive workplace wellbeing     = -10%

    The score is normalised to the range 0 to 1.
    """

    exhaustion = scores.get(
        "emotional exhaustion and burnout",
        0.0,
    )

    stress = scores.get(
        "workplace stress and overload",
        0.0,
    )

    dissatisfaction = scores.get(
        "job dissatisfaction",
        0.0,
    )

    positive_wellbeing = scores.get(
        "positive workplace wellbeing",
        0.0,
    )

    burnout_signal = (
        0.50 * exhaustion
        + 0.30 * stress
        + 0.10 * dissatisfaction
        - 0.10 * positive_wellbeing
    )

    return round(
        normalise_score(burnout_signal),
        4,
    )


def assign_risk_level(score: float) -> str:
    """
    Convert the NLP signal into the same binary target structure
    used by the structured machine-learning model.
    """

    if score >= RISK_THRESHOLD:
        return "Elevated Burnout Risk"

    return "Low Burnout Risk"


def create_interpretation(
    risk_level: str,
    dominant_category: str,
) -> str:
    """
    Generate a clear interpretation of the NLP result.
    """

    if risk_level == "Elevated Burnout Risk":
        return (
            "The employee feedback contains strong textual evidence "
            "associated with emotional exhaustion, workplace stress, "
            "overload, or burnout. The NLP module therefore identifies "
            "an elevated textual burnout-risk signal."
        )

    if dominant_category == "positive workplace wellbeing":
        return (
            "The employee feedback primarily indicates positive "
            "workplace wellbeing. The NLP module therefore identifies "
            "a low textual burnout-risk signal."
        )

    if dominant_category == "job dissatisfaction":
        return (
            "The feedback indicates job dissatisfaction, but the "
            "textual evidence does not reach the threshold for an "
            "elevated burnout-risk classification."
        )

    if dominant_category == "neutral workplace experience":
        return (
            "The feedback is primarily neutral and does not provide "
            "sufficient evidence of elevated workplace burnout risk."
        )

    return (
        "Some workplace pressure may be present, but the combined "
        "textual evidence remains below the elevated burnout-risk "
        "threshold."
    )


# =========================================================
# COMPLETE NLP ANALYSIS
# =========================================================

def analyse_employee_feedback(
    feedback: str,
) -> dict[str, Any]:
    """
    Analyse employee feedback using DistilBERT zero-shot classification.

    Returns a dictionary containing the full NLP analysis and a
    fusion-ready score.
    """

    cleaned_feedback = clean_feedback(feedback)

    model = load_distilbert_model()

    result = model(
        cleaned_feedback,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template=(
            "This employee feedback indicates {}."
        ),
        multi_label=True,
    )

    scores = {
        label: round(float(score), 4)
        for label, score in zip(
            result["labels"],
            result["scores"],
        )
    }

    burnout_signal = calculate_burnout_signal(
        scores
    )

    dominant_category = max(
        scores,
        key=scores.get,
    )

    risk_level = assign_risk_level(
        burnout_signal
    )

    interpretation = create_interpretation(
        risk_level=risk_level,
        dominant_category=dominant_category,
    )

    return {
        "model_name": MODEL_NAME,
        "model_method": MODEL_METHOD,
        "feedback": cleaned_feedback,
        "category_scores": scores,
        "dominant_category": dominant_category,
        "nlp_burnout_signal": burnout_signal,
        "fusion_score": burnout_signal,
        "risk_threshold": RISK_THRESHOLD,
        "nlp_risk_level": risk_level,
        "interpretation": interpretation,
    }


# =========================================================
# RESULT DISPLAY
# =========================================================

def display_result(
    result: dict[str, Any],
) -> None:
    """
    Display the complete NLP analysis in the terminal.
    """

    print("\n" + "=" * 70)
    print("EMPLOYEE WORKPLACE FEEDBACK ANALYSIS")
    print("=" * 70)

    print("\nModel:")
    print(result["model_method"])

    print("\nFeedback:")
    print(result["feedback"])

    print("\nWorkplace category scores:")

    for category, score in sorted(
        result["category_scores"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"- {category.title()}: "
            f"{score:.2%}"
        )

    print("\nDominant feedback category:")
    print(
        result["dominant_category"].title()
    )

    print("\nCombined NLP burnout signal:")
    print(
        f'{result["nlp_burnout_signal"]:.2%}'
    )

    print("\nFusion score:")
    print(
        f'{result["fusion_score"]:.4f}'
    )

    print("\nBinary NLP risk level:")
    print(result["nlp_risk_level"])

    print("\nInterpretation:")
    print(result["interpretation"])

    print("=" * 70)


# =========================================================
# MAIN PROGRAM
# =========================================================

def main() -> None:
    """
    Run the NLP module interactively from the terminal.
    """

    print(
        "\nAI-Based Workplace Burnout Risk "
        "Prediction System"
    )

    print(
        "DistilBERT Employee Feedback Module"
    )

    print("-" * 70)

    feedback = input(
        "Enter employee workplace feedback:\n> "
    )

    try:
        result = analyse_employee_feedback(
            feedback
        )

        display_result(result)

        save_nlp_result(
            result=result,
            source="Interactive Input",
        )

    except (TypeError, ValueError) as error:
        print(f"\nInput error: {error}")

    except Exception as error:
        print(
            "\nThe NLP analysis could not be completed."
        )
        print(f"Technical details: {error}")


if __name__ == "__main__":
    main()