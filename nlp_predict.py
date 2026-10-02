from __future__ import annotations

from typing import Any

from nlp.nlp_feedback import analyse_employee_feedback as run_distilbert_analysis


def analyse_employee_feedback(
    feedback_text: str,
    project_root=None,
) -> dict[str, Any]:
    """
    Run the genuine DistilBERT zero-shot NLP module.

    project_root is accepted only for compatibility with the
    existing Streamlit call. It is not required by DistilBERT.
    """

    try:
        result = run_distilbert_analysis(feedback_text)

        return {
            **result,
            "available": True,
            "nlp_score": float(result["fusion_score"]),
            "prediction": int(
                float(result["fusion_score"]) >= float(result["risk_threshold"])
            ),
            "label": result["nlp_risk_level"],
            "error": None,
        }

    except (TypeError, ValueError) as exc:
        return {
            "available": False,
            "nlp_score": None,
            "fusion_score": None,
            "error": str(exc),
        }

    except Exception as exc:
        return {
            "available": False,
            "nlp_score": None,
            "fusion_score": None,
            "error": f"NLP prediction failed: {exc}",
        }