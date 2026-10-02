"""
NLP Result Saving Module

This module saves DistilBERT employee-feedback analysis results
to a CSV file for documentation, evaluation, dashboard integration,
and multimodal feature fusion.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any


# =========================================================
# OUTPUT CONFIGURATION
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "outputs",
)

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "nlp_feedback_results.csv",
)


# =========================================================
# CSV COLUMN CONFIGURATION
# =========================================================

FIELDNAMES = [
    "timestamp",
    "source",
    "model_name",
    "model_method",
    "employee_feedback",
    "dominant_category",
    "emotional_exhaustion_score",
    "workplace_stress_score",
    "job_dissatisfaction_score",
    "positive_wellbeing_score",
    "neutral_experience_score",
    "nlp_burnout_signal",
    "fusion_score",
    "risk_threshold",
    "nlp_risk_level",
    "interpretation",
]


# =========================================================
# RESULT VALIDATION
# =========================================================

def validate_result(
    result: dict[str, Any],
) -> None:
    """
    Validate that the NLP result contains all required fields.

    Args:
        result:
            Dictionary returned by analyse_employee_feedback().

    Raises:
        TypeError:
            If result is not a dictionary.

        ValueError:
            If one or more required fields are missing.
    """

    if not isinstance(result, dict):
        raise TypeError(
            "The NLP result must be provided as a dictionary."
        )

    required_fields = {
        "model_name",
        "model_method",
        "feedback",
        "category_scores",
        "dominant_category",
        "nlp_burnout_signal",
        "fusion_score",
        "risk_threshold",
        "nlp_risk_level",
        "interpretation",
    }

    missing_fields = (
        required_fields - result.keys()
    )

    if missing_fields:
        raise ValueError(
            "Cannot save NLP result. "
            f"Missing fields: {sorted(missing_fields)}"
        )

    if not isinstance(
        result["category_scores"],
        dict,
    ):
        raise TypeError(
            "The category_scores field must be a dictionary."
        )


# =========================================================
# ROW PREPARATION
# =========================================================

def prepare_csv_row(
    result: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    """
    Convert the NLP result into one CSV row.

    Args:
        result:
            Validated NLP result dictionary.

        source:
            Origin of the feedback, such as:
            - Interactive Input
            - Automated Test
            - Streamlit Dashboard
            - Multimodal Fusion

    Returns:
        A dictionary matching the CSV column structure.
    """

    category_scores = result[
        "category_scores"
    ]

    return {
        "timestamp": (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ),

        "source": source,

        "model_name": result[
            "model_name"
        ],

        "model_method": result[
            "model_method"
        ],

        "employee_feedback": result[
            "feedback"
        ],

        "dominant_category": result[
            "dominant_category"
        ],

        "emotional_exhaustion_score": (
            category_scores.get(
                "emotional exhaustion and burnout",
                0.0,
            )
        ),

        "workplace_stress_score": (
            category_scores.get(
                "workplace stress and overload",
                0.0,
            )
        ),

        "job_dissatisfaction_score": (
            category_scores.get(
                "job dissatisfaction",
                0.0,
            )
        ),

        "positive_wellbeing_score": (
            category_scores.get(
                "positive workplace wellbeing",
                0.0,
            )
        ),

        "neutral_experience_score": (
            category_scores.get(
                "neutral workplace experience",
                0.0,
            )
        ),

        "nlp_burnout_signal": result[
            "nlp_burnout_signal"
        ],

        "fusion_score": result[
            "fusion_score"
        ],

        "risk_threshold": result[
            "risk_threshold"
        ],

        "nlp_risk_level": result[
            "nlp_risk_level"
        ],

        "interpretation": result[
            "interpretation"
        ],
    }


# =========================================================
# CSV FILE HANDLING
# =========================================================

def should_write_header(
    file_path: str,
) -> bool:
    """
    Determine whether the CSV header must be written.

    The header is written if the file does not exist
    or if the existing file is empty.
    """

    if not os.path.exists(file_path):
        return True

    return os.path.getsize(file_path) == 0


def save_nlp_result(
    result: dict[str, Any],
    source: str = "Unknown",
) -> None:
    """
    Save one NLP prediction result to the CSV output file.

    Args:
        result:
            Dictionary returned by
            analyse_employee_feedback().

        source:
            Describes where the feedback came from.

    Example:
        save_nlp_result(
            result=result,
            source="Interactive Input",
        )
    """

    validate_result(result)

    cleaned_source = str(source).strip()

    if not cleaned_source:
        cleaned_source = "Unknown"

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True,
    )

    row = prepare_csv_row(
        result=result,
        source=cleaned_source,
    )

    write_header = should_write_header(
        OUTPUT_FILE
    )

    try:
        with open(
            OUTPUT_FILE,
            mode="a",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=FIELDNAMES,
                extrasaction="ignore",
            )

            if write_header:
                writer.writeheader()

            writer.writerow(row)

    except PermissionError as error:
        raise PermissionError(
            "The NLP results file could not be updated. "
            "Close the CSV file if it is currently open in Excel "
            "and run the program again."
        ) from error

    except OSError as error:
        raise OSError(
            "The NLP result could not be saved "
            f"to {OUTPUT_FILE}: {error}"
        ) from error

    print("\nNLP result saved successfully.")
    print(f"Location: {OUTPUT_FILE}")


# =========================================================
# OPTIONAL UTILITY
# =========================================================

def get_output_file_path() -> str:
    """
    Return the absolute path of the NLP results CSV file.

    This can later be used by the Streamlit dashboard
    or the multimodal fusion module.
    """

    return OUTPUT_FILE