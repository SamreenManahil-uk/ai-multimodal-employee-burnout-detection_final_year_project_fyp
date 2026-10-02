"""
Computer Vision Result-Saving Module.

This module optionally saves Computer Vision assessment results as JSON.

Important:
    The saved JSON file is not required for real-time fusion.

    During runtime, the ML, NLP, and CV outputs should be passed directly
    in memory to the fusion module. Saving is only useful for:

    - experiment records
    - debugging
    - dissertation evidence
    - evaluation
    - optional audit logs
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from computer_vision.config import CV_OUTPUT_DIR


def make_json_serialisable(value: Any) -> Any:
    """
    Convert values into JSON-compatible Python types.

    This handles dictionaries, lists, tuples, Path objects, NumPy-style
    scalar values, and ordinary Python values.

    Parameters
    ----------
    value:
        Any Python value.

    Returns
    -------
    Any
        JSON-compatible representation of the supplied value.
    """

    if isinstance(value, Mapping):
        return {
            str(key): make_json_serialisable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_serialisable(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    # Handle NumPy scalar-style objects without requiring NumPy import.
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            type(None),
        ),
    ):
        return value

    return str(value)


def build_cv_result_record(
    cv_result: Mapping[str, Any],
    emotion_analysis: Optional[Mapping[str, Any]] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a complete Computer Vision result record.

    Parameters
    ----------
    cv_result:
        Fusion-ready result returned by build_cv_fusion_output().

    emotion_analysis:
        Optional full output returned by analyse_frame_sequence().

    session_id:
        Optional application session identifier. This can later be used
        to associate ML, NLP, CV, and fusion outputs from the same runtime
        assessment.

    Returns
    -------
    Dict[str, Any]
        JSON-ready Computer Vision result record.

    Raises
    ------
    TypeError
        If cv_result is not a mapping.
    """

    if not isinstance(cv_result, Mapping):
        raise TypeError(
            "cv_result must be a dictionary or mapping."
        )

    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    cv_available = bool(
        cv_result.get("cv_available", False)
    )

    record: Dict[str, Any] = {
        "timestamp": timestamp,
        "session_id": session_id,
        "module": "computer_vision",
        "status": (
            "successful"
            if cv_available
            else "unavailable"
        ),
        "runtime_usage": (
            "This file is an optional record. The real-time fusion "
            "module should receive the CV result directly in memory."
        ),
        "cv_result": {
            "cv_available": cv_available,
            "cv_score": cv_result.get("cv_score"),
            "cv_score_percentage": cv_result.get(
                "cv_score_percentage"
            ),
            "negative_emotion_mass": cv_result.get(
                "negative_emotion_mass"
            ),
            "dominant_emotion": cv_result.get(
                "dominant_emotion"
            ),
            "dominant_confidence": cv_result.get(
                "dominant_confidence"
            ),
            "valid_frame_ratio": cv_result.get(
                "valid_frame_ratio"
            ),
            "frames_requested": cv_result.get(
                "frames_requested"
            ),
            "frames_analysed": cv_result.get(
                "frames_analysed"
            ),
            "frames_rejected": cv_result.get(
                "frames_rejected"
            ),
            "model_name": cv_result.get(
                "model_name"
            ),
            "detector_backend": cv_result.get(
                "detector_backend"
            ),
            "reason": cv_result.get("reason"),
        },
        "score_details": cv_result.get(
            "score_details"
        ),
        "disclaimer": cv_result.get(
            "disclaimer",
            (
                "This score represents visual facial-expression "
                "evidence only. It does not independently diagnose "
                "burnout, stress, depression, or anxiety."
            ),
        ),
    }

    if emotion_analysis is not None:
        if not isinstance(emotion_analysis, Mapping):
            raise TypeError(
                "emotion_analysis must be a dictionary or mapping."
            )

        record["emotion_analysis"] = dict(
            emotion_analysis
        )

    return make_json_serialisable(record)


def generate_result_filename(
    prefix: str = "cv_result",
    session_id: Optional[str] = None,
) -> str:
    """
    Generate a unique JSON filename.

    Parameters
    ----------
    prefix:
        Beginning of the filename.

    session_id:
        Optional session identifier.

    Returns
    -------
    str
        Generated filename.
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    safe_prefix = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in prefix
    )

    if session_id:
        safe_session_id = "".join(
            character
            if character.isalnum() or character in {"-", "_"}
            else "_"
            for character in str(session_id)
        )

        return (
            f"{safe_prefix}_{safe_session_id}_{timestamp}.json"
        )

    return f"{safe_prefix}_{timestamp}.json"


def save_cv_result(
    cv_result: Mapping[str, Any],
    emotion_analysis: Optional[Mapping[str, Any]] = None,
    session_id: Optional[str] = None,
    output_directory: Optional[Path | str] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Save a Computer Vision result as a JSON file.

    Parameters
    ----------
    cv_result:
        Fusion-ready result returned by build_cv_fusion_output().

    emotion_analysis:
        Optional full multi-frame emotion-analysis output.

    session_id:
        Optional runtime session identifier.

    output_directory:
        Optional custom output directory. If omitted, CV_OUTPUT_DIR from
        config.py is used.

    filename:
        Optional filename. If omitted, a timestamped filename is created.

    Returns
    -------
    Path
        Path of the saved JSON file.
    """

    destination_directory = Path(
        output_directory
        if output_directory is not None
        else CV_OUTPUT_DIR
    )

    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_record = build_cv_result_record(
        cv_result=cv_result,
        emotion_analysis=emotion_analysis,
        session_id=session_id,
    )

    if filename is None:
        filename = generate_result_filename(
            prefix="cv_result",
            session_id=session_id,
        )

    if not filename.lower().endswith(".json"):
        filename = f"{filename}.json"

    output_path = destination_directory / filename

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            result_record,
            json_file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


def load_cv_result(
    file_path: Path | str,
) -> Dict[str, Any]:
    """
    Load a saved Computer Vision JSON result.

    This function is useful for debugging and evaluation. It should not
    be required by the real-time fusion workflow.

    Parameters
    ----------
    file_path:
        Path to a previously saved JSON result.

    Returns
    -------
    Dict[str, Any]
        Loaded JSON data.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"CV result file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"The provided path is not a file: {path}"
        )

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as json_file:
        loaded_data = json.load(json_file)

    if not isinstance(loaded_data, dict):
        raise ValueError(
            "The CV result JSON does not contain a dictionary."
        )

    return loaded_data


def test_save_results() -> None:
    """
    Test JSON result saving without opening the camera.
    """

    sample_cv_result = {
        "cv_available": True,
        "cv_score": 0.2003,
        "cv_score_percentage": 20.03,
        "negative_emotion_mass": 0.4571,
        "dominant_emotion": "neutral",
        "dominant_confidence": 0.5427,
        "valid_frame_ratio": 1.0,
        "frames_requested": 15,
        "frames_analysed": 15,
        "frames_rejected": 0,
        "model_name": (
            "DeepFace pretrained facial-expression model"
        ),
        "detector_backend": "opencv",
        "score_details": {
            "visual_negative_affect_score": 0.2003,
            "visual_negative_affect_percentage": 20.03,
            "weighted_negative_component": 0.2004,
            "happiness_reduction": 0.0,
            "negative_emotion_mass": 0.4571,
            "dominant_emotion": "neutral",
            "dominant_confidence": 0.5427,
        },
        "disclaimer": (
            "This score represents visual facial-expression evidence "
            "only. It does not independently diagnose burnout, stress, "
            "depression, or anxiety."
        ),
    }

    sample_emotion_analysis = {
        "status": "Successful",
        "frames_requested": 15,
        "frames_analysed": 15,
        "frames_rejected": 0,
        "valid_frame_ratio": 1.0,
        "averaged_emotions": {
            "angry": 0.0251,
            "disgust": 0.0,
            "fear": 0.1011,
            "happy": 0.0002,
            "sad": 0.3309,
            "surprise": 0.0,
            "neutral": 0.5427,
        },
    }

    saved_path = save_cv_result(
        cv_result=sample_cv_result,
        emotion_analysis=sample_emotion_analysis,
        session_id="test_session",
    )

    print("\n" + "=" * 60)
    print("CV RESULT SAVING TEST")
    print("=" * 60)
    print("Status: Successful")
    print(f"Saved file: {saved_path}")

    loaded_result = load_cv_result(
        saved_path
    )

    print(
        f"Loaded CV score: "
        f"{loaded_result['cv_result']['cv_score']}"
    )
    print(
        "\nImportant: This saved file is optional. "
        "Real-time fusion should use the CV dictionary directly."
    )


if __name__ == "__main__":
    test_save_results()