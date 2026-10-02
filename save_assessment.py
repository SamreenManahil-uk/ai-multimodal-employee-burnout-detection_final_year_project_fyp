from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import uuid


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, Path):
        return str(value)

    return value


def save_complete_assessment(
    employee_data: dict[str, Any],
    feedback_text: str,
    ml_result: dict[str, Any],
    nlp_result: dict[str, Any],
    cv_result: dict[str, Any],
    fusion_result: dict[str, Any],
    project_root: str | Path,
) -> Path:
    """
    Save one complete multimodal assessment as JSON.

    The face image itself is not stored by this function.
    """

    root = Path(project_root)
    output_dir = root / "outputs" / "assessments"
    output_dir.mkdir(parents=True, exist_ok=True)

    assessment_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc)

    record = {
        "assessment_id": assessment_id,
        "created_at_utc": timestamp.isoformat(),
        "employee_data": employee_data,
        "feedback_text": feedback_text,
        "ml_result": ml_result,
        "nlp_result": nlp_result,
        "cv_result": {
            key: value
            for key, value in cv_result.items()
            if key != "temporary_image_path"
        },
        "fusion_result": fusion_result,
    }

    output_path = (
        output_dir
        / f"assessment_{timestamp:%Y%m%d_%H%M%S}_{assessment_id[:8]}.json"
    )

    output_path.write_text(
        json.dumps(
            _json_safe(record),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_path