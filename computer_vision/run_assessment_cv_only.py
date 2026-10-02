"""
Standalone CV assessment runner.

This version does not open a webcam.
It analyses an image file supplied by the user.
"""

from __future__ import annotations

from pathlib import Path

from computer_vision.cv_pipeline import run_cv_pipeline


def run_assessment(image_path: str | Path) -> None:
    """
    Run one Computer Vision assessment using an existing image.

    Parameters
    ----------
    image_path:
        Path to a JPG, JPEG, or PNG face image.
    """

    print("\n" + "=" * 70)
    print("COMPUTER VISION ASSESSMENT")
    print("=" * 70)

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image file not found: {path}"
        )

    print(f"\nAnalysing image: {path}")

    cv_result = run_cv_pipeline(
        image_path=path,
        display_result=False,
    )

    print("\n" + "-" * 70)
    print("CV RESULT")
    print("-" * 70)

    print(
        f"CV available: "
        f"{cv_result.get('cv_available')}"
    )

    print(
        f"Visual negative-affect score: "
        f"{cv_result.get('cv_score')}"
    )

    print(
        f"Dominant emotion: "
        f"{cv_result.get('dominant_emotion')}"
    )

    print(
        f"Frames analysed: "
        f"{cv_result.get('frames_analysed')}"
    )

    print(
        f"Valid-frame ratio: "
        f"{cv_result.get('valid_frame_ratio')}"
    )

    print("\nAssessment completed.")


if __name__ == "__main__":
    image_file = input(
        "Enter the full path of a face image: "
    ).strip()

    run_assessment(image_file)