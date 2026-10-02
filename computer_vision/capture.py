"""
Legacy local-webcam capture module.

The Streamlit application must not call this module. To prevent accidental
second-camera windows, local webcam capture is disabled by default.
"""

from __future__ import annotations

from typing import List

import numpy as np


def capture_frame_sequence(
    *,
    allow_local_webcam: bool = False,
) -> List[np.ndarray]:
    """
    Block accidental local webcam capture.

    The Streamlit app already obtains an image through st.camera_input().
    """
    if not allow_local_webcam:
        raise RuntimeError(
            "Local OpenCV webcam capture is disabled. "
            "Use st.camera_input() and pass the captured image to "
            "cv_predict.analyse_face_image()."
        )

    raise RuntimeError(
        "Local webcam mode has been removed from this Streamlit build."
    )


def open_camera():
    raise RuntimeError(
        "Local OpenCV camera access is disabled in the Streamlit build."
    )


def test_camera_sequence() -> None:
    print(
        "Local webcam capture is disabled. "
        "Test image capture through the Streamlit application."
    )


if __name__ == "__main__":
    test_camera_sequence()
