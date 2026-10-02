"""Configuration for the Computer Vision image-analysis module."""

from pathlib import Path

CV_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CV_DIR.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CV_OUTPUT_DIR = OUTPUTS_DIR / "computer_vision"
OUTPUTS_DIR.mkdir(exist_ok=True)
CV_OUTPUT_DIR.mkdir(exist_ok=True)

# DeepFace
DEEPFACE_ACTIONS = ["emotion"]
DETECTOR_BACKEND = "opencv"
ENFORCE_DETECTION = True
ALIGN_FACE = True

SUPPORTED_EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

# Research heuristic weights; these are not medical weights.
NEGATIVE_AFFECT_WEIGHTS = {
    "sad": 0.35,
    "fear": 0.25,
    "angry": 0.20,
    "disgust": 0.10,
    "neutral": 0.10,
}

HAPPINESS_REDUCTION_WEIGHT = 0.25

LOW_INDICATOR_THRESHOLD = 0.25
MODERATE_INDICATOR_THRESHOLD = 0.50

SAVE_INDIVIDUAL_CV_RESULTS = False
SAVE_COMPLETE_ASSESSMENTS = True
SAVE_CAPTURED_IMAGES = False
SAVE_JSON_RESULTS = True

PRIVACY_MESSAGE = (
    "Facial images are processed only during runtime and are not "
    "permanently stored. Only numerical outputs may be saved."
)

DISCLAIMER = (
    "This module performs facial-expression analysis using a pretrained "
    "DeepFace model. It does not diagnose burnout, stress, fatigue, "
    "anxiety, or depression."
)
