"""
Utility functions for the NLP module.

Additional text processing, result saving, and multimodal fusion
functions can be added here later.
"""


def normalise_score(score: float) -> float:
    """
    Ensure a score remains between 0 and 1.
    """
    return max(0.0, min(1.0, float(score)))