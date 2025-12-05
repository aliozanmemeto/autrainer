# Enrollment personalization module for autrainer
"""Enrollment-based personalization for speech emotion recognition.

This module implements the enrollment personalization approach for adapting
emotion recognition models to individual speakers using reference utterances.
"""

from .enrollment_aibo import (
    EnrollmentAIBO,
    EnrollmentDataBatch,
    EnrollmentDataItem,
)
from .wav2vec2_enrollment import Wav2Vec2Enrollment
from .visualization import (
    EnrollmentVisualizer,
    extract_attention_from_batch,
    create_attention_report,
    compute_uar,
)

__all__ = [
    "EnrollmentAIBO",
    "EnrollmentDataItem",
    "EnrollmentDataBatch",
    "Wav2Vec2Enrollment",
    "EnrollmentVisualizer",
    "extract_attention_from_batch",
    "create_attention_report",
    "compute_uar",
]
