from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EmotionPrediction:
    """Standard output returned by an expression recognizer."""

    label: str
    class_index: Optional[int]
    confidence: Optional[float]
    success: bool
    error: Optional[str] = None
