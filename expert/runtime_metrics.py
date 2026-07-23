"""Shared runtime-metric definitions for the expression pipeline."""

from __future__ import annotations

from typing import Iterable, Optional


# Expert Task 1 requires one expression prediction to complete within 30 ms.
PREDICTION_TARGET_MS = 30.0


def percentile(
    values: Iterable[float],
    q: float,
) -> Optional[float]:
    """Return a linearly interpolated percentile, or None for no samples."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None

    position = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def prediction_runtime_status(
    average_ms: Optional[float],
    p95_ms: Optional[float],
    target_ms: float = PREDICTION_TARGET_MS,
) -> str:
    """Classify mean/P95 latency without allowing a good mean to hide spikes."""
    if average_ms is None:
        return "WAIT"
    if average_ms > target_ms:
        return "OVER"
    if p95_ms is not None and p95_ms > target_ms:
        return "SPIKES"
    return "PASS"
