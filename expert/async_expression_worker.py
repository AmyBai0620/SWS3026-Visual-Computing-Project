from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

class ExpressionRecognizerProtocol(Protocol):
    last_prediction_ms: Optional[float]
    last_pose_ms: Optional[float]
    last_classifier_ms: Optional[float]
    last_input_prepare_ms: Optional[float]

    def predict(self, face_roi: np.ndarray) -> Any:
        ...


@dataclass(frozen=True)
class AsyncPredictionResult:
    """One completed prediction published by the background worker."""

    sequence: int
    prediction: Optional[Any]
    prediction_ms: Optional[float]
    pose_ms: Optional[float]
    classifier_ms: Optional[float]
    input_prepare_ms: Optional[float]
    submitted_at: float
    started_at: float
    finished_at: float
    error: Optional[str] = None

    @property
    def queue_wait_ms(self) -> float:
        return max(0.0, (self.started_at - self.submitted_at) * 1000.0)

    @property
    def end_to_end_ms(self) -> float:
        return max(0.0, (self.finished_at - self.submitted_at) * 1000.0)


class AsyncExpressionWorker:
    """
    Run exactly one RTMPose recognizer on one dedicated thread.

    Input uses a single latest-frame slot rather than an unbounded queue. If
    the worker is busy, a newer face crop replaces the older waiting crop, so
    predictions never build up behind the live camera.
    """

    def __init__(
        self,
        recognizer: ExpressionRecognizerProtocol,
        thread_name: str = "rtmpose-expression-worker",
    ) -> None:
        self.recognizer = recognizer
        self.thread_name = str(thread_name)

        self._condition = threading.Condition()
        self._pending_face: Optional[np.ndarray] = None
        self._pending_sequence = 0
        self._pending_submitted_at = 0.0
        self._last_submitted_sequence = 0

        self._latest_result: Optional[AsyncPredictionResult] = None
        self._stop_requested = False
        self._started = False
        self._thread: Optional[threading.Thread] = None

    @property
    def last_submitted_sequence(self) -> int:
        with self._condition:
            return self._last_submitted_sequence

    @property
    def is_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name=self.thread_name,
                daemon=True,
            )
            self._thread.start()

    def submit(self, face_roi: np.ndarray) -> int:
        """Replace the waiting crop with the newest crop and return its id."""
        if face_roi is None or face_roi.size == 0:
            return self.last_submitted_sequence

        # The main thread continues drawing on the camera frame immediately
        # after submission, so the worker must own an independent copy.
        owned_face = np.ascontiguousarray(face_roi.copy())
        submitted_at = time.perf_counter()

        with self._condition:
            if self._stop_requested:
                return self._last_submitted_sequence

            self._last_submitted_sequence += 1
            sequence = self._last_submitted_sequence
            self._pending_face = owned_face
            self._pending_sequence = sequence
            self._pending_submitted_at = submitted_at
            self._condition.notify()
            return sequence

    def clear_pending(self) -> int:
        """Drop a crop that has not started inference yet."""
        with self._condition:
            self._pending_face = None
            self._pending_sequence = 0
            self._pending_submitted_at = 0.0
            return self._last_submitted_sequence

    def get_latest_result(
        self,
        after_sequence: int = 0,
    ) -> Optional[AsyncPredictionResult]:
        """Return the newest completed result if it is newer than requested."""
        with self._condition:
            result = self._latest_result
            if result is None or result.sequence <= int(after_sequence):
                return None
            return result

    def close(self, timeout: float = 10.0) -> None:
        """Stop after the current inference and join the worker thread."""
        with self._condition:
            self._stop_requested = True
            self._pending_face = None
            self._condition.notify_all()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending_face is None and not self._stop_requested:
                    self._condition.wait()

                if self._stop_requested:
                    return

                face = self._pending_face
                sequence = self._pending_sequence
                submitted_at = self._pending_submitted_at

                # Clear the one-slot buffer before inference. New submissions
                # made while inference is running will fill this slot again.
                self._pending_face = None
                self._pending_sequence = 0
                self._pending_submitted_at = 0.0

            started_at = time.perf_counter()
            prediction: Optional[Any] = None
            error: Optional[str] = None

            try:
                prediction = self.recognizer.predict(face)
            except Exception as exc:  # Keep the UI alive and expose the error.
                error = f"{type(exc).__name__}: {exc}"

            finished_at = time.perf_counter()
            result = AsyncPredictionResult(
                sequence=sequence,
                prediction=prediction,
                prediction_ms=self.recognizer.last_prediction_ms,
                pose_ms=self.recognizer.last_pose_ms,
                classifier_ms=self.recognizer.last_classifier_ms,
                input_prepare_ms=self.recognizer.last_input_prepare_ms,
                submitted_at=submitted_at,
                started_at=started_at,
                finished_at=finished_at,
                error=error,
            )

            with self._condition:
                self._latest_result = result
