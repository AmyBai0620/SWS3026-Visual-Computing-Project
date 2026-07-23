from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from typing import Optional

import cv2
import numpy as np

try:
    from rtmpose_emotion_recognizer import (
        RTMPoseEmotionRecognizer,
    )
    from robust_face_detector import RobustFaceDetector
    from runtime_metrics import (
        PREDICTION_TARGET_MS,
        percentile,
        prediction_runtime_status,
    )
except ImportError:
    from expert.rtmpose_emotion_recognizer import (
        RTMPoseEmotionRecognizer,
    )
    from expert.robust_face_detector import RobustFaceDetector
    from expert.runtime_metrics import (
        PREDICTION_TARGET_MS,
        percentile,
        prediction_runtime_status,
    )

WINDOW_NAME = "RTMPose Expression Classifier Test"

SMOOTH_WINDOW = 5
MIN_VOTES = 3
RESET_AFTER_MISSING = 8
TIMING_WINDOW = 60

EMOTION_COLORS = {
    "angry": (70, 70, 235),
    "disgust": (105, 155, 85),
    "fear": (190, 120, 180),
    "happy": (80, 210, 250),
    "neutral": (185, 185, 185),
    "sad": (220, 150, 70),
    "surprise": (80, 190, 245),
    "?": (180, 180, 180),
}


def runtime_text(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{float(value):.1f} ms"


def fps_text(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{float(value):.1f} FPS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Webcam smoke test for the final "
            "RTMPose expression classifier."
        )
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=540,
    )
    return parser.parse_args()



def draw_keypoints(
    frame: np.ndarray,
    keypoints: Optional[np.ndarray],
    scores: Optional[np.ndarray],
    face_box: tuple[int, int, int, int],
    score_threshold: float = 0.20,
) -> None:
    if keypoints is None or scores is None:
        return

    x, y, _, _ = face_box

    for point, score in zip(
        keypoints,
        scores,
    ):
        if float(score) < score_threshold:
            continue

        px = int(round(float(point[0]))) + x
        py = int(round(float(point[1]))) + y

        cv2.circle(
            frame,
            (px, py),
            2,
            (60, 230, 120),
            -1,
            cv2.LINE_AA,
        )


def draw_status_panel(
    frame: np.ndarray,
    stable_label: str,
    raw_label: str,
    confidence: Optional[float],
    average_prediction_ms: Optional[float],
    prediction_p95_ms: Optional[float],
    display_fps: Optional[float],
    pose_ms: Optional[float],
    classifier_ms: Optional[float],
    landmarks_visible: bool,
    detector_source: str,
    detector_is_fresh: bool,
    error_text: Optional[str],
) -> None:
    height, width = frame.shape[:2]

    panel_width = min(570, width - 20)
    panel_height = 250

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_width, 10 + panel_height),
        (22, 22, 25),
        -1,
        cv2.LINE_AA,
    )

    cv2.addWeighted(
        overlay,
        0.76,
        frame,
        0.24,
        0.0,
        frame,
    )

    color = EMOTION_COLORS.get(
        stable_label,
        EMOTION_COLORS["?"],
    )

    confidence_text = (
        "--"
        if confidence is None
        else f"{confidence * 100.0:.1f}%"
    )

    pose_text = (
        "--"
        if pose_ms is None
        else f"{pose_ms:.1f} ms"
    )

    classifier_text = (
        "--"
        if classifier_ms is None
        else f"{classifier_ms:.2f} ms"
    )

    prediction_status = prediction_runtime_status(
        average_prediction_ms,
        prediction_p95_ms,
    )

    cv2.putText(
        frame,
        f"Stable: {stable_label.upper()}",
        (28, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        color,
        2,
        cv2.LINE_AA,
    )

    lines = [
        f"Raw: {raw_label}   Confidence: {confidence_text}",
        (
            f"Prediction avg: {runtime_text(average_prediction_ms)} "
            f"[{prediction_status}]"
        ),
        (
            f"Prediction P95: {runtime_text(prediction_p95_ms)}   "
            f"Target: <= {PREDICTION_TARGET_MS:.0f} ms"
        ),
        f"RTMPose avg: {pose_text}   Classifier avg: {classifier_text}",
        f"Display rate: {fps_text(display_fps)}",
        (
            f"Detector: {detector_source.upper()} "
            + ("(fresh)" if detector_is_fresh else "(held/missing)")
        ),
        (
            "Landmarks: ON"
            if landmarks_visible
            else "Landmarks: OFF"
        ),
    ]

    y = 78

    for line in lines:
        cv2.putText(
            frame,
            line,
            (28, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (235, 235, 238),
            1,
            cv2.LINE_AA,
        )
        y += 25

    if error_text:
        error_display = error_text[:58]

        cv2.putText(
            frame,
            f"Error: {error_display}",
            (16, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (70, 70, 245),
            1,
            cv2.LINE_AA,
        )


def open_camera(
    camera_index: int,
) -> cv2.VideoCapture:
    if hasattr(cv2, "CAP_DSHOW"):
        capture = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW,
        )

        if capture.isOpened():
            return capture

        capture.release()

    return cv2.VideoCapture(camera_index)


def main() -> int:
    args = parse_args()

    print("Loading robust MediaPipe/Haar face detector...")
    detector = RobustFaceDetector(
        smoothing_alpha=0.42,
        hold_frames=6,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    print("Loading final RTMPose expression recognizer...")
    recognizer = RTMPoseEmotionRecognizer(
        device=args.device,
        warmup_runs=10,
    )

    print(
        f"Ready. Device: {recognizer.device}. "
        "Opening camera..."
    )

    capture = open_camera(args.camera)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}."
        )

    capture.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )
    capture.set(
        cv2.CAP_PROP_FPS,
        30.0,
    )
    capture.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        args.width,
    )
    capture.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        args.height,
    )
    capture.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    label_history = deque(
        maxlen=SMOOTH_WINDOW
    )
    prediction_timing_history = deque(
        maxlen=TIMING_WINDOW
    )
    pose_timing_history = deque(
        maxlen=TIMING_WINDOW
    )
    classifier_timing_history = deque(
        maxlen=TIMING_WINDOW
    )
    display_interval_history = deque(
        maxlen=TIMING_WINDOW
    )
    last_presented_at: Optional[float] = None

    stable_label = "?"
    raw_label = "?"
    confidence: Optional[float] = None
    missing_frames = 0
    landmarks_visible = True
    error_text: Optional[str] = None
    detector_source = "none"
    detector_is_fresh = False

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera stream: {actual_width}x{actual_height} "
        f"@ {actual_fps:.1f} FPS requested/negotiated."
    )
    print("Controls: Q quit, D toggle landmarks")

    try:
        while True:
            ok, frame = capture.read()

            if not ok or frame is None:
                print("Could not read a camera frame.")
                break

            detection = detector.detect(frame)
            clipped_face = detection.box
            detector_source = detection.source
            detector_is_fresh = detection.is_fresh

            if clipped_face is not None:
                missing_frames = detection.missed_frames
                x, y, width, height = clipped_face
                prediction_succeeded = False

                # Match the final scrapbook app: update RTMPose only when the
                # detector produced a fresh box.  A short held box keeps the
                # stable label visible but does not pretend old landmarks are
                # current.
                if detection.is_fresh:
                    face_roi = frame[
                        y:y + height,
                        x:x + width,
                    ]

                    prediction = recognizer.predict(face_roi)

                    if prediction.success:
                        prediction_succeeded = True
                        if recognizer.last_prediction_ms is not None:
                            prediction_timing_history.append(
                                recognizer.last_prediction_ms
                            )

                        if recognizer.last_pose_ms is not None:
                            pose_timing_history.append(
                                recognizer.last_pose_ms
                            )

                        if recognizer.last_classifier_ms is not None:
                            classifier_timing_history.append(
                                recognizer.last_classifier_ms
                            )

                        error_text = None
                        raw_label = prediction.label
                        confidence = prediction.confidence
                        label_history.append(raw_label)

                        candidate, votes = Counter(
                            label_history
                        ).most_common(1)[0]

                        if votes >= MIN_VOTES:
                            stable_label = candidate
                    else:
                        raw_label = "?"
                        confidence = None
                        error_text = prediction.error
                else:
                    error_text = None

                box_color = EMOTION_COLORS.get(
                    stable_label,
                    EMOTION_COLORS["?"],
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    box_color,
                    2,
                    cv2.LINE_AA,
                )

                if landmarks_visible and prediction_succeeded:
                    draw_keypoints(
                        frame,
                        recognizer.last_keypoints,
                        recognizer.last_scores,
                        clipped_face,
                    )

                source_suffix = (
                    detector_source.upper()
                    if detection.is_fresh
                    else "HOLD"
                )
                label_text = f"{stable_label.upper()} [{source_suffix}]"

                cv2.putText(
                    frame,
                    label_text,
                    (x, max(26, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.66,
                    box_color,
                    2,
                    cv2.LINE_AA,
                )

            else:
                missing_frames = detection.missed_frames
                raw_label = "?"
                confidence = None
                error_text = None

                if missing_frames >= RESET_AFTER_MISSING:
                    label_history.clear()
                    stable_label = "?"

            average_prediction_ms = (
                sum(prediction_timing_history)
                / len(prediction_timing_history)
                if prediction_timing_history
                else None
            )
            prediction_p95_ms = percentile(
                prediction_timing_history,
                0.95,
            )
            average_pose_ms = (
                sum(pose_timing_history)
                / len(pose_timing_history)
                if pose_timing_history
                else None
            )
            average_classifier_ms = (
                sum(classifier_timing_history)
                / len(classifier_timing_history)
                if classifier_timing_history
                else None
            )
            average_display_interval_ms = (
                sum(display_interval_history)
                / len(display_interval_history)
                if display_interval_history
                else 0.0
            )
            display_fps = (
                1000.0 / average_display_interval_ms
                if average_display_interval_ms > 0.0
                else None
            )

            draw_status_panel(
                frame=frame,
                stable_label=stable_label,
                raw_label=raw_label,
                confidence=confidence,
                average_prediction_ms=average_prediction_ms,
                prediction_p95_ms=prediction_p95_ms,
                display_fps=display_fps,
                pose_ms=average_pose_ms,
                classifier_ms=average_classifier_ms,
                landmarks_visible=landmarks_visible,
                detector_source=detector_source,
                detector_is_fresh=detector_is_fresh,
                error_text=error_text,
            )

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            presented_at = time.perf_counter()
            if last_presented_at is not None:
                display_interval_history.append(
                    (
                        presented_at
                        - last_presented_at
                    ) * 1000.0
                )
            last_presented_at = presented_at

            key = cv2.waitKey(1) & 0xFF

            if key in {
                ord("q"),
                ord("Q"),
                27,
            }:
                break

            if key in {
                ord("d"),
                ord("D"),
            }:
                landmarks_visible = (
                    not landmarks_visible
                )

    finally:
        capture.release()
        detector.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
