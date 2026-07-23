import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional, Tuple

import cv2

try:
    from emotion_effects import (
        EMOTION_COLORS,
        EmotionEffects,
    )
    from emotion_recognizer import (
        LBFEmotionRecognizer,
    )
    from emotion_stickers import (
        EmotionStickerRenderer,
    )
except ImportError:
    from expert.emotion_effects import (
        EMOTION_COLORS,
        EmotionEffects,
    )
    from expert.emotion_recognizer import (
        LBFEmotionRecognizer,
    )
    from expert.emotion_stickers import (
        EmotionStickerRenderer,
    )


FaceBox = Tuple[int, int, int, int]


BASE = Path(__file__).resolve().parent

CASCADE_PATH = str(
    BASE.parent
    / "beginner"
    / "haarcascade_frontalface_default.xml"
)

SMOOTH_WINDOW = 3
MIN_VOTES = 2
RESET_AFTER_MISSING = 8

PREVIEW_KEYS = {
    ord("1"): "angry",
    ord("2"): "disgust",
    ord("3"): "fear",
    ord("4"): "happy",
    ord("5"): "neutral",
    ord("6"): "sad",
    ord("7"): "surprise",
}


cv2.setUseOptimized(True)

detector = cv2.CascadeClassifier(
    CASCADE_PATH
)

if detector.empty():
    raise RuntimeError(
        f"Could not load Haar cascade: {CASCADE_PATH}"
    )

recognizer = LBFEmotionRecognizer()

effect_engine = EmotionEffects(
    box_smoothing=0.70,
    transition_frames=2,
)

sticker_renderer = EmotionStickerRenderer()


def select_primary_face(
    faces,
) -> Optional[FaceBox]:
    """
    Remove nested duplicate Haar detections and select
    the largest remaining face.
    """
    if faces is None or len(faces) == 0:
        return None

    normalized_faces = [
        tuple(map(int, face))
        for face in faces
    ]

    filtered_faces = []

    for i, (x1, y1, w1, h1) in enumerate(
        normalized_faces
    ):
        contains_another_face = False

        for j, (x2, y2, w2, h2) in enumerate(
            normalized_faces
        ):
            if i == j:
                continue

            contains = (
                x1 <= x2
                and y1 <= y2
                and x1 + w1 >= x2 + w2
                and y1 + h1 >= y2 + h2
            )

            if contains:
                contains_another_face = True
                break

        if not contains_another_face:
            filtered_faces.append(
                (x1, y1, w1, h1)
            )

    candidates = (
        filtered_faces
        if filtered_faces
        else normalized_faces
    )

    return max(
        candidates,
        key=lambda face: face[2] * face[3],
    )


def clip_face_box(
    face_box: FaceBox,
    frame_width: int,
    frame_height: int,
) -> Optional[FaceBox]:
    """Keep a face box inside the current frame."""
    x, y, w, h = face_box

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(
        frame_width,
        x + w,
    )

    y2 = min(
        frame_height,
        y + h,
    )

    if x1 >= x2 or y1 >= y2:
        return None

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    )


def draw_plain_face_box(
    frame,
    label: str,
    face_box: FaceBox,
):
    """Draw a plain bounding box when effects are disabled."""
    x, y, w, h = face_box

    color = EMOTION_COLORS.get(
        label,
        EMOTION_COLORS["?"],
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        2,
        cv2.LINE_AA,
    )


def draw_face_label(
    frame,
    label: str,
    face_box: FaceBox,
):
    """Draw a text label above the face."""
    x, y, _, _ = face_box

    color = EMOTION_COLORS.get(
        label,
        EMOTION_COLORS["?"],
    )

    text_y = max(
        y - 12,
        32,
    )

    cv2.putText(
        frame,
        label,
        (x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (20, 20, 20),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        label,
        (x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        color,
        2,
        cv2.LINE_AA,
    )


def format_confidence(
    confidence: Optional[float],
) -> str:
    """Format classifier confidence."""
    if confidence is None:
        return "N/A"

    confidence = max(
        0.0,
        min(1.0, float(confidence)),
    )

    return f"{confidence * 100:.1f}%"


def fit_text_scale(
    text: str,
    max_width: int,
    initial_scale: float,
    thickness: int = 1,
) -> float:
    """
    Reduce text size when the status line is wider
    than the current video frame.
    """
    scale = initial_scale

    while scale > 0.35:
        text_width, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )[0]

        if text_width <= max_width:
            break

        scale -= 0.03

    return max(
        0.35,
        scale,
    )


def draw_status_bar(
    frame,
    detected_label: str,
    display_label: str,
    confidence: Optional[float],
    preview_label: Optional[str],
    average_ms: float,
    effects_enabled: bool,
):
    """
    Draw a compact, semi-transparent status bar
    along the bottom of the frame.
    """
    frame_height, frame_width = frame.shape[:2]

    bar_height = max(
        42,
        int(frame_height * 0.065),
    )

    bar_y1 = max(
        0,
        frame_height - bar_height,
    )

    bar = frame[
        bar_y1:frame_height,
        0:frame_width,
    ]

    if bar.size == 0:
        return

    overlay = bar.copy()
    overlay[:] = (18, 18, 22)

    cv2.addWeighted(
        overlay,
        0.52,
        bar,
        0.48,
        0,
        dst=bar,
    )

    mode_text = (
        "AUTO"
        if preview_label is None
        else f"PREVIEW {preview_label.upper()}"
    )

    effects_text = (
        "FX ON"
        if effects_enabled
        else "FX OFF"
    )

    confidence_text = format_confidence(
        confidence
    )

    status_text = (
        f"Detected: {detected_label}"
        f"   |   Display: {display_label}"
        f"   |   {mode_text}"
        f"   |   Confidence: {confidence_text}"
        f"   |   Avg: {average_ms:.1f} ms"
        f"   |   {effects_text}"
    )

    text_scale = fit_text_scale(
        text=status_text,
        max_width=max(
            100,
            frame_width - 28,
        ),
        initial_scale=0.52,
        thickness=1,
    )

    text_size = cv2.getTextSize(
        status_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        1,
    )[0]

    text_y = (
        bar_y1
        + (bar_height + text_size[1]) // 2
        - 2
    )

    display_color = EMOTION_COLORS.get(
        display_label,
        EMOTION_COLORS["?"],
    )

    # Soft dark outline
    cv2.putText(
        frame,
        status_text,
        (14, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (10, 10, 12),
        3,
        cv2.LINE_AA,
    )

    # Main status text
    cv2.putText(
        frame,
        status_text,
        (14, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (232, 232, 235),
        1,
        cv2.LINE_AA,
    )

    # Small emotion-colored marker
    marker_x = max(
        5,
        frame_width - 14,
    )

    cv2.circle(
        frame,
        (marker_x, bar_y1 + bar_height // 2),
        5,
        display_color,
        -1,
        cv2.LINE_AA,
    )


def open_camera():
    """Open the webcam using a low-buffer Windows backend."""
    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW,
    )

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if camera.isOpened():
        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

    return camera


def main():
    cap = open_camera()

    if not cap.isOpened():
        print(
            "Error: Could not open the camera."
        )
        return

    cv2.namedWindow(
        "Emotion Recognition",
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        "Emotion Recognition",
        800,
        600,
    )

    label_history = deque(
        maxlen=SMOOTH_WINDOW
    )

    timing_history = deque(
        maxlen=30
    )

    stable_label = "?"
    raw_label = "?"

    current_confidence: Optional[float] = None

    missing_face_frames = 0
    frame_index = 0

    effects_enabled = True
    preview_label: Optional[str] = None

    print(
        "Emotion recognition demo started."
    )

    print("Controls:")
    print("  Q - Quit")
    print("  E - Enable or disable effects")
    print("  1 - Preview Angry")
    print("  2 - Preview Disgust")
    print("  3 - Preview Fear")
    print("  4 - Preview Happy")
    print("  5 - Preview Neutral")
    print("  6 - Preview Sad")
    print("  7 - Preview Surprise")
    print("  0 - Return to automatic recognition")

    try:
        while True:
            start_time = time.perf_counter()

            ret, frame = cap.read()

            if not ret:
                print(
                    "Error: Could not read a frame "
                    "from the camera."
                )
                break

            frame_index += 1

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            frame_height, frame_width = (
                gray.shape[:2]
            )

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )

            primary_face = select_primary_face(
                faces
            )

            displayed_face: Optional[FaceBox] = None

            if primary_face is not None:
                missing_face_frames = 0

                clipped_face = clip_face_box(
                    primary_face,
                    frame_width,
                    frame_height,
                )

                if clipped_face is not None:
                    x, y, w, h = clipped_face

                    face_roi = gray[
                        y:y + h,
                        x:x + w,
                    ]

                    prediction = recognizer.predict(
                        face_roi
                    )

                    if prediction.success:
                        raw_label = prediction.label
                        current_confidence = (
                            prediction.confidence
                        )

                        label_history.append(
                            raw_label
                        )

                        candidate_label, candidate_votes = (
                            Counter(
                                label_history
                            ).most_common(1)[0]
                        )

                        if (
                            candidate_votes
                            >= MIN_VOTES
                        ):
                            stable_label = (
                                candidate_label
                            )

                    else:
                        raw_label = "?"
                        current_confidence = None

                    display_label = (
                        preview_label
                        if preview_label is not None
                        else stable_label
                    )

                    if effects_enabled:
                        displayed_face = (
                            effect_engine.render(
                                frame=frame,
                                label=display_label,
                                face_box=clipped_face,
                                frame_index=frame_index,
                                confidence=(
                                    current_confidence
                                ),
                            )
                        )

                        sticker_box = (
                            displayed_face
                            if displayed_face is not None
                            else clipped_face
                        )

                        sticker_renderer.render(
                            frame=frame,
                            label=display_label,
                            face_box=sticker_box,
                            frame_index=frame_index,
                            opacity=1.0,
                        )

                    else:
                        displayed_face = clipped_face

                        draw_plain_face_box(
                            frame=frame,
                            label=display_label,
                            face_box=displayed_face,
                        )

                    if displayed_face is not None:
                        # Sticker badges already display the emotion.
                        should_draw_text_label = not (
                            effects_enabled
                            and display_label != "?"
                        )

                        if should_draw_text_label:
                            draw_face_label(
                                frame=frame,
                                label=display_label,
                                face_box=displayed_face,
                            )

            else:
                missing_face_frames += 1
                raw_label = "?"
                current_confidence = None

                if (
                    missing_face_frames
                    >= RESET_AFTER_MISSING
                ):
                    label_history.clear()
                    stable_label = "?"
                    effect_engine.reset()

            display_label = (
                preview_label
                if preview_label is not None
                else stable_label
            )

            elapsed_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

            timing_history.append(
                elapsed_ms
            )

            average_ms = (
                sum(timing_history)
                / len(timing_history)
            )

            draw_status_bar(
                frame=frame,
                detected_label=stable_label,
                display_label=display_label,
                confidence=current_confidence,
                preview_label=preview_label,
                average_ms=average_ms,
                effects_enabled=effects_enabled,
            )

            cv2.imshow(
                "Emotion Recognition",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                ord("Q"),
            ):
                break

            if key in (
                ord("e"),
                ord("E"),
            ):
                effects_enabled = (
                    not effects_enabled
                )

                if effects_enabled:
                    effect_engine.reset()

                status = (
                    "enabled"
                    if effects_enabled
                    else "disabled"
                )

                print(
                    f"Visual effects {status}."
                )

            if key in PREVIEW_KEYS:
                preview_label = PREVIEW_KEYS[key]
                effect_engine.reset()

                print(
                    f"Preview mode: {preview_label}"
                )

            if key == ord("0"):
                preview_label = None
                effect_engine.reset()

                print(
                    "Preview mode disabled. "
                    "Returned to automatic recognition."
                )

    except KeyboardInterrupt:
        print(
            "\nStopped by keyboard interrupt."
        )

    finally:
        cap.release()
        cv2.destroyAllWindows()

        print(
            "Emotion recognition demo closed."
        )


if __name__ == "__main__":
    main()