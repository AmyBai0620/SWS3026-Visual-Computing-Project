import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_AVAILABLE = False

try:
    from happy_effect_renderer_v8f import HappyEffectRenderer
    from surprise_effect_renderer_v8f import SurpriseEffectRenderer
    from emotion_effects import (
        EMOTION_COLORS,
        EmotionEffects,
    )
    from rtmpose_emotion_recognizer import (
        RTMPoseEmotionRecognizer,
    )
    from wide_screen_emotion_renderer_v8f import (
        WideScreenEmotionRenderer,
    )
    from robust_face_detector import (
        RobustFaceDetector,
    )
    from async_expression_worker import (
        AsyncExpressionWorker,
    )
except ImportError:
    from expert.happy_effect_renderer_v8f import HappyEffectRenderer
    from expert.surprise_effect_renderer_v8f import SurpriseEffectRenderer
    from expert.emotion_effects import (
        EMOTION_COLORS,
        EmotionEffects,
    )
    from expert.rtmpose_emotion_recognizer import (
        RTMPoseEmotionRecognizer,
    )
    from expert.wide_screen_emotion_renderer_v8f import (
        WideScreenEmotionRenderer,
    )
    from expert.robust_face_detector import (
        RobustFaceDetector,
    )
    from expert.async_expression_worker import (
        AsyncExpressionWorker,
    )


FaceBox = Tuple[int, int, int, int]


BASE = Path(__file__).resolve().parent


SMOOTH_WINDOW = 5
MIN_VOTES = 3
RESET_AFTER_MISSING = 8

# V10 submits every fresh face crop to a single-slot asynchronous worker.
# The worker naturally runs only as fast as RTMPose can process; pending crops
# are replaced by newer ones, so no delayed queue can build up.
PREDICTION_INTERVAL = 1
PREDICTION_TIMING_WINDOW = 30
LIVE_TIMING_WARMUP_SAMPLES = 5
DISPLAY_TIMING_WINDOW = 60

WINDOW_NAME = "RTMPose Emotion Camera v10 - Threaded"

APP_WIDTH = 1280
APP_HEIGHT = 720
VIDEO_WIDTH = 960
SIDEBAR_WIDTH = APP_WIDTH - VIDEO_WIDTH

# Minimal, neutral UI palette in OpenCV BGR order.
PANEL_BG = (242, 244, 246)
TEXT_MAIN = (34, 35, 39)
TEXT_MUTED = (118, 121, 129)
DIVIDER = (214, 216, 221)
DEBUG_BG = (234, 236, 239)

_FONT_CACHE = {}

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

face_detector = RobustFaceDetector(
    smoothing_alpha=0.42,
    hold_frames=6,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45,
)

recognizer = RTMPoseEmotionRecognizer(
    device="auto",
    warmup_runs=10,
)

effect_engine = EmotionEffects(
    box_smoothing=0.70,
    transition_frames=2,
)

wide_renderer = WideScreenEmotionRenderer()
happy_effect_renderer = HappyEffectRenderer()
surprise_effect_renderer = SurpriseEffectRenderer()


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
    """Draw only a minimal face box when effects are disabled."""
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


def format_confidence(
    confidence: Optional[float],
) -> str:
    """Format classifier confidence for the sidebar."""
    if confidence is None:
        return "--"

    confidence = max(
        0.0,
        min(1.0, float(confidence)),
    )

    return f"{confidence * 100:.0f}%"


def clamp_color(
    color,
):
    """Convert a color-like value to a valid OpenCV BGR tuple."""
    return tuple(
        int(max(0, min(255, value)))
        for value in color
    )


def mix_color(
    color_a,
    color_b,
    amount: float,
):
    """Linearly mix two BGR colors."""
    amount = max(0.0, min(1.0, amount))

    return clamp_color(
        (
            color_a[0] * (1.0 - amount)
            + color_b[0] * amount,
            color_a[1] * (1.0 - amount)
            + color_b[1] * amount,
            color_a[2] * (1.0 - amount)
            + color_b[2] * amount,
        )
    )


def bgr_to_rgb(
    color,
):
    """Convert a BGR tuple to RGB for Pillow."""
    return (
        int(color[2]),
        int(color[1]),
        int(color[0]),
    )


def load_ui_font(
    size: int,
    semibold: bool = False,
):
    """
    Load Segoe UI on Windows, with portable fallbacks.
    Font objects are cached because this runs every frame.
    """
    cache_key = (
        int(size),
        bool(semibold),
    )

    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]

    if not PIL_AVAILABLE:
        return None

    windows_dir = Path(
        "C:/Windows/Fonts"
    )

    candidates = (
        [
            windows_dir / "seguisb.ttf",
            windows_dir / "segoeuib.ttf",
            Path(
                "/usr/share/fonts/truetype/dejavu/"
                "DejaVuSans-Bold.ttf"
            ),
        ]
        if semibold
        else [
            windows_dir / "segoeui.ttf",
            Path(
                "/usr/share/fonts/truetype/dejavu/"
                "DejaVuSans.ttf"
            ),
        ]
    )

    font = None

    for candidate in candidates:
        if candidate.exists():
            try:
                font = ImageFont.truetype(
                    str(candidate),
                    size,
                )
                break
            except OSError:
                continue

    if font is None:
        font = ImageFont.load_default()

    _FONT_CACHE[cache_key] = font
    return font


def fit_camera_view(
    frame,
    target_width: int,
    target_height: int,
):
    """
    Resize the camera frame without stretching it.
    The usual 4:3 webcam feed fills 960 x 720 exactly.
    """
    frame_height, frame_width = frame.shape[:2]

    if frame_width <= 0 or frame_height <= 0:
        return np.zeros(
            (
                target_height,
                target_width,
                3,
            ),
            dtype=np.uint8,
        )

    scale = min(
        target_width / frame_width,
        target_height / frame_height,
    )

    resized_width = max(
        1,
        int(round(frame_width * scale)),
    )
    resized_height = max(
        1,
        int(round(frame_height * scale)),
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        frame,
        (
            resized_width,
            resized_height,
        ),
        interpolation=interpolation,
    )

    camera_view = np.full(
        (
            target_height,
            target_width,
            3,
        ),
        (15, 15, 17),
        dtype=np.uint8,
    )

    offset_x = (
        target_width - resized_width
    ) // 2
    offset_y = (
        target_height - resized_height
    ) // 2

    camera_view[
        offset_y:offset_y + resized_height,
        offset_x:offset_x + resized_width,
    ] = resized

    return camera_view


def draw_pillow_text(
    draw,
    position,
    text: str,
    size: int,
    color,
    semibold: bool = False,
    anchor: Optional[str] = None,
):
    """Draw antialiased Segoe UI text through Pillow."""
    font = load_ui_font(
        size,
        semibold=semibold,
    )

    kwargs = {}

    if anchor is not None:
        kwargs["anchor"] = anchor

    draw.text(
        position,
        text,
        font=font,
        fill=bgr_to_rgb(color),
        **kwargs,
    )


def draw_sidebar_pillow(
    panel,
    display_label: str,
    confidence: Optional[float],
    preview_label: Optional[str],
    effects_enabled: bool,
    face_detected: bool,
    detected_label: str,
    raw_label: str,
    average_ms: float,
    display_fps: float,
    debug_enabled: bool,
    accent_color,
):
    """
    Draw a restrained, editorial-style sidebar.

    The camera remains the visual focus. The sidebar has no cards,
    glow, avatar letters, oversized headings, or colored badges.
    """
    panel_rgb = cv2.cvtColor(
        panel,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(
        panel_rgb
    )

    draw = ImageDraw.Draw(
        pil_image
    )

    left = 30
    right = SIDEBAR_WIDTH - 30

    tracking_text = (
        "Tracking"
        if face_detected
        else "Waiting"
    )

    tracking_color = (
        accent_color
        if face_detected
        else TEXT_MUTED
    )

    draw.ellipse(
        (
            right - 76,
            33,
            right - 66,
            43,
        ),
        fill=bgr_to_rgb(
            tracking_color
        ),
    )

    draw_pillow_text(
        draw,
        (
            right,
            39,
        ),
        tracking_text,
        14,
        TEXT_MUTED,
        anchor="rm",
    )

    draw_pillow_text(
        draw,
        (
            left,
            34,
        ),
        "Expression",
        15,
        TEXT_MUTED,
    )

    if (
        not face_detected
        and preview_label is None
    ):
        emotion_text = "No face"
        detail_text = "Move into the frame"
    elif display_label == "?":
        emotion_text = "Waiting"
        detail_text = "Reading expression"
    else:
        emotion_text = display_label.title()

        if preview_label is not None:
            detail_text = "Preview mode"
        else:
            detail_text = (
                f"{format_confidence(confidence)} confidence"
            )

    # A single accent rule is enough to connect the emotion
    # to the effect without tinting the whole interface.
    draw.rounded_rectangle(
        (
            left,
            82,
            left + 4,
            151,
        ),
        radius=2,
        fill=bgr_to_rgb(
            accent_color
        ),
    )

    draw_pillow_text(
        draw,
        (
            left + 18,
            74,
        ),
        emotion_text,
        43,
        TEXT_MAIN,
        semibold=True,
    )

    draw_pillow_text(
        draw,
        (
            left + 20,
            132,
        ),
        detail_text,
        16,
        (
            accent_color
            if face_detected
            or preview_label is not None
            else TEXT_MUTED
        ),
    )

    draw.line(
        (
            left,
            190,
            right,
            190,
        ),
        fill=bgr_to_rgb(
            DIVIDER
        ),
        width=1,
    )

    mode_value = (
        "Automatic"
        if preview_label is None
        else f"Preview: {preview_label.title()}"
    )

    effects_value = (
        "On"
        if effects_enabled
        else "Off"
    )

    rows = (
        (
            "Mode",
            mode_value,
        ),
        (
            "Effects",
            effects_value,
        ),
    )

    row_y = 229

    for label_text, value_text in rows:
        draw_pillow_text(
            draw,
            (
                left,
                row_y,
            ),
            label_text,
            15,
            TEXT_MUTED,
            anchor="lm",
        )

        draw_pillow_text(
            draw,
            (
                right,
                row_y,
            ),
            value_text,
            16,
            TEXT_MAIN,
            semibold=True,
            anchor="rm",
        )

        row_y += 48

    if debug_enabled:
        debug_top = 326

        draw.line(
            (
                left,
                debug_top,
                right,
                debug_top,
            ),
            fill=bgr_to_rgb(
                DIVIDER
            ),
            width=1,
        )

        speed_text = (
            "--"
            if average_ms <= 0.0
            else f"{average_ms:.1f} ms"
        )
        rate_text = (
            "--"
            if display_fps <= 0.0
            else f"{display_fps:.1f} FPS"
        )

        debug_lines = (
            f"Raw       {raw_label}",
            f"Stable    {detected_label}",
            f"Pred avg  {speed_text}",
            f"Display   {rate_text}",
        )

        debug_y = debug_top + 28

        for line in debug_lines:
            draw_pillow_text(
                draw,
                (
                    left,
                    debug_y,
                ),
                line,
                14,
                TEXT_MUTED,
            )

            debug_y += 27

    controls_y = 560

    draw.line(
        (
            left,
            controls_y - 26,
            right,
            controls_y - 26,
        ),
        fill=bgr_to_rgb(
            DIVIDER
        ),
        width=1,
    )

    draw_pillow_text(
        draw,
        (
            left,
            controls_y,
        ),
        "Keyboard",
        14,
        TEXT_MUTED,
    )

    control_lines = (
        (
            "0",
            "Automatic",
        ),
        (
            "1-7",
            "Preview",
        ),
        (
            "E",
            "Effects",
        ),
        (
            "D",
            "Details",
        ),
        (
            "Q",
            "Quit",
        ),
    )

    key_y = controls_y + 34

    for key_text, action_text in control_lines:
        draw_pillow_text(
            draw,
            (
                left,
                key_y,
            ),
            key_text,
            14,
            TEXT_MAIN,
            semibold=True,
        )

        draw_pillow_text(
            draw,
            (
                left + 48,
                key_y,
            ),
            action_text,
            14,
            TEXT_MUTED,
        )

        key_y += 25

    result_rgb = np.asarray(
        pil_image,
        dtype=np.uint8,
    )

    return cv2.cvtColor(
        result_rgb,
        cv2.COLOR_RGB2BGR,
    )


def draw_sidebar_opencv_fallback(
    panel,
    display_label: str,
    confidence: Optional[float],
    preview_label: Optional[str],
    effects_enabled: bool,
    face_detected: bool,
    detected_label: str,
    raw_label: str,
    average_ms: float,
    display_fps: float,
    debug_enabled: bool,
    accent_color,
):
    """Fallback layout used only when Pillow is unavailable."""
    left = 30

    emotion_text = (
        display_label.title()
        if display_label != "?"
        else (
            "No face"
            if not face_detected
            else "Waiting"
        )
    )

    cv2.rectangle(
        panel,
        (left, 82),
        (left + 4, 151),
        accent_color,
        -1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        "Expression",
        (left, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        emotion_text,
        (left + 18, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        TEXT_MAIN,
        2,
        cv2.LINE_AA,
    )

    detail_text = (
        "Preview mode"
        if preview_label is not None
        else (
            "Move into the frame"
            if not face_detected
            else f"{format_confidence(confidence)} confidence"
        )
    )

    cv2.putText(
        panel,
        detail_text,
        (left + 20, 151),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        accent_color,
        1,
        cv2.LINE_AA,
    )

    cv2.line(
        panel,
        (left, 190),
        (SIDEBAR_WIDTH - left, 190),
        DIVIDER,
        1,
        cv2.LINE_AA,
    )

    mode_value = (
        "Automatic"
        if preview_label is None
        else f"Preview: {preview_label.title()}"
    )

    cv2.putText(
        panel,
        f"Mode       {mode_value}",
        (left, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEXT_MAIN,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        f"Effects    {'On' if effects_enabled else 'Off'}",
        (left, 283),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEXT_MAIN,
        1,
        cv2.LINE_AA,
    )

    if debug_enabled:
        speed_text = (
            "--"
            if average_ms <= 0.0
            else f"{average_ms:.1f} ms"
        )
        rate_text = (
            "--"
            if display_fps <= 0.0
            else f"{display_fps:.1f} FPS"
        )

        debug_lines = (
            f"Raw: {raw_label}",
            f"Stable: {detected_label}",
            f"Pred {speed_text} / Display {rate_text}",
        )

        y = 360

        for line in debug_lines:
            cv2.putText(
                panel,
                line,
                (left, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                TEXT_MUTED,
                1,
                cv2.LINE_AA,
            )

            y += 28

    cv2.line(
        panel,
        (left, 534),
        (SIDEBAR_WIDTH - left, 534),
        DIVIDER,
        1,
        cv2.LINE_AA,
    )

    controls = (
        "0 Auto   1-7 Preview",
        "E Effects   D Details",
        "Q Quit",
    )

    y = 580

    for line in controls:
        cv2.putText(
            panel,
            line,
            (left, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

        y += 32

    return panel


def compose_app_frame(
    camera_frame,
    detected_label: str,
    display_label: str,
    confidence: Optional[float],
    preview_label: Optional[str],
    average_ms: float,
    effects_enabled: bool,
    face_detected: bool,
    raw_label: str,
    debug_enabled: bool,
    display_fps: float = 0.0,
):
    """Compose the camera-first layout and minimal neutral sidebar."""
    raw_accent = EMOTION_COLORS.get(
        display_label,
        EMOTION_COLORS["?"],
    )

    target_accent = mix_color(
        raw_accent,
        (64, 64, 68),
        0.32,
    )

    # Smooth the sidebar accent so prediction changes do not flash.
    if not hasattr(
        compose_app_frame,
        "_accent",
    ):
        compose_app_frame._accent = np.array(
            target_accent,
            dtype=np.float32,
        )

    current_accent = (
        compose_app_frame._accent
    )

    current_accent += (
        np.array(
            target_accent,
            dtype=np.float32,
        )
        - current_accent
    ) * 0.10

    accent_color = clamp_color(
        current_accent
    )

    camera_view = fit_camera_view(
        camera_frame,
        VIDEO_WIDTH,
        APP_HEIGHT,
    )

    panel = np.full(
        (
            APP_HEIGHT,
            SIDEBAR_WIDTH,
            3,
        ),
        PANEL_BG,
        dtype=np.uint8,
    )

    if PIL_AVAILABLE:
        panel = draw_sidebar_pillow(
            panel=panel,
            display_label=display_label,
            confidence=confidence,
            preview_label=preview_label,
            effects_enabled=effects_enabled,
            face_detected=face_detected,
            detected_label=detected_label,
            raw_label=raw_label,
            average_ms=average_ms,
            display_fps=display_fps,
            debug_enabled=debug_enabled,
            accent_color=accent_color,
        )
    else:
        panel = draw_sidebar_opencv_fallback(
            panel=panel,
            display_label=display_label,
            confidence=confidence,
            preview_label=preview_label,
            effects_enabled=effects_enabled,
            face_detected=face_detected,
            detected_label=detected_label,
            raw_label=raw_label,
            average_ms=average_ms,
            display_fps=display_fps,
            debug_enabled=debug_enabled,
            accent_color=accent_color,
        )

    canvas = np.zeros(
        (
            APP_HEIGHT,
            APP_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        :,
        :VIDEO_WIDTH,
    ] = camera_view

    canvas[
        :,
        VIDEO_WIDTH:APP_WIDTH,
    ] = panel

    cv2.line(
        canvas,
        (VIDEO_WIDTH, 0),
        (VIDEO_WIDTH, APP_HEIGHT),
        DIVIDER,
        1,
        cv2.LINE_AA,
    )

    return canvas


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

    # The app canvas is always 1280 x 720. WINDOW_AUTOSIZE
    # prevents accidental manual resizing during the demo.
    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_AUTOSIZE,
    )

    # RTMPose and the classifier are owned by exactly one background thread.
    # The main thread never calls recognizer.predict() in V10.
    inference_worker = AsyncExpressionWorker(recognizer)
    inference_worker.start()

    label_history = deque(
        maxlen=SMOOTH_WINDOW
    )

    prediction_timing_history = deque(
        maxlen=PREDICTION_TIMING_WINDOW
    )

    display_interval_history = deque(
        maxlen=DISPLAY_TIMING_WINDOW
    )

    last_presented_at: Optional[float] = None

    stable_label = "?"
    raw_label = "?"

    current_confidence: Optional[float] = None

    missing_face_frames = 0
    frame_index = 0
    live_prediction_count = 0
    last_consumed_result_sequence = 0
    discard_result_through_sequence = 0

    effects_enabled = True
    preview_label: Optional[str] = None
    debug_enabled = False

    print(
        "RTMPose v10 threaded wide-screen effects demo started."
    )
    print(
        "Inference: one RTMPose worker with a latest-frame buffer."
    )

    print("Controls:")
    print("  Q - Quit")
    print("  E - Enable or disable effects")
    print("  D - Show or hide debug details")
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
            ret, frame = cap.read()

            if not ret:
                print(
                    "Error: Could not read a frame "
                    "from the camera."
                )
                break

            frame_index += 1

            detection = face_detector.detect(
                frame
            )

            clipped_face = detection.box
            detector_source = detection.source

            displayed_face: Optional[FaceBox] = None
            face_detected = clipped_face is not None

            if clipped_face is not None:
                missing_face_frames = 0

                x, y, w, h = clipped_face

                # Submit the newest fresh crop without waiting for RTMPose.
                # AsyncExpressionWorker keeps only one waiting crop, so an old
                # camera frame can never accumulate behind newer frames.
                if detection.is_fresh:
                    face_roi = frame[
                        y:y + h,
                        x:x + w,
                    ]
                    inference_worker.submit(face_roi)

                # Consume each completed prediction exactly once. The worker
                # publishes immutable timing values together with the result.
                completed = inference_worker.get_latest_result(
                    after_sequence=last_consumed_result_sequence
                )

                if completed is not None:
                    last_consumed_result_sequence = completed.sequence

                    if completed.error is not None:
                        print(
                            "Async inference error: "
                            f"{completed.error}"
                        )
                    elif completed.sequence > discard_result_through_sequence:
                        prediction = completed.prediction

                        if prediction is not None and prediction.success:
                            raw_label = prediction.label
                            current_confidence = prediction.confidence

                            if completed.prediction_ms is not None:
                                live_prediction_count += 1
                                # Exclude only the first few live samples, just
                                # as the standalone benchmark excludes warm-up.
                                if live_prediction_count > LIVE_TIMING_WARMUP_SAMPLES:
                                    prediction_timing_history.append(
                                        float(completed.prediction_ms)
                                    )

                            label_history.append(raw_label)
                            candidate_label, candidate_votes = (
                                Counter(label_history).most_common(1)[0]
                            )

                            if candidate_votes >= MIN_VOTES:
                                stable_label = candidate_label
                        else:
                            raw_label = "?"
                            current_confidence = None

                display_label = (
                    preview_label
                    if preview_label is not None
                    else stable_label
                )

                if effects_enabled:
                    # Fill the complete camera frame first. The wide layer
                    # uses the whole asset collection over time while keeping
                    # the centre of the face mostly clear.
                    wide_renderer.render_background(
                        frame=frame,
                        label=display_label,
                        face_box=clipped_face,
                        frame_index=frame_index,
                        opacity=1.0,
                    )

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

                    if display_label == "happy":
                        happy_effect_renderer.render(
                            frame=frame,
                            label=display_label,
                            face_box=sticker_box,
                            frame_index=frame_index,
                            opacity=1.0,
                        )

                        surprise_effect_renderer.update_label(
                            display_label
                        )

                    elif display_label == "surprise":
                        surprise_effect_renderer.render(
                            frame=frame,
                            label=display_label,
                            face_box=sticker_box,
                            frame_index=frame_index,
                            opacity=1.0,
                        )

                        happy_effect_renderer.update_label(
                            display_label
                        )

                    else:
                        happy_effect_renderer.update_label(
                            display_label
                        )

                        surprise_effect_renderer.update_label(
                            display_label
                        )

                        # Angry, Disgust, Fear, Neutral and Sad receive a
                        # richer face-local layer in addition to the wide
                        # atmosphere above.
                        wide_renderer.render_local(
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

            else:
                missing_face_frames += 1
                raw_label = "?"
                current_confidence = None

                # Drop any crop that has not started yet and ignore a result
                # from a face that disappeared before inference completed.
                discard_result_through_sequence = max(
                    discard_result_through_sequence,
                    inference_worker.clear_pending(),
                )

                if (
                    missing_face_frames
                    >= RESET_AFTER_MISSING
                ):
                    label_history.clear()
                    stable_label = "?"
                    effect_engine.reset()
                    happy_effect_renderer.reset()
                    surprise_effect_renderer.reset()
                    wide_renderer.reset()

            if debug_enabled:
                source_text = {
                    "mediapipe": "MediaPipe",
                    "haar": "Haar fallback",
                    "hold": (
                        f"Held box "
                        f"{detection.missed_frames}/"
                        f"{face_detector.hold_frames}"
                    ),
                    "none": "No detector",
                }.get(
                    detector_source,
                    detector_source,
                )

                cv2.putText(
                    frame,
                    f"Detector: {source_text}",
                    (18, frame.shape[0] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (240, 240, 245),
                    2,
                    cv2.LINE_AA,
                )

            display_label = (
                preview_label
                if preview_label is not None
                else stable_label
            )

            average_ms = (
                sum(prediction_timing_history)
                / len(prediction_timing_history)
                if prediction_timing_history
                else 0.0
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
                else 0.0
            )

            app_frame = compose_app_frame(
                camera_frame=frame,
                detected_label=stable_label,
                display_label=display_label,
                confidence=current_confidence,
                preview_label=preview_label,
                average_ms=average_ms,
                effects_enabled=effects_enabled,
                face_detected=face_detected,
                raw_label=raw_label,
                debug_enabled=debug_enabled,
                display_fps=display_fps,
            )

            cv2.imshow(
                WINDOW_NAME,
                app_frame,
            )

            presented_at = time.perf_counter()
            if last_presented_at is not None:
                display_interval_history.append(
                    (presented_at - last_presented_at) * 1000.0
                )
            last_presented_at = presented_at

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
                    happy_effect_renderer.reset()
                    surprise_effect_renderer.reset()
                    wide_renderer.reset()

                status = (
                    "enabled"
                    if effects_enabled
                    else "disabled"
                )

                print(
                    f"Visual effects {status}."
                )

            if key in (
                ord("d"),
                ord("D"),
            ):
                debug_enabled = (
                    not debug_enabled
                )

                status = (
                    "shown"
                    if debug_enabled
                    else "hidden"
                )

                print(
                    f"Debug details {status}."
                )

            if key in PREVIEW_KEYS:
                preview_label = PREVIEW_KEYS[key]
                effect_engine.reset()
                happy_effect_renderer.reset()
                surprise_effect_renderer.reset()
                wide_renderer.reset()

                print(
                    f"Preview mode: {preview_label}"
                )

            if key == ord("0"):
                preview_label = None
                effect_engine.reset()
                happy_effect_renderer.reset()
                surprise_effect_renderer.reset()
                wide_renderer.reset()

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
        inference_worker.close()
        face_detector.close()
        cv2.destroyAllWindows()

        print(
            "Emotion recognition demo closed."
        )


if __name__ == "__main__":
    main()
