import math
from typing import Optional, Tuple

import cv2
import numpy as np


FaceBox = Tuple[int, int, int, int]


# OpenCV uses BGR rather than RGB.
EMOTION_COLORS = {
    "angry": (35, 35, 245),
    "disgust": (55, 185, 80),
    "fear": (190, 100, 195),
    "happy": (0, 215, 255),
    "neutral": (215, 215, 215),
    "sad": (255, 155, 55),
    "surprise": (0, 245, 255),
    "?": (180, 180, 180),
}


class EmotionEffects:
    """
    Lightweight real-time emotion effect renderer.

    Design goals:
    - effects should not cover the face;
    - all drawing should remain lightweight;
    - the face box should move smoothly but stay responsive;
    - label transitions should be short.
    """

    def __init__(
        self,
        box_smoothing: float = 0.70,
        transition_frames: int = 2,
    ):
        self.box_smoothing = float(
            np.clip(box_smoothing, 0.0, 1.0)
        )

        self.transition_frames = max(
            1,
            int(transition_frames),
        )

        self.smoothed_box: Optional[np.ndarray] = None

        self.current_label = "?"
        self.previous_label = "?"
        self.transition_progress = self.transition_frames

    def reset(self):
        """Reset all temporal effect state."""
        self.smoothed_box = None

        self.current_label = "?"
        self.previous_label = "?"

        self.transition_progress = self.transition_frames

    def _smooth_face_box(
        self,
        face_box: FaceBox,
    ) -> FaceBox:
        """
        Smooth the box using an exponential moving average.

        A larger alpha gives faster response to current movement.
        """
        current = np.asarray(
            face_box,
            dtype=np.float32,
        )

        if self.smoothed_box is None:
            self.smoothed_box = current
        else:
            alpha = self.box_smoothing

            self.smoothed_box = (
                alpha * current
                + (1.0 - alpha) * self.smoothed_box
            )

        x, y, w, h = np.rint(
            self.smoothed_box
        ).astype(int)

        return (
            int(x),
            int(y),
            max(1, int(w)),
            max(1, int(h)),
        )

    def _update_transition(
        self,
        label: str,
    ):
        """Start a short transition when the stable label changes."""
        if label != self.current_label:
            self.previous_label = self.current_label
            self.current_label = label
            self.transition_progress = 0
        else:
            self.transition_progress = min(
                self.transition_progress + 1,
                self.transition_frames,
            )

    def render(
        self,
        frame: np.ndarray,
        label: str,
        face_box: Optional[FaceBox],
        frame_index: int,
        confidence: Optional[float] = None,
    ) -> Optional[FaceBox]:
        """
        Draw an emotion effect and return the smoothed face box.
        """
        if face_box is None:
            return None

        display_box = self._smooth_face_box(
            face_box
        )

        self._update_transition(label)

        ratio = (
            self.transition_progress
            / self.transition_frames
        )

        ratio = float(
            np.clip(ratio, 0.0, 1.0)
        )

        previous_alpha = 1.0 - ratio
        current_alpha = ratio

        if (
            self.previous_label != "?"
            and previous_alpha > 0.05
        ):
            self._render_label(
                frame=frame,
                label=self.previous_label,
                face_box=display_box,
                frame_index=frame_index,
                alpha=previous_alpha,
            )

        if self.current_label != "?":
            self._render_label(
                frame=frame,
                label=self.current_label,
                face_box=display_box,
                frame_index=frame_index,
                alpha=max(current_alpha, 0.25),
            )
        else:
            self._draw_neutral_frame(
                frame=frame,
                face_box=display_box,
                color=EMOTION_COLORS["?"],
                alpha=0.50,
            )

        return display_box

    def _render_label(
        self,
        frame: np.ndarray,
        label: str,
        face_box: FaceBox,
        frame_index: int,
        alpha: float,
    ):
        alpha = float(
            np.clip(alpha, 0.0, 1.0)
        )

        if alpha <= 0.01:
            return

        if label == "happy":
            self._draw_happy(
                frame,
                face_box,
                frame_index,
                alpha,
            )

        elif label == "sad":
            self._draw_sad(
                frame,
                face_box,
                frame_index,
                alpha,
            )

        elif label == "angry":
            self._draw_angry(
                frame,
                face_box,
                frame_index,
                alpha,
            )

        elif label == "surprise":
            self._draw_surprise(
                frame,
                face_box,
                frame_index,
                alpha,
            )

        elif label == "fear":
            self._draw_fear(
                frame,
                face_box,
                frame_index,
                alpha,
            )

        elif label == "disgust":
            self._draw_disgust(
                frame,
                face_box,
                frame_index,
                alpha,
            )

        else:
            self._draw_neutral_frame(
                frame=frame,
                face_box=face_box,
                color=EMOTION_COLORS.get(
                    label,
                    EMOTION_COLORS["?"],
                ),
                alpha=alpha,
            )

    @staticmethod
    def _effect_region(
        frame: np.ndarray,
        face_box: FaceBox,
        padding_x: float = 0.42,
        padding_y: float = 0.42,
    ):
        """
        Return a local region around the face.

        Effects are blended only in this region, not over the
        complete video frame.
        """
        x, y, w, h = face_box
        frame_h, frame_w = frame.shape[:2]

        pad_x = int(w * padding_x)
        pad_y = int(h * padding_y)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)

        x2 = min(
            frame_w,
            x + w + pad_x,
        )

        y2 = min(
            frame_h,
            y + h + pad_y,
        )

        if x1 >= x2 or y1 >= y2:
            return None

        local_box = (
            x - x1,
            y - y1,
            w,
            h,
        )

        return x1, y1, x2, y2, local_box

    @staticmethod
    def _blend_region(
        target: np.ndarray,
        overlay: np.ndarray,
        alpha: float,
    ):
        alpha = float(
            np.clip(alpha, 0.0, 1.0)
        )

        if alpha <= 0.001:
            return

        cv2.addWeighted(
            overlay,
            alpha,
            target,
            1.0 - alpha,
            0,
            dst=target,
        )

    def _draw_happy(
        self,
        frame: np.ndarray,
        face_box: FaceBox,
        frame_index: int,
        alpha: float,
    ):
        """
        Happy:
        - six small orbiting stars;
        - six pieces of subtle confetti;
        - four decorative corner marks;
        - no large ring and no heavy face covering.
        """
        region = self._effect_region(
            frame,
            face_box,
            padding_x=0.46,
            padding_y=0.46,
        )

        if region is None:
            return

        x1, y1, x2, y2, local_box = region

        target = frame[y1:y2, x1:x2]
        overlay = target.copy()

        x, y, w, h = local_box
        cx = x + w // 2
        cy = y + h // 2

        color = EMOTION_COLORS["happy"]

        # Thin corner decorations instead of a full rectangle.
        self._draw_corner_frame(
            overlay,
            local_box,
            color,
            thickness=3,
        )

        # Small orbiting stars outside the face.
        star_count = 6

        for i in range(star_count):
            phase = (
                i
                * 2.0
                * math.pi
                / star_count
            )

            angle = (
                frame_index * 0.045
                + phase
            )

            radius_x = w * 0.67
            radius_y = h * 0.67

            star_x = int(
                cx + radius_x * math.cos(angle)
            )

            star_y = int(
                cy + radius_y * math.sin(angle)
            )

            twinkle = (
                0.78
                + 0.22
                * math.sin(
                    frame_index * 0.16
                    + i * 1.5
                )
            )

            radius = max(
                4,
                int(
                    min(w, h)
                    * 0.027
                    * twinkle
                ),
            )

            self._draw_star(
                overlay,
                (star_x, star_y),
                radius,
                color,
            )

        # A small amount of confetti.
        confetti_colors = (
            (0, 215, 255),
            (80, 220, 255),
            (100, 225, 155),
            (255, 175, 105),
            (210, 130, 240),
        )

        for i in range(6):
            horizontal_ratio = (
                ((i * 73 + 19) % 997)
                / 997.0
            )

            vertical_range = max(
                int(h * 1.25),
                1,
            )

            vertical_offset = (
                frame_index
                * (1.3 + 0.25 * (i % 3))
                + (i * 53 + 17)
            ) % vertical_range

            point_x = int(
                x
                - w * 0.22
                + horizontal_ratio * w * 1.44
            )

            point_y = int(
                y
                - h * 0.28
                + vertical_offset
            )

            point_x += int(
                4
                * math.sin(
                    frame_index * 0.06 + i
                )
            )

            length = max(
                3,
                int(min(w, h) * 0.020),
            )

            rotation = (
                frame_index * 0.10
                + i * 0.85
            )

            dx = int(
                math.cos(rotation) * length
            )

            dy = int(
                math.sin(rotation) * length
            )

            cv2.line(
                overlay,
                (point_x - dx, point_y - dy),
                (point_x + dx, point_y + dy),
                confetti_colors[
                    i % len(confetti_colors)
                ],
                2,
                cv2.LINE_AA,
            )

        self._blend_region(
            target,
            overlay,
            0.78 * alpha,
        )

    def _draw_sad(
        self,
        frame: np.ndarray,
        face_box: FaceBox,
        frame_index: int,
        alpha: float,
    ):
        """Sad: small rain drops outside the face."""
        region = self._effect_region(
            frame,
            face_box,
            padding_x=0.28,
            padding_y=0.35,
        )

        if region is None:
            return

        x1, y1, x2, y2, local_box = region

        target = frame[y1:y2, x1:x2]
        overlay = target.copy()

        x, y, w, h = local_box
        color = EMOTION_COLORS["sad"]

        self._draw_corner_frame(
            overlay,
            local_box,
            color,
            thickness=2,
        )

        fall = (
            frame_index * 7
        ) % max(h, 1)

        for i in range(6):
            drop_x = int(
                x
                - w * 0.12
                + i * w * 0.24
            )

            drop_y = int(
                y
                - h * 0.25
                + (fall + i * 37)
                % int(h * 1.40)
            )

            cv2.line(
                overlay,
                (drop_x, drop_y),
                (drop_x - 4, drop_y + 14),
                color,
                2,
                cv2.LINE_AA,
            )

        self._blend_region(
            target,
            overlay,
            0.72 * alpha,
        )

    def _draw_angry(
        self,
        frame: np.ndarray,
        face_box: FaceBox,
        frame_index: int,
        alpha: float,
    ):
        """Angry: red corner pulse and a small anger mark."""
        region = self._effect_region(
            frame,
            face_box,
            padding_x=0.20,
            padding_y=0.20,
        )

        if region is None:
            return

        x1, y1, x2, y2, local_box = region

        target = frame[y1:y2, x1:x2]
        overlay = target.copy()

        x, y, w, h = local_box
        color = EMOTION_COLORS["angry"]

        pulse = (
            2
            if frame_index % 2 == 0
            else 3
        )

        self._draw_corner_frame(
            overlay,
            local_box,
            color,
            thickness=pulse,
        )

        mark_x = x + int(w * 0.83)
        mark_y = y + int(h * 0.12)

        size = max(
            9,
            int(min(w, h) * 0.06),
        )

        cv2.line(
            overlay,
            (mark_x - size, mark_y),
            (mark_x, mark_y + size // 2),
            color,
            3,
            cv2.LINE_AA,
        )

        cv2.line(
            overlay,
            (mark_x, mark_y + size // 2),
            (mark_x + size, mark_y),
            color,
            3,
            cv2.LINE_AA,
        )

        self._blend_region(
            target,
            overlay,
            0.76 * alpha,
        )

    def _draw_surprise(
        self,
        frame: np.ndarray,
        face_box: FaceBox,
        frame_index: int,
        alpha: float,
    ):
        """Surprise: short rays outside the face."""
        region = self._effect_region(
            frame,
            face_box,
            padding_x=0.38,
            padding_y=0.38,
        )

        if region is None:
            return

        x1, y1, x2, y2, local_box = region

        target = frame[y1:y2, x1:x2]
        overlay = target.copy()

        x, y, w, h = local_box
        cx = x + w // 2
        cy = y + h // 2

        color = EMOTION_COLORS["surprise"]

        for i in range(10):
            angle = (
                i * 2 * math.pi / 10
            )

            inner_x = int(
                cx + w * 0.60 * math.cos(angle)
            )

            inner_y = int(
                cy + h * 0.60 * math.sin(angle)
            )

            outer_x = int(
                cx + w * 0.72 * math.cos(angle)
            )

            outer_y = int(
                cy + h * 0.72 * math.sin(angle)
            )

            cv2.line(
                overlay,
                (inner_x, inner_y),
                (outer_x, outer_y),
                color,
                2,
                cv2.LINE_AA,
            )

        self._blend_region(
            target,
            overlay,
            0.75 * alpha,
        )

    def _draw_fear(
        self,
        frame: np.ndarray,
        face_box: FaceBox,
        frame_index: int,
        alpha: float,
    ):
        """Fear: trembling four-corner frame and a sweat drop."""
        region = self._effect_region(
            frame,
            face_box,
            padding_x=0.16,
            padding_y=0.20,
        )

        if region is None:
            return

        x1, y1, x2, y2, local_box = region

        target = frame[y1:y2, x1:x2]
        overlay = target.copy()

        x, y, w, h = local_box
        color = EMOTION_COLORS["fear"]

        shift = int(
            2
            * math.sin(
                frame_index * 0.8
            )
        )

        # Keep the trembling motion, but use the same four-corner frame
        # style as the other emotions instead of drawing a full rectangle.
        shifted_box = (x + shift, y, w, h)
        self._draw_corner_frame(
            overlay,
            shifted_box,
            color,
            thickness=2,
        )

        drop_x = x + int(w * 0.82)
        drop_y = y + int(h * 0.15)

        cv2.ellipse(
            overlay,
            (drop_x, drop_y),
            (
                max(3, int(w * 0.022)),
                max(6, int(h * 0.040)),
            ),
            0,
            0,
            360,
            color,
            -1,
            cv2.LINE_AA,
        )

        self._blend_region(
            target,
            overlay,
            0.68 * alpha,
        )

    def _draw_disgust(
        self,
        frame: np.ndarray,
        face_box: FaceBox,
        frame_index: int,
        alpha: float,
    ):
        """Disgust: small green bubbles near the lower face."""
        region = self._effect_region(
            frame,
            face_box,
            padding_x=0.16,
            padding_y=0.16,
        )

        if region is None:
            return

        x1, y1, x2, y2, local_box = region

        target = frame[y1:y2, x1:x2]
        overlay = target.copy()

        x, y, w, h = local_box
        color = EMOTION_COLORS["disgust"]

        self._draw_corner_frame(
            overlay,
            local_box,
            color,
            thickness=2,
        )

        for i in range(5):
            bubble_x = int(
                x
                + w * (0.25 + i * 0.13)
                + 3
                * math.sin(
                    frame_index * 0.08 + i
                )
            )

            bubble_y = int(
                y
                + h * (0.72 + 0.06 * (i % 2))
            )

            radius = max(
                3,
                int(
                    min(w, h)
                    * (0.018 + 0.006 * (i % 3))
                ),
            )

            cv2.circle(
                overlay,
                (bubble_x, bubble_y),
                radius,
                color,
                2,
                cv2.LINE_AA,
            )

        self._blend_region(
            target,
            overlay,
            0.70 * alpha,
        )

    def _draw_neutral_frame(
        self,
        frame: np.ndarray,
        face_box: FaceBox,
        color,
        alpha: float,
    ):
        """Neutral: very subtle corner frame."""
        region = self._effect_region(
            frame,
            face_box,
            padding_x=0.05,
            padding_y=0.05,
        )

        if region is None:
            return

        x1, y1, x2, y2, local_box = region

        target = frame[y1:y2, x1:x2]
        overlay = target.copy()

        self._draw_corner_frame(
            overlay,
            local_box,
            color,
            thickness=2,
        )

        self._blend_region(
            target,
            overlay,
            0.58 * alpha,
        )

    @staticmethod
    def _draw_corner_frame(
        image: np.ndarray,
        face_box: FaceBox,
        color,
        thickness: int = 2,
    ):
        """Draw four short corner brackets instead of a full box."""
        x, y, w, h = face_box

        length = max(
            12,
            int(min(w, h) * 0.14),
        )

        # Top-left
        cv2.line(
            image,
            (x, y),
            (x + length, y),
            color,
            thickness,
            cv2.LINE_AA,
        )

        cv2.line(
            image,
            (x, y),
            (x, y + length),
            color,
            thickness,
            cv2.LINE_AA,
        )

        # Top-right
        cv2.line(
            image,
            (x + w, y),
            (x + w - length, y),
            color,
            thickness,
            cv2.LINE_AA,
        )

        cv2.line(
            image,
            (x + w, y),
            (x + w, y + length),
            color,
            thickness,
            cv2.LINE_AA,
        )

        # Bottom-left
        cv2.line(
            image,
            (x, y + h),
            (x + length, y + h),
            color,
            thickness,
            cv2.LINE_AA,
        )

        cv2.line(
            image,
            (x, y + h),
            (x, y + h - length),
            color,
            thickness,
            cv2.LINE_AA,
        )

        # Bottom-right
        cv2.line(
            image,
            (x + w, y + h),
            (x + w - length, y + h),
            color,
            thickness,
            cv2.LINE_AA,
        )

        cv2.line(
            image,
            (x + w, y + h),
            (x + w, y + h - length),
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_star(
        image: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        color,
    ):
        """Draw a small five-point star."""
        cx, cy = center
        points = []

        for i in range(10):
            angle = (
                -math.pi / 2
                + i * math.pi / 5
            )

            point_radius = (
                radius
                if i % 2 == 0
                else radius * 0.42
            )

            px = int(
                cx
                + point_radius
                * math.cos(angle)
            )

            py = int(
                cy
                + point_radius
                * math.sin(angle)
            )

            points.append([px, py])

        polygon = np.asarray(
            points,
            dtype=np.int32,
        )

        cv2.fillPoly(
            image,
            [polygon],
            color,
            cv2.LINE_AA,
        )