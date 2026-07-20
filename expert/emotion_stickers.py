import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


FaceBox = Tuple[int, int, int, int]


class EmotionStickerRenderer:
    """
    Load and render transparent PNG emotion stickers.

    Supported emotions:
    - angry
    - disgust
    - fear
    - happy
    - neutral
    - sad
    - surprise
    """

    LAYOUTS = {
        "happy": {
            "face": ("happy/face.png", (0.10, -0.28), 0.32, 1.00),
            "badge": ("happy/badge.png", (0.56, -0.24), 0.64, 1.00),
            "deco1": ("happy/star_cluster.png", (1.02, 0.03), 0.24, 0.92),
            "deco2": ("happy/party_popper.png", (1.02, 0.82), 0.22, 0.88),
        },
        "sad": {
            "face": ("sad/face.png", (0.11, -0.28), 0.30, 0.98),
            "badge": ("sad/badge.png", (0.56, -0.24), 0.68, 1.00),
            "deco1": ("sad/rain_cloud.png", (0.56, -0.42), 0.32, 0.92),
            "deco2": ("sad/tear_pair.png", (1.01, 0.20), 0.20, 0.86),
        },
        "angry": {
            "face": ("angry/face.png", (0.10, -0.28), 0.31, 1.00),
            "badge": ("angry/badge.png", (0.56, -0.24), 0.62, 1.00),
            "deco1": ("angry/anger_mark.png", (0.97, 0.00), 0.18, 0.92),
            "deco2": ("angry/flame.png", (1.01, 0.82), 0.22, 0.85),
        },
        "surprise": {
            "face": ("surprise/face.png", (0.10, -0.28), 0.31, 1.00),
            "badge": ("surprise/badge.png", (0.56, -0.24), 0.58, 1.00),
            "deco1": ("surprise/exclamation.png", (0.98, -0.04), 0.19, 0.92),
            "deco2": ("surprise/burst.png", (1.03, 0.80), 0.21, 0.82),
        },
        "fear": {
            "face": ("fear/face.png", (0.10, -0.28), 0.31, 1.00),
            "badge": ("fear/badge.png", (0.56, -0.24), 0.62, 1.00),
            "deco1": ("fear/ghost_spark.png", (0.99, 0.06), 0.21, 0.80),
            "deco2": ("fear/sweat.png", (0.92, -0.01), 0.17, 0.90),
        },
        "disgust": {
            "face": ("disgust/face.png", (0.10, -0.28), 0.31, 1.00),
            "badge": ("disgust/badge.png", (0.56, -0.24), 0.56, 1.00),
            "deco1": ("disgust/stink_lines.png", (1.02, 0.18), 0.19, 0.85),
            "deco2": ("disgust/bubbles.png", (0.99, 0.80), 0.20, 0.82),
        },
        "neutral": {
            "face": ("neutral/face.png", (0.10, -0.28), 0.29, 1.00),
            "badge": ("neutral/badge.png", (0.56, -0.24), 0.58, 1.00),
            "deco1": ("neutral/calm_spark.png", (1.00, 0.02), 0.18, 0.72),
            "deco2": ("neutral/chill_lines.png", (1.00, 0.80), 0.20, 0.72),
        },
    }

    def __init__(
        self,
        assets_root: Optional[Path] = None,
    ):
        self.assets_root = (
            Path(assets_root)
            if assets_root is not None
            else Path(__file__).resolve().parent
            / "assets"
            / "effects"
        )

        self._cache: Dict[str, Optional[np.ndarray]] = {}

    def _load(
        self,
        relative_path: str,
    ) -> Optional[np.ndarray]:
        """Load and cache one transparent BGRA PNG."""
        if relative_path in self._cache:
            return self._cache[relative_path]

        path = self.assets_root / relative_path

        if not path.exists():
            print(f"Warning: sticker not found: {path}")
            self._cache[relative_path] = None
            return None

        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED,
        )

        if (
            image is None
            or image.ndim != 3
            or image.shape[2] != 4
        ):
            print(
                "Warning: sticker must be a transparent "
                f"BGRA PNG: {path}"
            )
            self._cache[relative_path] = None
            return None

        self._cache[relative_path] = image
        return image

    def render(
        self,
        frame: np.ndarray,
        label: str,
        face_box: FaceBox,
        frame_index: int,
        opacity: float = 1.0,
    ):
        """Render a sticker composition for the requested emotion."""
        if label not in self.LAYOUTS:
            return

        x, y, w, h = face_box

        opacity = float(np.clip(opacity, 0.0, 1.0))
        if opacity <= 0.01:
            return

        layout = self.LAYOUTS[label]

        floating_y = int(3 * math.sin(frame_index * 0.08))
        pop_scale = 0.84 + 0.16 * opacity

        # Main face sticker
        face_file, face_anchor, face_ratio, face_opacity = layout["face"]
        self._place(
            frame=frame,
            file_path=face_file,
            face_box=face_box,
            anchor=face_anchor,
            width_ratio=face_ratio * pop_scale,
            opacity=opacity * face_opacity,
            vertical_offset=floating_y,
        )

        # Badge sticker
        badge_file, badge_anchor, badge_ratio, badge_opacity = layout["badge"]
        self._place(
            frame=frame,
            file_path=badge_file,
            face_box=face_box,
            anchor=badge_anchor,
            width_ratio=badge_ratio * pop_scale,
            opacity=opacity * badge_opacity,
            vertical_offset=floating_y,
        )

        # Decoration 1
        deco1_file, deco1_anchor, deco1_ratio, deco1_opacity = layout["deco1"]
        pulse1 = 0.96 + 0.05 * math.sin(frame_index * 0.12)
        self._place(
            frame=frame,
            file_path=deco1_file,
            face_box=face_box,
            anchor=deco1_anchor,
            width_ratio=deco1_ratio * pulse1,
            opacity=opacity * deco1_opacity,
            vertical_offset=floating_y,
        )

        # Decoration 2
        deco2_file, deco2_anchor, deco2_ratio, deco2_opacity = layout["deco2"]
        pulse2 = 0.97 + 0.04 * math.cos(frame_index * 0.10)
        self._place(
            frame=frame,
            file_path=deco2_file,
            face_box=face_box,
            anchor=deco2_anchor,
            width_ratio=deco2_ratio * pulse2,
            opacity=opacity * deco2_opacity,
            vertical_offset=0,
        )

    def _place(
        self,
        frame: np.ndarray,
        file_path: str,
        face_box: FaceBox,
        anchor: Tuple[float, float],
        width_ratio: float,
        opacity: float,
        vertical_offset: int = 0,
    ):
        """Place a sticker relative to the face box."""
        rgba = self._load(file_path)
        if rgba is None:
            return

        x, y, w, h = face_box
        ax, ay = anchor

        center_x = int(x + w * ax)
        center_y = int(y + h * ay + vertical_offset)

        target_width = max(1, int(w * width_ratio))

        self.overlay_rgba(
            frame=frame,
            rgba_image=rgba,
            center=(center_x, center_y),
            target_width=target_width,
            opacity=opacity,
        )

    @staticmethod
    def overlay_rgba(
        frame: np.ndarray,
        rgba_image: Optional[np.ndarray],
        center: Tuple[int, int],
        target_width: int,
        opacity: float = 1.0,
    ):
        """Overlay one transparent BGRA PNG on a BGR frame."""
        if rgba_image is None:
            return

        if (
            rgba_image.ndim != 3
            or rgba_image.shape[2] != 4
        ):
            return

        opacity = float(np.clip(opacity, 0.0, 1.0))
        if opacity <= 0.001:
            return

        target_width = max(1, int(target_width))

        source_h, source_w = rgba_image.shape[:2]
        if source_h <= 0 or source_w <= 0:
            return

        scale = target_width / source_w
        target_height = max(1, int(source_h * scale))

        interpolation = (
            cv2.INTER_AREA
            if target_width < source_w
            else cv2.INTER_LINEAR
        )

        resized = cv2.resize(
            rgba_image,
            (target_width, target_height),
            interpolation=interpolation,
        )

        center_x, center_y = center

        x1 = int(center_x - target_width / 2)
        y1 = int(center_y - target_height / 2)

        x2 = x1 + target_width
        y2 = y1 + target_height

        frame_h, frame_w = frame.shape[:2]

        clip_x1 = max(0, x1)
        clip_y1 = max(0, y1)
        clip_x2 = min(frame_w, x2)
        clip_y2 = min(frame_h, y2)

        if (
            clip_x1 >= clip_x2
            or clip_y1 >= clip_y2
        ):
            return

        source_x1 = clip_x1 - x1
        source_y1 = clip_y1 - y1
        source_x2 = source_x1 + (clip_x2 - clip_x1)
        source_y2 = source_y1 + (clip_y2 - clip_y1)

        asset_crop = resized[
            source_y1:source_y2,
            source_x1:source_x2,
        ]

        frame_crop = frame[
            clip_y1:clip_y2,
            clip_x1:clip_x2,
        ]

        asset_bgr = asset_crop[:, :, :3].astype(np.float32)

        asset_alpha = (
            asset_crop[:, :, 3].astype(np.float32)
            / 255.0
            * opacity
        )[..., None]

        frame_float = frame_crop.astype(np.float32)

        blended = (
            asset_bgr * asset_alpha
            + frame_float * (1.0 - asset_alpha)
        )

        frame[
            clip_y1:clip_y2,
            clip_x1:clip_x2,
        ] = np.clip(
            blended,
            0,
            255,
        ).astype(np.uint8)