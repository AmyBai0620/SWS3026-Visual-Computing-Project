from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


FaceBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class StickerAsset:
    name: str
    role: str
    image: np.ndarray


@dataclass
class StickerPack:
    main: StickerAsset
    cores: List[StickerAsset]
    fragments: List[StickerAsset]


class WideScreenEmotionRenderer:
    """
    Wide-screen atmosphere and rich local sticker animation.

    The renderer deliberately uses the complete asset pools over time rather
    than choosing only four cores and six fragments forever. Happy and
    Surprise can keep their dedicated face-local renderers while this class
    supplies the wider screen atmosphere for all seven emotions.
    """

    SUPPORTED = (
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    )

    THEMES = {
        "angry": {
            "tint": (20, 24, 210),
            "tint_alpha": 0.055,
            "global_count": 29,
            "local_cores": 8,
            "local_fragments": 13,
            "global_opacity": 0.68,
            "local_opacity": 0.96,
        },
        "disgust": {
            "tint": (50, 150, 62),
            "tint_alpha": 0.045,
            "global_count": 24,
            "local_cores": 7,
            "local_fragments": 14,
            "global_opacity": 0.58,
            "local_opacity": 0.92,
        },
        "fear": {
            "tint": (115, 52, 125),
            "tint_alpha": 0.060,
            "global_count": 25,
            "local_cores": 8,
            "local_fragments": 15,
            "global_opacity": 0.60,
            "local_opacity": 0.94,
        },
        "happy": {
            "tint": (25, 155, 235),
            "tint_alpha": 0.030,
            "global_count": 16,
            "local_cores": 0,
            "local_fragments": 0,
            "global_opacity": 0.52,
            "local_opacity": 0.0,
        },
        "neutral": {
            "tint": (205, 205, 205),
            "tint_alpha": 0.018,
            "global_count": 21,
            "local_cores": 7,
            "local_fragments": 13,
            "global_opacity": 0.40,
            "local_opacity": 0.82,
        },
        "sad": {
            "tint": (170, 105, 35),
            "tint_alpha": 0.060,
            "global_count": 27,
            "local_cores": 7,
            "local_fragments": 14,
            "global_opacity": 0.58,
            "local_opacity": 0.94,
        },
        "surprise": {
            "tint": (30, 170, 235),
            "tint_alpha": 0.025,
            "global_count": 17,
            "local_cores": 0,
            "local_fragments": 0,
            "global_opacity": 0.52,
            "local_opacity": 0.0,
        },
    }

    def __init__(
        self,
        assets_root: Optional[Path] = None,
    ) -> None:
        self.assets_root = (
            Path(assets_root)
            if assets_root is not None
            else Path(__file__).resolve().parent
            / "assets"
            / "effects"
        )

        self.packs: Dict[str, StickerPack] = {}
        self._sprite_cache: Dict[
            Tuple[str, str, int, int, bool],
            np.ndarray,
        ] = {}
        self._vignette_cache: Dict[
            Tuple[int, int, str],
            np.ndarray,
        ] = {}
        self._active_label = "?"
        self._activation_time = time.perf_counter()
        self._load_all_packs()

    def reset(self) -> None:
        self._active_label = "?"
        self._activation_time = time.perf_counter()

    @staticmethod
    def _trim_transparent(
        image: np.ndarray,
        padding: int = 3,
    ) -> np.ndarray:
        alpha = image[:, :, 3]
        points = cv2.findNonZero(
            (alpha > 3).astype(np.uint8)
        )

        if points is None:
            return image

        x, y, width, height = cv2.boundingRect(points)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + width + padding)
        y2 = min(image.shape[0], y + height + padding)
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def _normalise_record(
        record: dict,
    ) -> tuple[str, str, str]:
        source_file = str(record.get("file", "")).strip()
        name = str(
            record.get("name")
            or record.get("label")
            or Path(source_file).stem
        ).strip()
        role = str(
            record.get("role")
            or record.get("type")
            or ""
        ).strip().lower()
        runtime_file = str(
            record.get("runtime_file")
            or record.get("runtime_path")
            or ""
        ).strip()
        return name, role, runtime_file

    def _load_pack(
        self,
        emotion: str,
    ) -> StickerPack:
        root = self.assets_root / emotion
        manifest_path = root / "manifest.json"

        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing effect manifest: {manifest_path}"
            )

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        records = manifest.get("assets")

        if not isinstance(records, list):
            raise ValueError(
                f"Manifest has no assets list: {manifest_path}"
            )

        by_role: Dict[str, List[StickerAsset]] = {
            "main": [],
            "core": [],
            "fragment": [],
        }

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue

            name, role, runtime_file = self._normalise_record(
                record
            )

            if role not in by_role or not runtime_file:
                continue

            if not name:
                name = f"{role}_{index:02d}"

            image_path = root / runtime_file
            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_UNCHANGED,
            )

            if (
                image is None
                or image.ndim != 3
                or image.shape[2] != 4
            ):
                raise RuntimeError(
                    "Effect assets must be transparent BGRA PNGs: "
                    f"{image_path}"
                )

            by_role[role].append(
                StickerAsset(
                    name=name,
                    role=role,
                    image=self._trim_transparent(image),
                )
            )

        if not by_role["main"]:
            raise ValueError(
                f"No main sticker in {manifest_path}"
            )

        if not by_role["core"]:
            raise ValueError(
                f"No core stickers in {manifest_path}"
            )

        if not by_role["fragment"]:
            raise ValueError(
                f"No fragment stickers in {manifest_path}"
            )

        return StickerPack(
            main=by_role["main"][0],
            cores=by_role["core"],
            fragments=by_role["fragment"],
        )

    def _load_all_packs(self) -> None:
        errors = []

        for emotion in self.SUPPORTED:
            try:
                self.packs[emotion] = self._load_pack(
                    emotion
                )
            except Exception as exc:
                errors.append(
                    f"{emotion}: {type(exc).__name__}: {exc}"
                )

        if errors:
            raise RuntimeError(
                "Could not load all emotion sticker packs:\n  "
                + "\n  ".join(errors)
            )

    def _activate(self, emotion: str) -> tuple[float, float]:
        if emotion != self._active_label:
            self._active_label = emotion
            self._activation_time = time.perf_counter()

        elapsed = time.perf_counter() - self._activation_time
        intro = float(np.clip(elapsed / 0.34, 0.0, 1.0))
        intro = 1.0 - (1.0 - intro) ** 3
        return elapsed, intro

    def _get_sprite(
        self,
        emotion: str,
        asset: StickerAsset,
        target_width: int,
        angle: float = 0.0,
        flip: bool = False,
    ) -> np.ndarray:
        target_width = max(2, int(target_width))
        quantized_angle = int(round(angle / 4.0) * 4)
        key = (
            emotion,
            asset.name,
            target_width,
            quantized_angle,
            bool(flip),
        )

        cached = self._sprite_cache.get(key)
        if cached is not None:
            return cached

        source = (
            cv2.flip(asset.image, 1)
            if flip
            else asset.image
        )
        source_h, source_w = source.shape[:2]
        target_height = max(
            2,
            int(round(source_h * target_width / source_w)),
        )
        interpolation = (
            cv2.INTER_AREA
            if target_width < source_w
            else cv2.INTER_LINEAR
        )
        sprite = cv2.resize(
            source,
            (target_width, target_height),
            interpolation=interpolation,
        )

        if quantized_angle:
            center = (
                sprite.shape[1] / 2.0,
                sprite.shape[0] / 2.0,
            )
            matrix = cv2.getRotationMatrix2D(
                center,
                float(quantized_angle),
                1.0,
            )
            cosine = abs(matrix[0, 0])
            sine = abs(matrix[0, 1])
            bound_width = max(
                2,
                int(
                    sprite.shape[0] * sine
                    + sprite.shape[1] * cosine
                ),
            )
            bound_height = max(
                2,
                int(
                    sprite.shape[0] * cosine
                    + sprite.shape[1] * sine
                ),
            )
            matrix[0, 2] += bound_width / 2.0 - center[0]
            matrix[1, 2] += bound_height / 2.0 - center[1]
            sprite = cv2.warpAffine(
                sprite,
                matrix,
                (bound_width, bound_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

        if len(self._sprite_cache) > 1400:
            self._sprite_cache.clear()

        self._sprite_cache[key] = sprite
        return sprite

    @staticmethod
    def _overlay(
        frame: np.ndarray,
        sprite: np.ndarray,
        center_x: float,
        center_y: float,
        opacity: float,
    ) -> None:
        opacity = float(np.clip(opacity, 0.0, 1.0))
        if opacity <= 0.002:
            return

        height, width = sprite.shape[:2]
        x1 = int(round(center_x - width / 2.0))
        y1 = int(round(center_y - height / 2.0))
        x2 = x1 + width
        y2 = y1 + height

        frame_h, frame_w = frame.shape[:2]
        clip_x1 = max(0, x1)
        clip_y1 = max(0, y1)
        clip_x2 = min(frame_w, x2)
        clip_y2 = min(frame_h, y2)

        if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
            return

        source_x1 = clip_x1 - x1
        source_y1 = clip_y1 - y1
        source_x2 = source_x1 + clip_x2 - clip_x1
        source_y2 = source_y1 + clip_y2 - clip_y1

        asset_crop = sprite[
            source_y1:source_y2,
            source_x1:source_x2,
        ].astype(np.float32)
        frame_crop = frame[
            clip_y1:clip_y2,
            clip_x1:clip_x2,
        ].astype(np.float32)

        alpha = (
            asset_crop[:, :, 3:4] / 255.0
        ) * opacity

        frame[
            clip_y1:clip_y2,
            clip_x1:clip_x2,
        ] = np.clip(
            asset_crop[:, :, :3] * alpha
            + frame_crop * (1.0 - alpha),
            0,
            255,
        ).astype(np.uint8)

    @staticmethod
    def _blend_tint(
        frame: np.ndarray,
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        alpha = float(np.clip(alpha, 0.0, 0.20))
        if alpha <= 0.001:
            return

        tint = np.empty_like(frame)
        tint[:] = color
        cv2.addWeighted(
            tint,
            alpha,
            frame,
            1.0 - alpha,
            0.0,
            dst=frame,
        )

    def _apply_vignette(
        self,
        frame: np.ndarray,
        emotion: str,
        strength: float,
    ) -> None:
        height, width = frame.shape[:2]
        key = (height, width, emotion)
        mask = self._vignette_cache.get(key)

        if mask is None:
            yy, xx = np.mgrid[0:height, 0:width]
            nx = (xx - width * 0.5) / max(1.0, width * 0.5)
            ny = (yy - height * 0.5) / max(1.0, height * 0.5)
            radius = np.sqrt(nx * nx + ny * ny)
            mask = np.clip(
                (radius - 0.38) / 0.72,
                0.0,
                1.0,
            ).astype(np.float32)
            mask = cv2.GaussianBlur(
                mask,
                (0, 0),
                sigmaX=max(12.0, width * 0.025),
            )
            self._vignette_cache[key] = mask

        if emotion == "fear":
            color = np.array((70, 24, 74), dtype=np.float32)
        elif emotion == "sad":
            color = np.array((80, 45, 20), dtype=np.float32)
        elif emotion == "angry":
            color = np.array((18, 18, 90), dtype=np.float32)
        else:
            color = np.array((45, 45, 45), dtype=np.float32)

        alpha = (
            mask[:, :, None]
            * float(np.clip(strength, 0.0, 0.24))
        )
        frame_float = frame.astype(np.float32)
        frame[:] = np.clip(
            frame_float * (1.0 - alpha)
            + color[None, None, :] * alpha,
            0,
            255,
        ).astype(np.uint8)

    @staticmethod
    def _avoid_face(
        px: float,
        py: float,
        face_box: FaceBox,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float]:
        x, y, width, height = face_box
        pad_x = width * 0.42
        pad_y = height * 0.34
        x1 = x - pad_x
        y1 = y - pad_y
        x2 = x + width + pad_x
        y2 = y + height + pad_y

        if not (x1 <= px <= x2 and y1 <= py <= y2):
            return px, py

        distances = {
            "left": abs(px - x1),
            "right": abs(x2 - px),
            "top": abs(py - y1),
            "bottom": abs(y2 - py),
        }
        nearest = min(distances, key=distances.get)
        margin = 12.0

        if nearest == "left":
            px = max(0.0, x1 - margin)
        elif nearest == "right":
            px = min(frame_width - 1.0, x2 + margin)
        elif nearest == "top":
            py = max(0.0, y1 - margin)
        else:
            py = min(frame_height - 1.0, y2 + margin)

        return px, py

    @staticmethod
    def _cycle_assets(
        assets: List[StickerAsset],
        count: int,
        frame_index: int,
        shift_frames: int = 28,
    ) -> List[StickerAsset]:
        if not assets or count <= 0:
            return []

        offset = int(frame_index // max(1, shift_frames))
        return [
            assets[(offset + index) % len(assets)]
            for index in range(count)
        ]

    def _global_pool(
        self,
        pack: StickerPack,
    ) -> List[StickerAsset]:
        # Interleave larger cores with small fragments. This keeps the screen
        # visually varied and makes the complete asset collection rotate in.
        pool: List[StickerAsset] = []
        max_length = max(
            len(pack.cores),
            len(pack.fragments),
        )

        for index in range(max_length):
            if index < len(pack.fragments):
                pool.append(pack.fragments[index])
            if index < len(pack.cores):
                pool.append(pack.cores[index])

        return pool

    def render_background(
        self,
        frame: np.ndarray,
        label: str,
        face_box: FaceBox,
        frame_index: int,
        opacity: float = 1.0,
    ) -> None:
        emotion = str(label).strip().lower()
        pack = self.packs.get(emotion)
        theme = self.THEMES.get(emotion)

        if pack is None or theme is None:
            return

        _, intro = self._activate(emotion)
        opacity = float(np.clip(opacity, 0.0, 1.0)) * intro

        if opacity <= 0.01:
            return

        pulse = 0.88 + 0.12 * math.sin(frame_index * 0.055)
        self._blend_tint(
            frame,
            theme["tint"],
            theme["tint_alpha"] * opacity * pulse,
        )

        if emotion in {"angry", "fear", "sad"}:
            self._apply_vignette(
                frame,
                emotion,
                strength=(
                    0.11
                    + 0.025
                    * math.sin(frame_index * 0.045)
                )
                * opacity,
            )

        if emotion == "angry":
            self._render_angry_background(
                frame, pack, face_box, frame_index, opacity
            )
        elif emotion == "disgust":
            self._render_disgust_background(
                frame, pack, face_box, frame_index, opacity
            )
        elif emotion == "fear":
            self._render_fear_background(
                frame, pack, face_box, frame_index, opacity
            )
        elif emotion == "happy":
            self._render_happy_background(
                frame, pack, face_box, frame_index, opacity
            )
        elif emotion == "neutral":
            self._render_neutral_background(
                frame, pack, face_box, frame_index, opacity
            )
        elif emotion == "sad":
            self._render_sad_background(
                frame, pack, face_box, frame_index, opacity
            )
        elif emotion == "surprise":
            self._render_surprise_background(
                frame, pack, face_box, frame_index, opacity
            )

    def _draw_particle(
        self,
        frame: np.ndarray,
        emotion: str,
        asset: StickerAsset,
        x: float,
        y: float,
        width: int,
        angle: float,
        opacity: float,
        flip: bool = False,
    ) -> None:
        sprite = self._get_sprite(
            emotion,
            asset,
            width,
            angle=angle,
            flip=flip,
        )
        self._overlay(
            frame,
            sprite,
            x,
            y,
            opacity,
        )

    @staticmethod
    def _angry_seed(
        index: int,
        channel: int,
    ) -> float:
        """Return a stable pseudo-random value in [0, 1)."""
        value = math.sin(
            (index + 1) * 12.9898
            + (channel + 1) * 78.233
        ) * 43758.5453
        return value - math.floor(value)

    def _render_angry_background(
        self,
        frame: np.ndarray,
        pack: StickerPack,
        face_box: FaceBox,
        frame_index: int,
        opacity: float,
    ) -> None:
        """
        Airborne Angry motion with richer particle density.

        All particles occupy the open screen area. There is no dedicated
        bottom row. Former lower-layer slots are redistributed into fast
        edge arcs and free-flight curves at different screen heights.
        """
        height, width = frame.shape[:2]
        theme = self.THEMES["angry"]
        pool = self._global_pool(pack)
        assets = self._cycle_assets(
            pool,
            theme["global_count"],
            frame_index,
            shift_frames=22,
        )

        border_pulse = (
            0.45
            + 0.20
            * math.sin(frame_index * 0.11)
        )
        overlay = frame.copy()
        thickness = max(
            4,
            int(min(width, height) * 0.010),
        )
        cv2.rectangle(
            overlay,
            (thickness, thickness),
            (
                width - thickness,
                height - thickness,
            ),
            (20, 25, 235),
            thickness,
            cv2.LINE_AA,
        )
        cv2.addWeighted(
            overlay,
            border_pulse * 0.16 * opacity,
            frame,
            1.0
            - border_pulse * 0.16 * opacity,
            0.0,
            dst=frame,
        )

        for index, asset in enumerate(assets):
            seed_x = self._angry_seed(
                index,
                0,
            )
            seed_y = self._angry_seed(
                index,
                1,
            )
            seed_speed = self._angry_seed(
                index,
                2,
            )
            seed_phase = self._angry_seed(
                index,
                3,
            )
            seed_curve = self._angry_seed(
                index,
                4,
            )

            side = -1 if seed_x < 0.5 else 1
            motion_mode = index % 4

            if motion_mode in {0, 1}:
                # Fast shallow arc entering from either side.
                travel_distance = (
                    width * (
                        0.34
                        + 0.17 * seed_x
                    )
                    + 150.0
                )
                speed = (
                    5.2
                    + 2.9 * seed_speed
                )
                travelled = (
                    frame_index * speed
                    + seed_phase * travel_distance
                ) % travel_distance
                progress = travelled / travel_distance

                edge_x = (
                    -78.0
                    if side < 0
                    else width + 78.0
                )
                px = (
                    edge_x + travelled
                    if side < 0
                    else edge_x - travelled
                )

                base_y = height * (
                    0.07
                    + 0.86 * seed_y
                )
                arc_sign = (
                    -1.0
                    if seed_curve < 0.5
                    else 1.0
                )
                py = (
                    base_y
                    + arc_sign
                    * (
                        18.0
                        + 62.0 * seed_curve
                    )
                    * math.sin(
                        math.pi * progress
                    )
                    + (
                        6.0
                        + 13.0 * seed_speed
                    )
                    * math.sin(
                        progress
                        * math.tau
                        * (
                            1.45
                            + 0.75 * seed_x
                        )
                        + seed_phase * math.tau
                    )
                )

                fade = (
                    math.sin(
                        math.pi * progress
                    )
                    ** 0.42
                )

            elif motion_mode == 2:
                # Diagonal free-flight crossing a large part of the frame.
                travel_distance = (
                    width + 180.0
                )
                speed = (
                    4.6
                    + 2.5 * seed_speed
                )
                travelled = (
                    frame_index * speed
                    + seed_phase * travel_distance
                ) % travel_distance
                progress = travelled / travel_distance

                if side < 0:
                    px = -90.0 + travelled
                else:
                    px = width + 90.0 - travelled

                start_y = height * (
                    0.12
                    + 0.70 * seed_y
                )
                diagonal_shift = height * (
                    -0.22
                    + 0.44 * seed_curve
                )

                py = (
                    start_y
                    + diagonal_shift * progress
                    + (
                        18.0
                        + 32.0 * seed_speed
                    )
                    * math.sin(
                        math.pi * progress
                        + seed_phase * math.tau
                    )
                )

                fade = (
                    math.sin(
                        math.pi * progress
                    )
                    ** 0.46
                )

            else:
                # A compact airborne loop anchored in an outer screen zone.
                cycle_frames = (
                    118.0
                    + 54.0 * seed_speed
                )
                progress = (
                    (
                        frame_index
                        + seed_phase * cycle_frames
                    )
                    % cycle_frames
                ) / cycle_frames

                center_x = width * (
                    0.10
                    + 0.28 * seed_x
                    if side < 0
                    else 0.90
                    - 0.28 * seed_x
                )
                center_y = height * (
                    0.10
                    + 0.78 * seed_y
                )
                angle_value = (
                    progress * math.tau
                    * (
                        0.75
                        + 0.35 * seed_speed
                    )
                    * side
                    + seed_phase * math.tau
                )

                px = (
                    center_x
                    + (
                        28.0
                        + 55.0 * seed_curve
                    )
                    * math.cos(angle_value)
                    + 12.0
                    * math.sin(
                        frame_index
                        * (
                            0.045
                            + 0.018 * seed_x
                        )
                        + index
                    )
                )
                py = (
                    center_y
                    + (
                        20.0
                        + 38.0 * seed_speed
                    )
                    * math.sin(angle_value)
                    + 7.0
                    * math.sin(
                        frame_index
                        * (
                            0.067
                            + 0.015 * seed_curve
                        )
                        + seed_y * math.tau
                    )
                )

                # Loops remain visible, but breathe slightly.
                fade = (
                    0.72
                    + 0.28
                    * math.sin(
                        math.pi * progress
                    )
                )

            px, py = self._avoid_face(
                px,
                py,
                face_box,
                width,
                height,
            )

            base = width * (
                0.039
                if asset.role == "fragment"
                else 0.062
            )
            particle_width = int(
                base
                * (
                    0.84
                    + 0.28 * seed_speed
                )
            )

            angle = (
                (-9.0 if side < 0 else 9.0)
                + (
                    5.0
                    + 6.0 * seed_curve
                )
                * math.sin(
                    frame_index
                    * (
                        0.062
                        + 0.026 * seed_x
                    )
                    + seed_phase * math.tau
                )
            )

            self._draw_particle(
                frame,
                "angry",
                asset,
                px,
                py,
                particle_width,
                angle=angle,
                opacity=(
                    theme["global_opacity"]
                    * opacity
                    * fade
                ),
                flip=side > 0,
            )

    def _render_disgust_background(
        self,
        frame: np.ndarray,
        pack: StickerPack,
        face_box: FaceBox,
        frame_index: int,
        opacity: float,
    ) -> None:
        height, width = frame.shape[:2]
        theme = self.THEMES["disgust"]
        pool = self._global_pool(pack)
        assets = self._cycle_assets(
            pool,
            theme["global_count"],
            frame_index,
            shift_frames=26,
        )

        for index, asset in enumerate(assets):
            seed_x = (index * 0.173 + 0.07) % 1.0
            speed = 1.25 + (index % 5) * 0.16
            py = height + 70.0 - (
                frame_index * speed + index * 83.0
            ) % (height + 150.0)
            px = (
                width * seed_x
                + 30.0 * math.sin(frame_index * 0.025 + index * 1.4)
            )
            px, py = self._avoid_face(
                px, py, face_box, width, height
            )
            base = width * (0.035 if asset.role == "fragment" else 0.070)
            self._draw_particle(
                frame,
                "disgust",
                asset,
                px,
                py,
                int(base * (0.90 + 0.17 * math.sin(index))),
                angle=8.0 * math.sin(frame_index * 0.032 + index),
                opacity=theme["global_opacity"] * opacity,
                flip=index % 3 == 0,
            )

    def _render_fear_background(
        self,
        frame: np.ndarray,
        pack: StickerPack,
        face_box: FaceBox,
        frame_index: int,
        opacity: float,
    ) -> None:
        height, width = frame.shape[:2]
        theme = self.THEMES["fear"]
        pool = self._global_pool(pack)
        assets = self._cycle_assets(
            pool,
            theme["global_count"],
            frame_index,
            shift_frames=24,
        )

        for index, asset in enumerate(assets):
            side_bias = 0.12 if index % 2 == 0 else 0.88
            spread = ((index * 0.097) % 0.24) - 0.12
            px = width * (side_bias + spread)
            py = height * (0.08 + (index * 0.137) % 0.84)
            tremor_x = 5.0 * math.sin(frame_index * 0.43 + index * 2.1)
            tremor_y = 4.0 * math.sin(frame_index * 0.37 + index * 1.3)
            drift = 14.0 * math.sin(frame_index * 0.025 + index)
            px += tremor_x + drift
            py += tremor_y
            px, py = self._avoid_face(
                px, py, face_box, width, height
            )
            base = width * (0.036 if asset.role == "fragment" else 0.063)
            self._draw_particle(
                frame,
                "fear",
                asset,
                px,
                py,
                int(base),
                angle=6.0 * math.sin(frame_index * 0.15 + index),
                opacity=theme["global_opacity"] * opacity,
            )

    def _render_happy_background(
        self,
        frame: np.ndarray,
        pack: StickerPack,
        face_box: FaceBox,
        frame_index: int,
        opacity: float,
    ) -> None:
        height, width = frame.shape[:2]
        theme = self.THEMES["happy"]
        pool = self._global_pool(pack)
        assets = self._cycle_assets(
            pool,
            theme["global_count"],
            frame_index,
            shift_frames=34,
        )

        for index, asset in enumerate(assets):
            progress = (
                frame_index * (0.9 + (index % 3) * 0.15)
                + index * 137.0
            ) % (width + 180.0)
            px = -90.0 + progress
            py = height * (0.10 + (index * 0.157) % 0.76)
            py += 20.0 * math.sin(frame_index * 0.03 + index)
            px, py = self._avoid_face(
                px, py, face_box, width, height
            )
            base = width * (0.032 if asset.role == "fragment" else 0.056)
            self._draw_particle(
                frame,
                "happy",
                asset,
                px,
                py,
                int(base),
                angle=(frame_index * 0.35 + index * 17.0) % 18.0 - 9.0,
                opacity=theme["global_opacity"] * opacity,
            )

    def _render_neutral_background(
        self,
        frame: np.ndarray,
        pack: StickerPack,
        face_box: FaceBox,
        frame_index: int,
        opacity: float,
    ) -> None:
        height, width = frame.shape[:2]
        theme = self.THEMES["neutral"]
        pool = self._global_pool(pack)
        assets = self._cycle_assets(
            pool,
            theme["global_count"],
            frame_index,
            shift_frames=45,
        )
        center_x = width * 0.5
        center_y = height * 0.5

        for index, asset in enumerate(assets):
            angle = (
                frame_index * 0.0025
                + index * (2.0 * math.pi / len(assets))
            )
            radius_x = width * (0.38 + 0.035 * math.sin(index))
            radius_y = height * (0.34 + 0.04 * math.cos(index * 1.7))
            px = center_x + radius_x * math.cos(angle)
            py = center_y + radius_y * math.sin(angle)
            px, py = self._avoid_face(
                px, py, face_box, width, height
            )
            base = width * (0.026 if asset.role == "fragment" else 0.050)
            breathe = 0.90 + 0.10 * math.sin(frame_index * 0.035 + index)
            self._draw_particle(
                frame,
                "neutral",
                asset,
                px,
                py,
                int(base * breathe),
                angle=3.0 * math.sin(frame_index * 0.02 + index),
                opacity=theme["global_opacity"] * opacity,
            )

    def _render_sad_background(
        self,
        frame: np.ndarray,
        pack: StickerPack,
        face_box: FaceBox,
        frame_index: int,
        opacity: float,
    ) -> None:
        height, width = frame.shape[:2]
        theme = self.THEMES["sad"]
        pool = self._global_pool(pack)
        assets = self._cycle_assets(
            pool,
            theme["global_count"],
            frame_index,
            shift_frames=22,
        )

        for index, asset in enumerate(assets):
            px = width * (0.03 + (index * 0.137) % 0.94)
            speed = 3.2 + (index % 5) * 0.42
            py = -70.0 + (
                frame_index * speed + index * 89.0
            ) % (height + 150.0)
            px += 8.0 * math.sin(frame_index * 0.025 + index)
            px, py = self._avoid_face(
                px, py, face_box, width, height
            )
            base = width * (0.027 if asset.role == "fragment" else 0.055)
            self._draw_particle(
                frame,
                "sad",
                asset,
                px,
                py,
                int(base),
                angle=3.0 * math.sin(frame_index * 0.04 + index),
                opacity=theme["global_opacity"] * opacity,
            )

        bottom_assets = self._cycle_assets(
            pack.cores,
            min(5, len(pack.cores)),
            frame_index,
            shift_frames=50,
        )
        for index, asset in enumerate(bottom_assets):
            px = width * (0.10 + index * 0.20)
            py = height - 24.0 + 4.0 * math.sin(frame_index * 0.06 + index)
            px, py = self._avoid_face(
                px, py, face_box, width, height
            )
            self._draw_particle(
                frame,
                "sad",
                asset,
                px,
                py,
                int(width * 0.075),
                angle=0.0,
                opacity=0.56 * opacity,
            )

    def _render_surprise_background(
        self,
        frame: np.ndarray,
        pack: StickerPack,
        face_box: FaceBox,
        frame_index: int,
        opacity: float,
    ) -> None:
        height, width = frame.shape[:2]
        theme = self.THEMES["surprise"]
        pool = self._global_pool(pack)
        assets = self._cycle_assets(
            pool,
            theme["global_count"],
            frame_index,
            shift_frames=28,
        )
        x, y, face_width, face_height = face_box
        center_x = x + face_width * 0.5
        center_y = y + face_height * 0.46
        maximum_radius = math.hypot(width, height) * 0.62

        for index, asset in enumerate(assets):
            angle = (
                index * (2.0 * math.pi / len(assets))
                + 0.15 * math.sin(index)
            )
            radius = (
                frame_index * (4.0 + (index % 3) * 0.6)
                + index * 91.0
            ) % maximum_radius
            px = center_x + math.cos(angle) * radius
            py = center_y + math.sin(angle) * radius * 0.70
            px, py = self._avoid_face(
                px, py, face_box, width, height
            )
            base = width * (0.030 if asset.role == "fragment" else 0.055)
            fade = 1.0 - 0.35 * (radius / maximum_radius)
            self._draw_particle(
                frame,
                "surprise",
                asset,
                px,
                py,
                int(base),
                angle=math.degrees(angle) + 8.0 * math.sin(frame_index * 0.05),
                opacity=theme["global_opacity"] * fade * opacity,
            )

    def render_local(
        self,
        frame: np.ndarray,
        label: str,
        face_box: FaceBox,
        frame_index: int,
        opacity: float = 1.0,
    ) -> None:
        emotion = str(label).strip().lower()
        pack = self.packs.get(emotion)
        theme = self.THEMES.get(emotion)

        if pack is None or theme is None:
            return

        local_core_count = int(theme["local_cores"])
        local_fragment_count = int(theme["local_fragments"])

        if local_core_count <= 0 and local_fragment_count <= 0:
            return

        _, intro = self._activate(emotion)
        opacity = (
            float(np.clip(opacity, 0.0, 1.0))
            * intro
            * float(theme["local_opacity"])
        )

        if opacity <= 0.01:
            return

        x, y, width, height = face_box
        center_x = x + width * 0.5
        center_y = y + height * 0.50

        pulse = 1.0 + 0.025 * math.sin(frame_index * 0.075)
        cloud_width = int(
            np.clip(width * 0.76 * pulse * intro, 130, 285)
        )
        cloud = self._get_sprite(
            emotion,
            pack.main,
            cloud_width,
            angle=1.5 * math.sin(frame_index * 0.028),
        )
        cloud_center_y = max(
            cloud.shape[0] * 0.52 + 7.0,
            y - cloud.shape[0] * 0.34,
        )
        cloud_center_y += 4.0 * math.sin(frame_index * 0.055)
        self._overlay(
            frame,
            cloud,
            center_x,
            cloud_center_y,
            opacity,
        )

        cores = self._cycle_assets(
            pack.cores,
            local_core_count,
            frame_index,
            shift_frames=34,
        )
        fragments = self._cycle_assets(
            pack.fragments,
            local_fragment_count,
            frame_index,
            shift_frames=24,
        )

        if emotion == "sad":
            core_radius_x = width * 0.82
            core_radius_y = height * 0.65
        elif emotion == "angry":
            core_radius_x = width * 0.78
            core_radius_y = height * 0.60
        else:
            core_radius_x = width * 0.74
            core_radius_y = height * 0.58

        for index, asset in enumerate(cores):
            angle = (
                -0.18 * math.pi
                + index * (
                    1.36 * math.pi
                    / max(1, len(cores) - 1)
                )
            )
            if len(cores) == 1:
                angle = 0.0
            orbit = 0.035 * math.sin(frame_index * 0.045 + index)
            angle += orbit
            px = center_x + math.cos(angle) * core_radius_x
            py = center_y + math.sin(angle) * core_radius_y
            size_ratio = 0.16 + 0.025 * ((index + 1) % 3)
            local_pulse = 0.94 + 0.08 * math.sin(frame_index * 0.09 + index)
            self._draw_particle(
                frame,
                emotion,
                asset,
                px,
                py,
                int(width * size_ratio * local_pulse),
                angle=(
                    6.0 * math.sin(frame_index * 0.045 + index)
                ),
                opacity=opacity,
                flip=index % 2 == 1,
            )

        for index, asset in enumerate(fragments):
            angle = (
                index * (2.0 * math.pi / len(fragments))
                + frame_index * 0.004
            )
            radius_x = width * (0.95 + 0.09 * math.sin(index * 1.7))
            radius_y = height * (0.76 + 0.08 * math.cos(index))
            px = center_x + math.cos(angle) * radius_x
            py = center_y + math.sin(angle) * radius_y
            drift = 8.0 * math.sin(frame_index * 0.075 + index * 1.3)
            py += drift
            self._draw_particle(
                frame,
                emotion,
                asset,
                px,
                py,
                int(width * (0.070 + 0.010 * (index % 3))),
                angle=(frame_index * 0.22 + index * 11.0) % 18.0 - 9.0,
                opacity=opacity * 0.82,
            )
