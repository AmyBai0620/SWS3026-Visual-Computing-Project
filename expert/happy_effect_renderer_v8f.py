from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


FaceBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class AssetInfo:
    name: str
    role: str
    runtime_file: str
    recommended_min_px: int
    recommended_max_px: int


@dataclass(frozen=True)
class BurstParticle:
    asset_name: str
    direction_x: float
    direction_y: float
    distance: float
    width_ratio: float
    rotation: float
    delay: float
    duration: float


@dataclass(frozen=True)
class Floater:
    asset_name: str
    offset_x: float
    offset_y: float
    width_ratio: float
    phase: float
    orbit_x: float
    orbit_y: float
    speed: float
    rotation_amplitude: float
    opacity_min: float
    opacity_max: float


class HappyEffectRenderer:
    """
    Happy renderer for the manifest-based multi-element sticker pack.

    Expected pack:
        manifest.json
        runtime/main_cloud.png
        runtime/core_*.png
        runtime/fragment_*.png

    The renderer automatically discovers the pack in several common layouts:

        expert/assets/effects/happy/
        expert/assets/effects/happy/happy_sticker_pack/
        expert/assets/effects/happy_sticker_pack/

    Animation structure:
        - 1 main cloud
        - 3–4 core stickers
        - 6–10 atmosphere fragments
        - intro burst
        - continuous floating loop
        - periodic mini-bursts
        - occasional low-opacity rainbow pass

    All PNGs are loaded once. Resized/rotated sprites are cached.
    """

    INTRO_DURATION = 1.18
    MINI_BURST_PERIOD = 2.35
    MINI_BURST_VISIBLE_DURATION = 0.90
    RAINBOW_PERIOD = 6.10
    RAINBOW_DURATION = 1.05

    INTRO_CORE_COUNT = 4
    INTRO_FRAGMENT_COUNT = 11
    IDLE_CORE_COUNT = 3
    IDLE_FRAGMENT_COUNT = 12

    def __init__(
        self,
        asset_dir: Optional[Path] = None,
    ):
        base = Path(__file__).resolve().parent

        self.asset_root = self._resolve_asset_root(
            base=base,
            requested=asset_dir,
        )

        self.asset_info: Dict[str, AssetInfo] = {}
        self.assets: Dict[str, np.ndarray] = {}

        self.main_asset_name = ""
        self.core_asset_names: List[str] = []
        self.fragment_asset_names: List[str] = []
        self.rainbow_asset_name: Optional[str] = None

        self._sprite_cache: Dict[
            Tuple[str, int, int],
            np.ndarray,
        ] = {}

        self._load_manifest_and_assets()

        self.active = False
        self.started_at = 0.0
        self.last_label = "?"
        self.activation_index = 0

        self.intro_core_particles: List[BurstParticle] = []
        self.intro_fragment_particles: List[BurstParticle] = []
        self.core_floaters: List[Floater] = []
        self.fragment_floaters: List[Floater] = []

        self._prepare_activation_layout()

    @staticmethod
    def _resolve_asset_root(
        base: Path,
        requested: Optional[Path],
    ) -> Path:
        candidates: List[Path] = []

        if requested is not None:
            candidates.append(Path(requested))

        candidates.extend(
            [
                base / "assets" / "effects" / "happy",
                base / "assets" / "effects" / "happy" / "happy_sticker_pack",
                base / "assets" / "effects" / "happy_sticker_pack",
            ]
        )

        expanded: List[Path] = []

        for candidate in candidates:
            expanded.append(candidate)
            expanded.append(candidate / "happy_sticker_pack")

        checked: List[str] = []

        for candidate in expanded:
            manifest_path = candidate / "manifest.json"
            checked.append(str(manifest_path))

            if manifest_path.is_file():
                return candidate

        raise FileNotFoundError(
            "Cannot find the Happy sticker pack manifest.\n"
            "Expected one of:\n"
            + "\n".join(checked)
        )

    def _load_manifest_and_assets(self) -> None:
        manifest_path = self.asset_root / "manifest.json"

        try:
            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8",
                )
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot read Happy manifest: {manifest_path}"
            ) from exc

        records = manifest.get("assets")

        if not isinstance(records, list):
            raise ValueError(
                "Happy manifest must contain an 'assets' list."
            )

        for record in records:
            if not isinstance(record, dict):
                continue

            source_file = str(
                record.get("file", "")
            ).strip()

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

            recommended = record.get(
                "recommended_display_px"
            )

            if recommended is None:
                recommended = {
                    "main": [120, 360],
                    "core": [28, 190],
                    "fragment": [10, 100],
                }.get(role, [10, 100])

            if (
                not name
                or role not in {"main", "core", "fragment"}
                or not runtime_file
            ):
                continue

            if (
                not isinstance(recommended, list)
                or len(recommended) != 2
            ):
                recommended = [10, 80]

            info = AssetInfo(
                name=name,
                role=role,
                runtime_file=runtime_file,
                recommended_min_px=int(recommended[0]),
                recommended_max_px=int(recommended[1]),
            )

            image_path = self.asset_root / runtime_file

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_UNCHANGED,
            )

            if image is None:
                raise FileNotFoundError(
                    f"Happy asset is missing: {image_path}"
                )

            if image.ndim != 3 or image.shape[2] != 4:
                raise RuntimeError(
                    "Happy assets must be transparent RGBA PNGs: "
                    f"{image_path}"
                )

            # Runtime canvases are intentionally uniform. Trim transparent
            # margins in memory so display sizes refer to visible content.
            image = self._trim_transparent_rgba(
                image,
                padding=3,
            )

            self.asset_info[name] = info
            self.assets[name] = image

            if role == "main":
                if not self.main_asset_name:
                    self.main_asset_name = name
            elif role == "core":
                self.core_asset_names.append(name)
            elif role == "fragment":
                self.fragment_asset_names.append(name)

        if not self.main_asset_name:
            raise ValueError(
                "Happy manifest does not contain a main sticker."
            )

        if len(self.core_asset_names) < 3:
            raise ValueError(
                "Happy manifest needs at least 3 core stickers."
            )

        if len(self.fragment_asset_names) < 6:
            raise ValueError(
                "Happy manifest needs at least 6 fragments."
            )

        self.rainbow_asset_name = next(
            (
                name
                for name in self.core_asset_names
                if "rainbow" in name.lower()
            ),
            None,
        )

    @staticmethod
    def _trim_transparent_rgba(
        image: np.ndarray,
        padding: int,
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

    def reset(self) -> None:
        self.active = False
        self.started_at = 0.0
        self.last_label = "?"

    def update_label(
        self,
        label: str,
    ) -> None:
        normalized = str(label).lower()

        if (
            normalized == "happy"
            and self.last_label != "happy"
        ):
            self.active = True
            self.started_at = time.perf_counter()
            self.activation_index += 1
            self._prepare_activation_layout()

        if normalized != "happy":
            self.active = False

        self.last_label = normalized

    def _prepare_activation_layout(self) -> None:
        rng = random.Random(
            20260720 + self.activation_index * 7919
        )

        core_without_rainbow = [
            name
            for name in self.core_asset_names
            if name != self.rainbow_asset_name
        ]

        intro_core_names = self._sample_names(
            rng,
            core_without_rainbow,
            self.INTRO_CORE_COUNT,
        )

        intro_fragment_names = self._sample_names(
            rng,
            self.fragment_asset_names,
            self.INTRO_FRAGMENT_COUNT,
        )

        idle_core_names = self._sample_names(
            rng,
            core_without_rainbow,
            self.IDLE_CORE_COUNT,
        )

        idle_fragment_names = self._sample_names(
            rng,
            self.fragment_asset_names,
            self.IDLE_FRAGMENT_COUNT,
        )

        self.intro_core_particles = self._build_burst_particles(
            rng=rng,
            names=intro_core_names,
            radius_min=0.72,
            radius_max=1.02,
            width_ratio_min=0.17,
            width_ratio_max=0.26,
            duration_min=0.70,
            duration_max=0.92,
            start_angle_degrees=-165.0,
            end_angle_degrees=15.0,
            angle_jitter_degrees=14.0,
        )

        self.intro_fragment_particles = self._build_burst_particles(
            rng=rng,
            names=intro_fragment_names,
            radius_min=0.80,
            radius_max=1.16,
            width_ratio_min=0.065,
            width_ratio_max=0.105,
            duration_min=0.76,
            duration_max=1.02,
            start_angle_degrees=-178.0,
            end_angle_degrees=18.0,
            angle_jitter_degrees=10.0,
        )

        self.core_floaters = self._build_floaters(
            rng=rng,
            names=idle_core_names,
            layout=(
                (-0.48, -0.30),
                (0.47, -0.25),
                (0.43, 0.13),
                (-0.45, 0.16),
            ),
            width_ratio_range=(0.12, 0.17),
            orbit_range=(0.035, 0.060),
            opacity_range=(0.66, 0.94),
        )

        self.fragment_floaters = self._build_floaters(
            rng=rng,
            names=idle_fragment_names,
            layout=(
                (-0.54, -0.45),
                (-0.22, -0.51),
                (0.17, -0.50),
                (0.53, -0.43),
                (-0.57, -0.03),
                (0.56, 0.03),
                (-0.43, 0.31),
                (0.43, 0.30),
                (0.00, -0.59),
                (0.00, 0.39),
            ),
            width_ratio_range=(0.045, 0.078),
            orbit_range=(0.025, 0.050),
            opacity_range=(0.48, 0.82),
        )

    @staticmethod
    def _sample_names(
        rng: random.Random,
        pool: Sequence[str],
        count: int,
    ) -> List[str]:
        if not pool:
            return []

        if len(pool) >= count:
            return rng.sample(
                list(pool),
                count,
            )

        return [
            rng.choice(
                list(pool)
            )
            for _ in range(count)
        ]

    @staticmethod
    def _build_burst_particles(
        rng: random.Random,
        names: Sequence[str],
        radius_min: float,
        radius_max: float,
        width_ratio_min: float,
        width_ratio_max: float,
        duration_min: float,
        duration_max: float,
        start_angle_degrees: float,
        end_angle_degrees: float,
        angle_jitter_degrees: float,
    ) -> List[BurstParticle]:
        particles: List[BurstParticle] = []

        count = max(1, len(names))

        for index, name in enumerate(names):
            fraction = (
                index / max(1, count - 1)
            )

            degrees = (
                start_angle_degrees
                + (
                    end_angle_degrees
                    - start_angle_degrees
                )
                * fraction
                + rng.uniform(
                    -angle_jitter_degrees,
                    angle_jitter_degrees,
                )
            )

            radians = math.radians(degrees)

            particles.append(
                BurstParticle(
                    asset_name=name,
                    direction_x=math.cos(radians),
                    direction_y=math.sin(radians),
                    distance=rng.uniform(
                        radius_min,
                        radius_max,
                    ),
                    width_ratio=rng.uniform(
                        width_ratio_min,
                        width_ratio_max,
                    ),
                    rotation=rng.uniform(
                        -20.0,
                        20.0,
                    ),
                    delay=0.025 * index,
                    duration=rng.uniform(
                        duration_min,
                        duration_max,
                    ),
                )
            )

        rng.shuffle(particles)
        return particles

    @staticmethod
    def _build_floaters(
        rng: random.Random,
        names: Sequence[str],
        layout: Sequence[Tuple[float, float]],
        width_ratio_range: Tuple[float, float],
        orbit_range: Tuple[float, float],
        opacity_range: Tuple[float, float],
    ) -> List[Floater]:
        positions = list(layout)
        rng.shuffle(positions)

        floaters: List[Floater] = []

        for index, name in enumerate(names):
            offset_x, offset_y = positions[
                index % len(positions)
            ]

            floaters.append(
                Floater(
                    asset_name=name,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    width_ratio=rng.uniform(
                        *width_ratio_range
                    ),
                    phase=rng.uniform(
                        0.0,
                        math.tau,
                    ),
                    orbit_x=rng.uniform(
                        *orbit_range
                    ),
                    orbit_y=rng.uniform(
                        *orbit_range
                    ),
                    speed=rng.uniform(
                        0.72,
                        1.18,
                    ),
                    rotation_amplitude=rng.uniform(
                        3.0,
                        9.0,
                    ),
                    opacity_min=opacity_range[0],
                    opacity_max=opacity_range[1],
                )
            )

        return floaters

    @staticmethod
    def _smoothstep(
        value: float,
    ) -> float:
        value = max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

        return (
            value
            * value
            * (
                3.0
                - 2.0 * value
            )
        )

    @staticmethod
    def _ease_out_back(
        value: float,
    ) -> float:
        value = max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

        c1 = 1.70158
        c3 = c1 + 1.0

        return (
            1.0
            + c3
            * (
                value - 1.0
            ) ** 3
            + c1
            * (
                value - 1.0
            ) ** 2
        )

    @staticmethod
    def _transform_rgba(
        image: np.ndarray,
        width: int,
        angle: float,
    ) -> np.ndarray:
        width = max(
            1,
            int(width),
        )

        source_height, source_width = image.shape[:2]

        height = max(
            1,
            int(
                round(
                    source_height
                    * width
                    / source_width
                )
            ),
        )

        interpolation = (
            cv2.INTER_AREA
            if width < source_width
            else cv2.INTER_LINEAR
        )

        resized = cv2.resize(
            image,
            (
                width,
                height,
            ),
            interpolation=interpolation,
        )

        if abs(angle) < 0.1:
            return resized

        center = (
            resized.shape[1] / 2.0,
            resized.shape[0] / 2.0,
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0,
        )

        cosine = abs(
            matrix[0, 0]
        )
        sine = abs(
            matrix[0, 1]
        )

        bound_width = int(
            resized.shape[0] * sine
            + resized.shape[1] * cosine
        )

        bound_height = int(
            resized.shape[0] * cosine
            + resized.shape[1] * sine
        )

        matrix[0, 2] += (
            bound_width / 2.0
            - center[0]
        )
        matrix[1, 2] += (
            bound_height / 2.0
            - center[1]
        )

        return cv2.warpAffine(
            resized,
            matrix,
            (
                bound_width,
                bound_height,
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(
                0,
                0,
                0,
                0,
            ),
        )

    def _get_sprite(
        self,
        asset_name: str,
        width: int,
        angle: float = 0.0,
    ) -> np.ndarray:
        width = max(
            1,
            int(width),
        )

        quantized_angle = int(
            round(
                angle / 3.0
            )
            * 3
        )

        key = (
            asset_name,
            width,
            quantized_angle,
        )

        cached = self._sprite_cache.get(
            key
        )

        if cached is not None:
            return cached

        sprite = self._transform_rgba(
            self.assets[asset_name],
            width,
            float(quantized_angle),
        )

        if len(self._sprite_cache) > 600:
            self._sprite_cache.clear()

        self._sprite_cache[key] = sprite
        return sprite

    @staticmethod
    def _overlay_rgba(
        frame: np.ndarray,
        overlay: np.ndarray,
        center_x: float,
        center_y: float,
        opacity: float,
    ) -> None:
        if overlay is None or overlay.size == 0:
            return

        opacity = max(
            0.0,
            min(
                1.0,
                float(opacity),
            ),
        )

        if opacity <= 0.0:
            return

        overlay_height, overlay_width = overlay.shape[:2]
        frame_height, frame_width = frame.shape[:2]

        x1 = int(
            round(
                center_x
                - overlay_width / 2.0
            )
        )
        y1 = int(
            round(
                center_y
                - overlay_height / 2.0
            )
        )
        x2 = x1 + overlay_width
        y2 = y1 + overlay_height

        clipped_x1 = max(0, x1)
        clipped_y1 = max(0, y1)
        clipped_x2 = min(frame_width, x2)
        clipped_y2 = min(frame_height, y2)

        if (
            clipped_x1 >= clipped_x2
            or clipped_y1 >= clipped_y2
        ):
            return

        overlay_x1 = clipped_x1 - x1
        overlay_y1 = clipped_y1 - y1
        overlay_x2 = (
            overlay_x1
            + clipped_x2
            - clipped_x1
        )
        overlay_y2 = (
            overlay_y1
            + clipped_y2
            - clipped_y1
        )

        crop = overlay[
            overlay_y1:overlay_y2,
            overlay_x1:overlay_x2,
        ].astype(np.float32)

        target = frame[
            clipped_y1:clipped_y2,
            clipped_x1:clipped_x2,
        ].astype(np.float32)

        alpha = (
            crop[:, :, 3:4] / 255.0
        ) * opacity

        frame[
            clipped_y1:clipped_y2,
            clipped_x1:clipped_x2,
        ] = np.clip(
            crop[:, :, :3] * alpha
            + target * (
                1.0 - alpha
            ),
            0,
            255,
        ).astype(np.uint8)

    def _draw_cloud(
        self,
        frame: np.ndarray,
        face_box: FaceBox,
        elapsed: float,
        opacity: float,
    ) -> Tuple[float, float]:
        x, y, width, height = face_box

        entry_progress = min(
            1.0,
            elapsed / 0.42,
        )

        pop = self._ease_out_back(
            entry_progress
        )

        scale = (
            0.76
            + 0.24 * pop
        )

        scale *= (
            1.0
            + 0.020
            * math.sin(
                elapsed * 2.15
            )
        )

        cloud_width = int(
            max(
                150,
                min(
                    300,
                    width * 0.74 * scale,
                ),
            )
        )

        cloud = self._get_sprite(
            self.main_asset_name,
            cloud_width,
            angle=(
                1.2
                * math.sin(
                    elapsed * 1.40
                )
            ),
        )

        center_x = (
            x + width / 2.0
        )

        preferred_y = (
            y - cloud.shape[0] * 0.38
        )

        fallback_y = (
            y + height * 0.025
        )

        center_y = (
            preferred_y
            if preferred_y
            - cloud.shape[0] / 2.0
            > 4
            else fallback_y
        )

        center_y += (
            4.0
            * math.sin(
                elapsed * 1.65
            )
        )

        local_opacity = min(
            1.0,
            elapsed / 0.16,
        )

        self._overlay_rgba(
            frame,
            cloud,
            center_x,
            center_y,
            local_opacity * opacity,
        )

        return center_x, center_y

    def _draw_rainbow_pass(
        self,
        frame: np.ndarray,
        face_box: FaceBox,
        local_time: float,
        duration: float,
        opacity: float,
        intro: bool,
    ) -> None:
        if self.rainbow_asset_name is None:
            return

        if local_time < 0.0 or local_time > duration:
            return

        x, y, width, height = face_box

        progress = self._smoothstep(
            local_time / duration
        )

        rainbow_width = int(
            max(
                80,
                width
                * (
                    0.48
                    if intro
                    else 0.36
                ),
            )
        )

        rainbow = self._get_sprite(
            self.rainbow_asset_name,
            rainbow_width,
            angle=(
                -4.0
                + 8.0 * progress
            ),
        )

        start_x = (
            x - rainbow_width * 0.40
        )
        end_x = (
            x
            + width
            + rainbow_width * 0.36
        )

        center_x = (
            start_x
            + (
                end_x - start_x
            )
            * progress
        )

        preferred_y = (
            y - height
            * (
                0.40
                if intro
                else 0.32
            )
        )

        center_y = (
            preferred_y
            if preferred_y > 45
            else y + height * 0.04
        )

        fade_in = min(
            1.0,
            local_time / 0.14,
        )

        fade_out = min(
            1.0,
            (
                duration - local_time
            )
            / 0.20,
        )

        alpha = (
            min(
                fade_in,
                fade_out,
            )
            * (
                0.92
                if intro
                else 0.34
            )
            * opacity
        )

        self._overlay_rgba(
            frame,
            rainbow,
            center_x,
            center_y,
            alpha,
        )

    def _draw_burst(
        self,
        frame: np.ndarray,
        face_box: FaceBox,
        local_time: float,
        origin: Tuple[float, float],
        particles: Sequence[BurstParticle],
        opacity: float,
        distance_multiplier: float = 1.0,
    ) -> None:
        _, _, face_width, face_height = face_box
        origin_x, origin_y = origin

        for particle in particles:
            particle_time = (
                local_time
                - particle.delay
            )

            if (
                particle_time < 0.0
                or particle_time > particle.duration
            ):
                continue

            progress = self._smoothstep(
                particle_time
                / particle.duration
            )

            travel = (
                particle.distance
                * max(
                    face_width,
                    face_height,
                )
                * 0.54
                * distance_multiplier
                * progress
            )

            center_x = (
                origin_x
                + particle.direction_x
                * travel
            )

            center_y = (
                origin_y
                + particle.direction_y
                * travel
            )

            sprite_width = int(
                max(
                    12,
                    face_width
                    * particle.width_ratio,
                )
            )

            sprite = self._get_sprite(
                particle.asset_name,
                sprite_width,
                particle.rotation * progress,
            )

            fade_in = min(
                1.0,
                particle_time / 0.10,
            )

            fade_out = min(
                1.0,
                (
                    particle.duration
                    - particle_time
                )
                / 0.24,
            )

            self._overlay_rgba(
                frame,
                sprite,
                center_x,
                center_y,
                min(
                    fade_in,
                    fade_out,
                )
                * opacity,
            )

    def _draw_floaters(
        self,
        frame: np.ndarray,
        face_box: FaceBox,
        elapsed: float,
        floaters: Sequence[Floater],
        opacity: float,
    ) -> None:
        x, y, width, height = face_box

        base_x = (
            x + width / 2.0
        )
        base_y = (
            y + height / 2.0
        )

        for floater in floaters:
            time_value = (
                elapsed * floater.speed
                + floater.phase
            )

            center_x = (
                base_x
                + floater.offset_x * width
                + math.sin(
                    time_value
                )
                * floater.orbit_x
                * width
            )

            center_y = (
                base_y
                + floater.offset_y * height
                + math.cos(
                    time_value * 0.88
                )
                * floater.orbit_y
                * height
            )

            pulse = (
                0.90
                + 0.10
                * math.sin(
                    time_value * 1.75
                )
            )

            sprite_width = int(
                max(
                    10,
                    width
                    * floater.width_ratio
                    * pulse,
                )
            )

            sprite = self._get_sprite(
                floater.asset_name,
                sprite_width,
                angle=(
                    floater.rotation_amplitude
                    * math.sin(
                        time_value * 0.92
                    )
                ),
            )

            local_opacity = (
                floater.opacity_min
                + (
                    floater.opacity_max
                    - floater.opacity_min
                )
                * (
                    0.5
                    + 0.5
                    * math.sin(
                        time_value * 1.25
                    )
                )
            )

            self._overlay_rgba(
                frame,
                sprite,
                center_x,
                center_y,
                local_opacity * opacity,
            )

    def _periodic_burst_layout(
        self,
        cycle_index: int,
    ) -> Tuple[
        List[BurstParticle],
        List[BurstParticle],
    ]:
        rng = random.Random(
            9719
            + self.activation_index * 6151
            + cycle_index * 104729
        )

        core_pool = [
            name
            for name in self.core_asset_names
            if name != self.rainbow_asset_name
        ]

        core_names = self._sample_names(
            rng,
            core_pool,
            1,
        )

        fragment_names = self._sample_names(
            rng,
            self.fragment_asset_names,
            5,
        )

        core_particles = self._build_burst_particles(
            rng=rng,
            names=core_names,
            radius_min=0.50,
            radius_max=0.64,
            width_ratio_min=0.12,
            width_ratio_max=0.16,
            duration_min=0.64,
            duration_max=0.74,
            start_angle_degrees=-150.0,
            end_angle_degrees=-30.0,
            angle_jitter_degrees=30.0,
        )

        fragment_particles = self._build_burst_particles(
            rng=rng,
            names=fragment_names,
            radius_min=0.54,
            radius_max=0.78,
            width_ratio_min=0.045,
            width_ratio_max=0.070,
            duration_min=0.62,
            duration_max=0.82,
            start_angle_degrees=-175.0,
            end_angle_degrees=-5.0,
            angle_jitter_degrees=12.0,
        )

        return (
            core_particles,
            fragment_particles,
        )

    def _draw_periodic_burst(
        self,
        frame: np.ndarray,
        face_box: FaceBox,
        loop_elapsed: float,
        cloud_center: Tuple[float, float],
        opacity: float,
    ) -> None:
        cycle_index = int(
            loop_elapsed
            // self.MINI_BURST_PERIOD
        )

        local_time = (
            loop_elapsed
            % self.MINI_BURST_PERIOD
        )

        if (
            local_time
            > self.MINI_BURST_VISIBLE_DURATION
        ):
            return

        core_particles, fragment_particles = (
            self._periodic_burst_layout(
                cycle_index
            )
        )

        self._draw_burst(
            frame=frame,
            face_box=face_box,
            local_time=local_time,
            origin=cloud_center,
            particles=core_particles,
            opacity=0.76 * opacity,
            distance_multiplier=0.78,
        )

        self._draw_burst(
            frame=frame,
            face_box=face_box,
            local_time=local_time,
            origin=cloud_center,
            particles=fragment_particles,
            opacity=0.72 * opacity,
            distance_multiplier=0.82,
        )

    def render(
        self,
        frame: np.ndarray,
        label: str,
        face_box: FaceBox,
        frame_index: int,
        opacity: float = 1.0,
    ) -> None:
        # Kept for compatibility with realtime_demo_v6_happy_surprise.py.
        del frame_index

        self.update_label(
            label
        )

        if str(label).lower() != "happy":
            return

        elapsed = (
            time.perf_counter()
            - self.started_at
            if self.active
            else 999.0
        )

        # Large intro rainbow is behind all other elements.
        self._draw_rainbow_pass(
            frame=frame,
            face_box=face_box,
            local_time=elapsed,
            duration=0.98,
            opacity=opacity,
            intro=True,
        )

        cloud_center = self._draw_cloud(
            frame,
            face_box,
            elapsed,
            opacity,
        )

        if elapsed <= self.INTRO_DURATION:
            self._draw_burst(
                frame=frame,
                face_box=face_box,
                local_time=elapsed,
                origin=cloud_center,
                particles=self.intro_fragment_particles,
                opacity=0.92 * opacity,
            )

            self._draw_burst(
                frame=frame,
                face_box=face_box,
                local_time=elapsed,
                origin=cloud_center,
                particles=self.intro_core_particles,
                opacity=opacity,
            )

            return

        loop_elapsed = (
            elapsed - self.INTRO_DURATION
        )

        # Persistent 3 core stickers + 8 atmosphere fragments.
        self._draw_floaters(
            frame,
            face_box,
            elapsed,
            self.fragment_floaters,
            opacity,
        )

        self._draw_floaters(
            frame,
            face_box,
            elapsed,
            self.core_floaters,
            opacity,
        )

        # New mini combination every cycle.
        self._draw_periodic_burst(
            frame,
            face_box,
            loop_elapsed,
            cloud_center,
            opacity,
        )

        # Low-frequency, low-opacity rainbow pass.
        if loop_elapsed >= 2.90:
            rainbow_local_time = (
                loop_elapsed - 2.90
            ) % self.RAINBOW_PERIOD

            self._draw_rainbow_pass(
                frame=frame,
                face_box=face_box,
                local_time=rainbow_local_time,
                duration=self.RAINBOW_DURATION,
                opacity=opacity,
                intro=False,
            )
