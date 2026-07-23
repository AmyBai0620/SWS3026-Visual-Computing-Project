from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "MediaPipe is required for the robust face detector. "
        "Install it in the active environment with: "
        "python -m pip install mediapipe==0.10.21"
    ) from exc


FaceBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class FaceDetectionResult:
    box: Optional[FaceBox]
    source: str
    is_fresh: bool
    missed_frames: int


class RobustFaceDetector:
    """
    MediaPipe FaceMesh primary detector/tracker with Haar fallback.

    The detector also:
      - derives a face crop from FaceMesh landmarks;
      - smooths box position and size;
      - holds the latest valid box for a few missed frames;
      - exposes the active source for debugging.
    """

    def __init__(
        self,
        haar_path: Optional[Path] = None,
        smoothing_alpha: float = 0.42,
        hold_frames: int = 6,
        min_detection_confidence: float = 0.45,
        min_tracking_confidence: float = 0.45,
    ) -> None:
        self.smoothing_alpha = float(
            max(0.05, min(1.0, smoothing_alpha))
        )
        self.hold_frames = max(0, int(hold_frames))
        self.missed_frames = 0
        self._smoothed_box: Optional[np.ndarray] = None

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "The installed MediaPipe package does not expose "
                "mp.solutions. Recreate the Python 3.10 environment "
                "with mediapipe==0.10.21."
            )

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=float(
                min_detection_confidence
            ),
            min_tracking_confidence=float(
                min_tracking_confidence
            ),
        )

        if haar_path is None:
            haar_path = (
                Path(__file__).resolve().parent.parent
                / "beginner"
                / "haarcascade_frontalface_default.xml"
            )

        self.haar_path = Path(haar_path)
        self._haar = cv2.CascadeClassifier(
            str(self.haar_path)
        )

        if self._haar.empty():
            raise RuntimeError(
                f"Could not load Haar fallback: {self.haar_path}"
            )

    @staticmethod
    def _clip_box(
        box: FaceBox,
        frame_width: int,
        frame_height: int,
    ) -> Optional[FaceBox]:
        x, y, width, height = map(int, box)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_width, x + width)
        y2 = min(frame_height, y + height)

        if x2 - x1 < 24 or y2 - y1 < 24:
            return None

        return (
            x1,
            y1,
            x2 - x1,
            y2 - y1,
        )

    @staticmethod
    def _largest_haar_face(
        faces,
    ) -> Optional[FaceBox]:
        if faces is None or len(faces) == 0:
            return None

        normalized = [
            tuple(map(int, face))
            for face in faces
        ]

        return max(
            normalized,
            key=lambda face: face[2] * face[3],
        )

    @staticmethod
    def _mesh_to_box(
        face_landmarks,
        frame_width: int,
        frame_height: int,
    ) -> Optional[FaceBox]:
        points = np.array(
            [
                (
                    landmark.x * frame_width,
                    landmark.y * frame_height,
                )
                for landmark in face_landmarks.landmark
            ],
            dtype=np.float32,
        )

        if (
            points.shape[0] < 100
            or not np.isfinite(points).all()
        ):
            return None

        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)

        x1, y1 = min_xy
        x2, y2 = max_xy

        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)

        # FaceMesh follows the visible facial surface closely.
        # Add margins so RTMPose receives a complete face crop.
        x1 -= 0.13 * width
        x2 += 0.13 * width
        y1 -= 0.18 * height
        y2 += 0.09 * height

        width = x2 - x1
        height = y2 - y1

        # Keep the crop reasonably balanced under moderate yaw.
        target_width = max(width, height * 0.82)
        center_x = (x1 + x2) * 0.5

        x1 = center_x - target_width * 0.5
        x2 = center_x + target_width * 0.5

        return (
            int(round(x1)),
            int(round(y1)),
            int(round(x2 - x1)),
            int(round(y2 - y1)),
        )

    @staticmethod
    def _expand_haar_box(
        box: FaceBox,
    ) -> FaceBox:
        x, y, width, height = box

        pad_x = int(round(width * 0.05))
        pad_top = int(round(height * 0.10))
        pad_bottom = int(round(height * 0.04))

        return (
            x - pad_x,
            y - pad_top,
            width + pad_x * 2,
            height + pad_top + pad_bottom,
        )

    @staticmethod
    def _box_iou(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        fx, fy, fw, fh = first
        sx, sy, sw, sh = second

        first_x2 = fx + fw
        first_y2 = fy + fh
        second_x2 = sx + sw
        second_y2 = sy + sh

        inter_x1 = max(fx, sx)
        inter_y1 = max(fy, sy)
        inter_x2 = min(first_x2, second_x2)
        inter_y2 = min(first_y2, second_y2)

        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height

        union = (
            fw * fh
            + sw * sh
            - intersection
        )

        if union <= 0.0:
            return 0.0

        return float(intersection / union)

    def _smooth(
        self,
        box: FaceBox,
        frame_width: int,
        frame_height: int,
    ) -> Optional[FaceBox]:
        current = np.asarray(
            box,
            dtype=np.float32,
        )

        if self._smoothed_box is None:
            self._smoothed_box = current
        else:
            overlap = self._box_iou(
                self._smoothed_box,
                current,
            )

            # Snap faster after a real movement or detector-source change,
            # but use gentler smoothing for normal frame-to-frame jitter.
            alpha = (
                0.78
                if overlap < 0.25
                else self.smoothing_alpha
            )

            self._smoothed_box = (
                self._smoothed_box * (1.0 - alpha)
                + current * alpha
            )

        rounded = tuple(
            int(round(value))
            for value in self._smoothed_box
        )

        clipped = self._clip_box(
            rounded,
            frame_width,
            frame_height,
        )

        if clipped is not None:
            self._smoothed_box = np.asarray(
                clipped,
                dtype=np.float32,
            )

        return clipped

    def _detect_mediapipe(
        self,
        frame_bgr: np.ndarray,
    ) -> Optional[FaceBox]:
        frame_height, frame_width = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )
        rgb.flags.writeable = False

        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        return self._mesh_to_box(
            result.multi_face_landmarks[0],
            frame_width,
            frame_height,
        )

    def _detect_haar(
        self,
        frame_bgr: np.ndarray,
    ) -> Optional[FaceBox]:
        gray = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        # Mild equalisation improves fallback behaviour under uneven light.
        gray = cv2.equalizeHist(gray)

        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(60, 60),
        )

        primary = self._largest_haar_face(
            faces
        )

        if primary is None:
            return None

        return self._expand_haar_box(
            primary
        )

    def detect(
        self,
        frame_bgr: np.ndarray,
    ) -> FaceDetectionResult:
        if frame_bgr is None or frame_bgr.size == 0:
            return FaceDetectionResult(
                box=None,
                source="none",
                is_fresh=False,
                missed_frames=self.missed_frames,
            )

        frame_height, frame_width = frame_bgr.shape[:2]

        raw_box = self._detect_mediapipe(
            frame_bgr
        )
        source = "mediapipe"

        if raw_box is None:
            raw_box = self._detect_haar(
                frame_bgr
            )
            source = "haar"

        if raw_box is not None:
            clipped_raw = self._clip_box(
                raw_box,
                frame_width,
                frame_height,
            )

            if clipped_raw is not None:
                smoothed = self._smooth(
                    clipped_raw,
                    frame_width,
                    frame_height,
                )

                if smoothed is not None:
                    self.missed_frames = 0

                    return FaceDetectionResult(
                        box=smoothed,
                        source=source,
                        is_fresh=True,
                        missed_frames=0,
                    )

        self.missed_frames += 1

        if (
            self._smoothed_box is not None
            and self.missed_frames <= self.hold_frames
        ):
            held = tuple(
                int(round(value))
                for value in self._smoothed_box
            )

            held = self._clip_box(
                held,
                frame_width,
                frame_height,
            )

            if held is not None:
                return FaceDetectionResult(
                    box=held,
                    source="hold",
                    is_fresh=False,
                    missed_frames=self.missed_frames,
                )

        return FaceDetectionResult(
            box=None,
            source="none",
            is_fresh=False,
            missed_frames=self.missed_frames,
        )

    def reset(self) -> None:
        self.missed_frames = 0
        self._smoothed_box = None

    def close(self) -> None:
        self._face_mesh.close()
