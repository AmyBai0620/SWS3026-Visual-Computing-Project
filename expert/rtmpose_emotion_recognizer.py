from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import joblib
import numpy as np
import torch

import mmpose
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

if __package__:
    from .prediction_types import EmotionPrediction
else:
    from prediction_types import EmotionPrediction


BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent

DEFAULT_MODEL_DIR = (
    BASE / "models" / "rtmpose_expression"
)
DEFAULT_CLASSIFIER_PATH = (
    DEFAULT_MODEL_DIR / "classifier.joblib"
)
DEFAULT_SCALER_PATH = (
    DEFAULT_MODEL_DIR / "scaler.joblib"
)
DEFAULT_CHECKPOINT_PATH = (
    BASE
    / "models"
    / "rtmpose_face"
    / (
        "rtmpose-m_simcc-face6_pt-in1k_120e-"
        "256x256-72a37400_20230529.pth"
    )
)

CONFIG_NAME = (
    "rtmpose-m_8xb256-120e_face6-256x256.py"
)

EXPECTED_KEYPOINTS = 106
EXPECTED_FEATURES = 318
MODEL_INPUT_SIZE = (256, 256)
EXPECTED_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


if torch.cuda.is_available():
    # Fixed-size RTMPose inference benefits from cuDNN autotuning.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@dataclass(frozen=True)
class ModelInputTransform:
    """Map landmarks between the fixed model image and the source face ROI."""

    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int

    def to_source(self, keypoints: np.ndarray) -> np.ndarray:
        mapped = np.asarray(keypoints, dtype=np.float32).copy()
        if mapped.ndim != 2 or mapped.shape[1] != 2:
            raise ValueError(
                "Expected keypoints with shape (N, 2), "
                f"got {mapped.shape}."
            )

        scale_x = self.source_width / max(1, self.resized_width)
        scale_y = self.source_height / max(1, self.resized_height)
        mapped[:, 0] = (mapped[:, 0] - self.pad_left) * scale_x
        mapped[:, 1] = (mapped[:, 1] - self.pad_top) * scale_y
        return mapped


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class RTMPoseEmotionRecognizer:
    """
    Final realtime expression recognizer.

    BGR face crop
    -> RTMPose-Face 106 landmarks + confidence
    -> 318-dimensional normalized feature vector
    -> StandardScaler
    -> LogisticRegression
    """

    def __init__(
        self,
        device: str = "cuda",
        classifier_path: Path = DEFAULT_CLASSIFIER_PATH,
        scaler_path: Path = DEFAULT_SCALER_PATH,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        warmup_runs: int = 3,
    ) -> None:
        self.classifier_path = Path(classifier_path)
        self.scaler_path = Path(scaler_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = self._resolve_device(device)

        for required_path in (
            self.classifier_path,
            self.scaler_path,
            self.checkpoint_path,
        ):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {required_path}"
                )

        self.classifier = joblib.load(
            self.classifier_path
        )
        self.scaler = joblib.load(
            self.scaler_path
        )

        self.classes = [
            str(value)
            for value in self.classifier.classes_
        ]

        if self.classes != EXPECTED_CLASSES:
            raise ValueError(
                "Unexpected classifier class order: "
                f"{self.classes}"
            )

        if (
            int(self.classifier.n_features_in_)
            != EXPECTED_FEATURES
        ):
            raise ValueError(
                "Classifier must expect 318 features."
            )

        if (
            int(self.scaler.n_features_in_)
            != EXPECTED_FEATURES
        ):
            raise ValueError(
                "Scaler must expect 318 features."
            )

        register_all_modules()

        config_root = Path(
            mmpose.__file__
        ).resolve().parent

        self.config_path = next(
            config_root.rglob(CONFIG_NAME),
            None,
        )

        if self.config_path is None:
            raise FileNotFoundError(
                "Could not locate installed MMPose "
                f"config: {CONFIG_NAME}"
            )

        self.pose_model = init_model(
            str(self.config_path),
            str(self.checkpoint_path),
            device=self.device,
        )

        self.last_keypoints: Optional[
            np.ndarray
        ] = None
        self.last_scores: Optional[
            np.ndarray
        ] = None
        self.last_probabilities: Optional[
            np.ndarray
        ] = None
        self.last_pose_ms: Optional[float] = None
        self.last_classifier_ms: Optional[
            float
        ] = None
        self.last_prediction_ms: Optional[
            float
        ] = None
        self.last_input_prepare_ms: Optional[float] = None

        self._warm_up(max(0, int(warmup_runs)))

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = str(device).strip().lower()

        if normalized == "auto":
            return (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if normalized not in {"cpu", "cuda"}:
            raise ValueError(
                "device must be cpu, cuda, or auto."
            )

        if (
            normalized == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA was requested, but "
                "torch.cuda.is_available() is False."
            )

        return normalized

    def _sync(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _warm_up(self, runs: int) -> None:
        if runs <= 0:
            return

        image = np.full(
            (256, 256, 3),
            127,
            dtype=np.uint8,
        )

        for _ in range(runs):
            # Warm the same complete path that is measured at runtime, not
            # only the pose network.  This avoids letting the first scaler or
            # logistic-regression call distort the live rolling average.
            prediction = self.predict(image)
            if not prediction.success:
                raise RuntimeError(
                    "Expression-pipeline warm-up failed: "
                    f"{prediction.error or 'unknown error'}"
                )

        self.last_keypoints = None
        self.last_scores = None
        self.last_probabilities = None
        self.last_pose_ms = None
        self.last_classifier_ms = None
        self.last_prediction_ms = None
        self.last_input_prepare_ms = None

    @staticmethod
    def _prepare_model_input(
        face_bgr: np.ndarray,
    ) -> tuple[np.ndarray, ModelInputTransform]:
        """Letterbox a face ROI to 256x256 without distorting facial shape.

        The previous fast path resized every rectangular face crop directly to
        a square.  That was quick, but it stretched moderate-yaw faces and the
        returned landmarks were left in the 256x256 coordinate system.  This
        version preserves aspect ratio, pads the short side, and returns the
        exact inverse transform so landmarks can be mapped back to the original
        face ROI.  The operation remains inside the measured prediction call.
        """
        target_w, target_h = MODEL_INPUT_SIZE
        source_h, source_w = face_bgr.shape[:2]

        if source_h < 2 or source_w < 2:
            raise ValueError("Face crop is too small for preprocessing.")

        scale = min(
            target_w / float(source_w),
            target_h / float(source_h),
        )
        resized_w = max(1, min(target_w, int(round(source_w * scale))))
        resized_h = max(1, min(target_h, int(round(source_h * scale))))

        if (resized_w, resized_h) == (source_w, source_h):
            resized = np.ascontiguousarray(face_bgr)
        else:
            interpolation = (
                cv2.INTER_AREA
                if scale < 1.0
                else cv2.INTER_LINEAR
            )
            resized = cv2.resize(
                face_bgr,
                (resized_w, resized_h),
                interpolation=interpolation,
            )

        pad_x = target_w - resized_w
        pad_y = target_h - resized_h
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top

        if pad_x or pad_y:
            model_input = cv2.copyMakeBorder(
                resized,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_REPLICATE,
            )
        else:
            model_input = resized

        transform = ModelInputTransform(
            source_width=source_w,
            source_height=source_h,
            resized_width=resized_w,
            resized_height=resized_h,
            pad_left=pad_left,
            pad_top=pad_top,
        )

        return np.ascontiguousarray(model_input), transform

    def _run_pose(
        self,
        face_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = face_bgr.shape[:2]

        if height < 2 or width < 2:
            raise ValueError(
                "Face crop is too small for inference."
            )

        bbox = np.array(
            [[
                0.0,
                0.0,
                float(width - 1),
                float(height - 1),
            ]],
            dtype=np.float32,
        )

        self._sync()
        started = time.perf_counter()

        with torch.inference_mode():
            results = inference_topdown(
                self.pose_model,
                face_bgr,
                bboxes=bbox,
                bbox_format="xyxy",
            )

        self._sync()
        self.last_pose_ms = (
            time.perf_counter() - started
        ) * 1000.0

        if not results:
            raise RuntimeError(
                "RTMPose returned no pose result."
            )

        pred_instances = (
            results[0].pred_instances
        )

        keypoints = _to_numpy(
            pred_instances.keypoints
        )

        if keypoints.ndim == 3:
            keypoints = keypoints[0]

        if hasattr(
            pred_instances,
            "keypoint_scores",
        ):
            scores = _to_numpy(
                pred_instances.keypoint_scores
            )

            if scores.ndim == 2:
                scores = scores[0]
        else:
            scores = np.ones(
                keypoints.shape[0],
                dtype=np.float32,
            )

        keypoints = keypoints.astype(
            np.float32
        )
        scores = scores.astype(
            np.float32
        )

        if keypoints.shape != (
            EXPECTED_KEYPOINTS,
            2,
        ):
            raise RuntimeError(
                "Unexpected RTMPose keypoint shape: "
                f"{keypoints.shape}"
            )

        if scores.shape != (
            EXPECTED_KEYPOINTS,
        ):
            raise RuntimeError(
                "Unexpected RTMPose score shape: "
                f"{scores.shape}"
            )

        if (
            not np.isfinite(keypoints).all()
            or not np.isfinite(scores).all()
        ):
            raise RuntimeError(
                "RTMPose output contains "
                "non-finite values."
            )

        return keypoints, scores

    @staticmethod
    def build_feature_vector(
        keypoints: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        centered = (
            keypoints
            - keypoints.mean(
                axis=0,
                keepdims=True,
            )
        )

        scale = np.sqrt(
            (centered * centered).sum(axis=1)
        ).mean()

        if (
            not np.isfinite(scale)
            or scale <= 1e-6
        ):
            raise ValueError(
                "Invalid landmark normalization scale."
            )

        normalized_coordinates = (
            centered / scale
        ).reshape(-1).astype(np.float32)

        feature_vector = np.concatenate(
            [
                normalized_coordinates,
                scores.astype(np.float32),
            ],
            axis=0,
        )

        if feature_vector.shape != (
            EXPECTED_FEATURES,
        ):
            raise RuntimeError(
                "Unexpected feature-vector shape: "
                f"{feature_vector.shape}"
            )

        return feature_vector

    def predict(
        self,
        face_roi: np.ndarray,
    ) -> EmotionPrediction:
        # Measure the complete public prediction call.  This is the number
        # compared with the assignment's 30 ms target; camera capture, face
        # detection and visual-effect rendering are intentionally excluded.
        prediction_started = time.perf_counter()

        self.last_keypoints = None
        self.last_scores = None
        self.last_probabilities = None
        self.last_pose_ms = None
        self.last_classifier_ms = None
        self.last_prediction_ms = None
        self.last_input_prepare_ms = None

        try:
            if (
                face_roi is None
                or face_roi.size == 0
            ):
                return EmotionPrediction(
                    label="?",
                    class_index=None,
                    confidence=None,
                    success=False,
                    error="empty_face_roi",
                )

            if face_roi.ndim == 2:
                face_bgr = cv2.cvtColor(
                    face_roi,
                    cv2.COLOR_GRAY2BGR,
                )
            elif (
                face_roi.ndim == 3
                and face_roi.shape[2] == 3
            ):
                face_bgr = face_roi
            elif (
                face_roi.ndim == 3
                and face_roi.shape[2] == 4
            ):
                face_bgr = cv2.cvtColor(
                    face_roi,
                    cv2.COLOR_BGRA2BGR,
                )
            else:
                raise ValueError(
                    "Expected grayscale, BGR, or BGRA "
                    "face crop."
                )

            prepare_started = time.perf_counter()
            model_input, input_transform = self._prepare_model_input(
                face_bgr
            )
            self.last_input_prepare_ms = (
                time.perf_counter() - prepare_started
            ) * 1000.0

            model_keypoints, scores = self._run_pose(model_input)
            keypoints = input_transform.to_source(model_keypoints)

            # Publish landmarks in the caller's face-ROI coordinate system.
            # The classifier features remain equivalent because their
            # normalization removes the uniform scale and translation.
            self.last_keypoints = keypoints
            self.last_scores = scores

            features = self.build_feature_vector(
                keypoints,
                scores,
            )

            classification_started = (
                time.perf_counter()
            )

            scaled_features = (
                self.scaler.transform(
                    features.reshape(1, -1)
                )
            )

            probabilities = (
                self.classifier.predict_proba(
                    scaled_features
                )[0]
            )

            self.last_classifier_ms = (
                time.perf_counter()
                - classification_started
            ) * 1000.0

            self.last_probabilities = (
                probabilities.astype(
                    np.float32
                )
            )

            # ``predict`` and ``predict_proba`` would perform almost the same
            # classifier work twice.  The predicted class is simply the
            # maximum-probability entry, so one ``predict_proba`` call is
            # enough and trims avoidable latency from every frame.
            class_index = int(
                np.argmax(probabilities)
            )
            label = self.classes[class_index]
            confidence = float(
                probabilities[class_index]
            )

            return EmotionPrediction(
                label=label,
                class_index=class_index,
                confidence=confidence,
                success=True,
                error=None,
            )

        except Exception as exc:
            return EmotionPrediction(
                label="?",
                class_index=None,
                confidence=None,
                success=False,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        finally:
            self.last_prediction_ms = (
                time.perf_counter()
                - prediction_started
            ) * 1000.0
