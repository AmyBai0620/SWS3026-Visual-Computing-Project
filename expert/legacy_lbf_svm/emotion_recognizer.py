from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import joblib
import numpy as np


BASE = Path(__file__).resolve().parent

DEFAULT_LBF_PATH = (
    BASE.parent / "beginner" / "lbfmodel.yaml"
)
DEFAULT_MODEL_PATH = (
    BASE / "svm_model.pkl"
)

DEFAULT_EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


@dataclass
class EmotionPrediction:
    """
    Standard output format shared by all future emotion classifiers.
    """

    label: str
    class_index: Optional[int]
    confidence: Optional[float]
    success: bool
    error: Optional[str] = None


class LBFEmotionRecognizer:
    """
    Current baseline recognizer:

    LBF 68 landmarks
    -> normalized landmark coordinates
    -> StandardScaler
    -> SVM classifier
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        lbf_path: Path = DEFAULT_LBF_PATH,
        emotions=None,
        upscale_size: int = 200,
    ):
        self.model_path = Path(model_path)
        self.lbf_path = Path(lbf_path)
        self.emotions = (
            list(emotions)
            if emotions is not None
            else DEFAULT_EMOTIONS.copy()
        )
        self.upscale_size = upscale_size

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SVM model not found: {self.model_path}"
            )

        if not self.lbf_path.exists():
            raise FileNotFoundError(
                f"LBF model not found: {self.lbf_path}"
            )

        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(str(self.lbf_path))

        bundle = joblib.load(self.model_path)

        if not isinstance(bundle, dict):
            raise TypeError(
                "The model file must contain a dictionary."
            )

        if "model" not in bundle or "scaler" not in bundle:
            raise KeyError(
                "The model bundle must contain "
                "'model' and 'scaler'."
            )

        self.classifier = bundle["model"]
        self.scaler = bundle["scaler"]

    def extract_features(
        self,
        gray_face_roi: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Extract normalized 68-point facial landmark features
        from a grayscale face crop.
        """
        if gray_face_roi is None:
            return None

        if gray_face_roi.size == 0:
            return None

        if gray_face_roi.ndim == 3:
            gray_face_roi = cv2.cvtColor(
                gray_face_roi,
                cv2.COLOR_BGR2GRAY,
            )

        upscaled = cv2.resize(
            gray_face_roi,
            (self.upscale_size, self.upscale_size),
            interpolation=cv2.INTER_CUBIC,
        )

        face_box = np.array(
            [[
                0,
                0,
                self.upscale_size,
                self.upscale_size,
            ]],
            dtype=np.int32,
        )

        try:
            ok, landmarks = self.facemark.fit(
                upscaled,
                face_box,
            )
        except cv2.error:
            return None

        if (
            not ok
            or landmarks is None
            or len(landmarks) == 0
        ):
            return None

        points = landmarks[0].reshape(-1, 2)

        centroid = points.mean(axis=0)
        centered = points - centroid

        scale = np.sqrt(
            (centered ** 2).sum(axis=1)
        ).mean()

        if scale < 1e-6:
            return None

        normalized = centered / scale

        return normalized.flatten().astype(
            np.float32
        )

    def _estimate_confidence(
        self,
        scaled_features: np.ndarray,
        class_index: int,
    ) -> Optional[float]:
        """
        Return probability when the SVM supports predict_proba.

        The old baseline may not have been trained with
        probability=True, so confidence can legitimately be None.
        """
        if not hasattr(
            self.classifier,
            "predict_proba",
        ):
            return None

        try:
            probabilities = (
                self.classifier.predict_proba(
                    scaled_features
                )[0]
            )

            classes = list(
                self.classifier.classes_
            )

            if class_index not in classes:
                return None

            position = classes.index(class_index)

            return float(probabilities[position])

        except (AttributeError, ValueError, IndexError):
            return None

    def predict(
        self,
        gray_face_roi: np.ndarray,
    ) -> EmotionPrediction:
        """
        Predict one facial-expression label from a face crop.
        """
        features = self.extract_features(
            gray_face_roi
        )

        if features is None:
            return EmotionPrediction(
                label="?",
                class_index=None,
                confidence=None,
                success=False,
                error="landmark_extraction_failed",
            )

        try:
            scaled_features = self.scaler.transform(
                features.reshape(1, -1)
            )

            class_index = int(
                self.classifier.predict(
                    scaled_features
                )[0]
            )

        except Exception as exc:
            return EmotionPrediction(
                label="?",
                class_index=None,
                confidence=None,
                success=False,
                error=(
                    f"classification_failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        if not 0 <= class_index < len(self.emotions):
            return EmotionPrediction(
                label="?",
                class_index=class_index,
                confidence=None,
                success=False,
                error=(
                    "predicted_class_index_out_of_range"
                ),
            )

        confidence = self._estimate_confidence(
            scaled_features,
            class_index,
        )

        return EmotionPrediction(
            label=self.emotions[class_index],
            class_index=class_index,
            confidence=confidence,
            success=True,
            error=None,
        )


if __name__ == "__main__":
    recognizer = LBFEmotionRecognizer()

    print("Emotion recognizer loaded successfully.")
    print(f"Model: {recognizer.model_path}")
    print(f"LBF:   {recognizer.lbf_path}")
    print(f"Classes: {recognizer.emotions}")