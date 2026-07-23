from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import sklearn
import torch

import mmpose
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules


CONFIG_NAME = "rtmpose-m_8xb256-120e_face6-256x256.py"
CHECKPOINT_RELATIVE_PATH = Path(
    "expert/models/rtmpose_face/"
    "rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth"
)
MODEL_RELATIVE_DIR = Path("expert/models/rtmpose_expression")

EXPECTED_KEYPOINTS = 106
EXPECTED_FEATURES = 318
EXPECTED_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end smoke test: RTMPose 106 landmarks -> "
            "318 features -> scaler -> expression classifier."
        )
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cuda",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
    )
    return parser.parse_args()


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def extract_predictions(pose_results) -> tuple[np.ndarray, np.ndarray]:
    if not pose_results:
        raise RuntimeError("inference_topdown returned no result.")

    pred_instances = pose_results[0].pred_instances

    keypoints = to_numpy(pred_instances.keypoints)
    if keypoints.ndim == 3:
        keypoints = keypoints[0]

    if hasattr(pred_instances, "keypoint_scores"):
        scores = to_numpy(pred_instances.keypoint_scores)
        if scores.ndim == 2:
            scores = scores[0]
    else:
        scores = np.ones(keypoints.shape[0], dtype=np.float32)

    keypoints = keypoints.astype(np.float32)
    scores = scores.astype(np.float32)

    if keypoints.shape != (EXPECTED_KEYPOINTS, 2):
        raise RuntimeError(
            f"Expected keypoints shape {(EXPECTED_KEYPOINTS, 2)}, "
            f"received {keypoints.shape}."
        )

    if scores.shape != (EXPECTED_KEYPOINTS,):
        raise RuntimeError(
            f"Expected scores shape {(EXPECTED_KEYPOINTS,)}, "
            f"received {scores.shape}."
        )

    if not np.isfinite(keypoints).all():
        raise RuntimeError("RTMPose keypoints contain non-finite values.")

    if not np.isfinite(scores).all():
        raise RuntimeError("RTMPose scores contain non-finite values.")

    return keypoints, scores


def build_feature_vector(
    keypoints: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """
    Match the exact final-training feature construction:

    106 x/y landmarks
    -> subtract landmark mean
    -> divide by mean landmark radius
    -> flatten to 212 values
    -> append 106 confidence values
    -> 318 values
    """
    center = keypoints - keypoints.mean(
        axis=0,
        keepdims=True,
    )

    scale = np.sqrt(
        (center * center).sum(axis=1)
    ).mean()

    if not np.isfinite(scale) or scale <= 1e-6:
        raise RuntimeError(
            f"Invalid landmark normalization scale: {scale}"
        )

    normalized_coordinates = (
        center / scale
    ).reshape(-1).astype(np.float32)

    feature_vector = np.concatenate(
        [
            normalized_coordinates,
            scores.astype(np.float32),
        ],
        axis=0,
    )

    if feature_vector.shape != (EXPECTED_FEATURES,):
        raise RuntimeError(
            f"Expected feature shape {(EXPECTED_FEATURES,)}, "
            f"received {feature_vector.shape}."
        )

    if not np.isfinite(feature_vector).all():
        raise RuntimeError("Feature vector contains non-finite values.")

    return feature_vector


def main() -> int:
    args = parse_args()

    print("=" * 72)
    print("RTMPose realtime expression pipeline check")
    print("=" * 72)
    print(f"Python:          {platform.python_version()}")
    print(f"Executable:      {sys.executable}")
    print(f"NumPy:           {np.__version__}")
    print(f"OpenCV:          {cv2.__version__}")
    print(f"scikit-learn:    {sklearn.__version__}")
    print(f"PyTorch:         {torch.__version__}")
    print(f"MMPose:          {mmpose.__version__}")
    print(f"CUDA ready:      {torch.cuda.is_available()}")

    project_root = Path(__file__).resolve().parent.parent

    checkpoint_path = project_root / CHECKPOINT_RELATIVE_PATH
    model_dir = project_root / MODEL_RELATIVE_DIR
    classifier_path = model_dir / "classifier.joblib"
    scaler_path = model_dir / "scaler.joblib"

    for required_path in (
        checkpoint_path,
        classifier_path,
        scaler_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {required_path}"
            )

    classifier = joblib.load(classifier_path)
    scaler = joblib.load(scaler_path)

    classes = [str(value) for value in classifier.classes_]

    print(f"Classifier:      {type(classifier).__name__}")
    print(f"Scaler:          {type(scaler).__name__}")
    print(f"Classes:         {classes}")
    print(
        "Input dims:      "
        f"classifier={classifier.n_features_in_}, "
        f"scaler={scaler.n_features_in_}"
    )

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested, but torch.cuda.is_available() is False."
        )

    register_all_modules()

    config_root = Path(mmpose.__file__).resolve().parent
    config_path = next(
        config_root.rglob(CONFIG_NAME),
        None,
    )

    if config_path is None:
        raise FileNotFoundError(
            f"Installed MMPose config not found: {CONFIG_NAME}"
        )

    print(f"Device:          {device}")
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Config:          {config_path}")
    print()
    print("Loading RTMPose-Face...")

    model = init_model(
        str(config_path),
        str(checkpoint_path),
        device=device,
    )

    print(
        f"Model device:    "
        f"{next(model.parameters()).device}"
    )

    # Synthetic input checks the API and feature path.
    # Its predicted emotion is not a semantic accuracy test.
    image = np.full(
        (256, 256, 3),
        127,
        dtype=np.uint8,
    )
    bbox = np.array(
        [[0.0, 0.0, 255.0, 255.0]],
        dtype=np.float32,
    )

    print()
    print(f"Warming up ({args.warmup} runs)...")

    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            _ = inference_topdown(
                model,
                image,
                bboxes=bbox,
                bbox_format="xyxy",
            )
            sync(device)

    print("Running end-to-end inference...")

    sync(device)
    started = time.perf_counter()

    with torch.inference_mode():
        pose_results = inference_topdown(
            model,
            image,
            bboxes=bbox,
            bbox_format="xyxy",
        )

    sync(device)
    rtmpose_ms = (
        time.perf_counter() - started
    ) * 1000.0

    keypoints, scores = extract_predictions(
        pose_results
    )
    feature_vector = build_feature_vector(
        keypoints,
        scores,
    )

    classification_started = time.perf_counter()

    scaled_features = scaler.transform(
        feature_vector.reshape(1, -1)
    )
    predicted_label = str(
        classifier.predict(scaled_features)[0]
    )
    probabilities = classifier.predict_proba(
        scaled_features
    )[0]

    classification_ms = (
        time.perf_counter()
        - classification_started
    ) * 1000.0

    predicted_position = classes.index(
        predicted_label
    )
    confidence = float(
        probabilities[predicted_position]
    )

    checks = {
        "class order": classes == EXPECTED_CLASSES,
        "classifier feature count": (
            int(classifier.n_features_in_)
            == EXPECTED_FEATURES
        ),
        "scaler feature count": (
            int(scaler.n_features_in_)
            == EXPECTED_FEATURES
        ),
        "keypoint shape": (
            keypoints.shape
            == (EXPECTED_KEYPOINTS, 2)
        ),
        "score shape": (
            scores.shape
            == (EXPECTED_KEYPOINTS,)
        ),
        "feature shape": (
            feature_vector.shape
            == (EXPECTED_FEATURES,)
        ),
        "scaled shape": (
            scaled_features.shape
            == (1, EXPECTED_FEATURES)
        ),
        "probability count": (
            probabilities.shape
            == (len(EXPECTED_CLASSES),)
        ),
        "probability sum": bool(
            np.isclose(
                probabilities.sum(),
                1.0,
                atol=1e-6,
            )
        ),
        "known prediction": (
            predicted_label
            in EXPECTED_CLASSES
        ),
    }

    print()
    print("Pipeline result:")
    print(f"  keypoints:       {keypoints.shape}")
    print(f"  scores:          {scores.shape}")
    print(f"  feature vector:  {feature_vector.shape}")
    print(f"  scaled features: {scaled_features.shape}")
    print(f"  prediction:      {predicted_label}")
    print(f"  confidence:      {confidence:.6f}")
    print(f"  RTMPose time:    {rtmpose_ms:.2f} ms")
    print(
        f"  classifier time: "
        f"{classification_ms:.3f} ms"
    )

    print()
    print("Checks:")
    for name, passed in checks.items():
        print(
            f"  [{'PASS' if passed else 'FAIL'}] "
            f"{name}"
        )

    if not all(checks.values()):
        print()
        print(
            "[FAIL] The combined RTMPose expression "
            "pipeline is not ready."
        )
        return 1

    print()
    print(
        "[PASS] RTMPose -> 318 features -> scaler -> "
        "classifier is ready for webcam integration."
    )
    print(
        "Note: the synthetic-image label above is only "
        "an interface smoke test."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
