from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import mmpose
import numpy as np
from mmpose.apis import inference_topdown, init_model


CONFIG_NAME = "rtmpose-m_8xb256-120e_face6-256x256.py"
CHECKPOINT_RELATIVE_PATH = Path(
    "experiments/landmark_benchmark/models/rtmpose_face/"
    "rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth"
)

PATH_COLUMN_CANDIDATES = (
    "relative_path",
    "image_path",
    "file_path",
    "filepath",
    "path",
)

ID_COLUMN_CANDIDATES = (
    "sample_id",
    "image_id",
    "id",
    "name",
)


def find_existing_column(
    row: dict[str, str],
    candidates: tuple[str, ...],
) -> str | None:
    """Return the first candidate column present in a CSV row."""
    lowered = {key.lower(): key for key in row.keys()}

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


def read_first_manifest_row(manifest_path: Path) -> dict[str, str]:
    """Read the first sample from the manifest."""
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        row = next(reader, None)

    if row is None:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    return row


def resolve_image_path(
    dataset_root: Path,
    manifest_value: str,
) -> Path:
    """Resolve either an absolute or relative image path."""
    raw_path = Path(manifest_value.replace("\\", "/"))

    if raw_path.is_absolute():
        candidates = [raw_path]
    else:
        candidates = [
            dataset_root / raw_path,
            dataset_root.parent / raw_path,
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not locate the image referenced by the manifest.\n"
        f"Manifest value: {manifest_value}\n"
        f"Tried:\n{tried}"
    )


def to_numpy(value: object) -> np.ndarray:
    """Convert Torch tensors or array-like values to NumPy."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    return np.asarray(value)


def extract_predictions(
    pose_results: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one face's keypoints and confidence scores."""
    if not pose_results:
        raise RuntimeError("inference_topdown returned no pose result.")

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

    if keypoints.shape != (106, 2):
        raise RuntimeError(
            "Unexpected keypoint shape. "
            f"Expected (106, 2), received {keypoints.shape}."
        )

    if scores.shape != (106,):
        raise RuntimeError(
            "Unexpected score shape. "
            f"Expected (106,), received {scores.shape}."
        )

    return keypoints.astype(np.float32), scores.astype(np.float32)


def create_overlay(
    image_bgr: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
    draw_indices: bool,
    scale: int = 12,
) -> np.ndarray:
    """Draw enlarged landmarks for visual inspection."""
    overlay = cv2.resize(
        image_bgr,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    for index, ((x, y), score) in enumerate(zip(keypoints, scores)):
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        if float(score) < score_threshold:
            continue

        center = (
            int(round(float(x) * scale)),
            int(round(float(y) * scale)),
        )

        cv2.circle(
            overlay,
            center,
            radius=3,
            color=(0, 255, 0),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        if draw_indices:
            cv2.putText(
                overlay,
                str(index),
                (center[0] + 3, center[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RTMPose-Face on the first FER manifest sample."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="FER train directory containing the seven class folders.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "experiments/landmark_benchmark/manifests/"
            "sample_manifest_35.csv"
        ),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    # runners -> landmark_benchmark -> experiments -> project root
    project_root = Path(__file__).resolve().parents[3]

    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    manifest_path = manifest_path.resolve()

    dataset_root = args.dataset_root.resolve()
    checkpoint_path = project_root / CHECKPOINT_RELATIVE_PATH

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    config_root = Path(mmpose.__file__).resolve().parent
    config_path = next(config_root.rglob(CONFIG_NAME), None)

    if config_path is None:
        raise FileNotFoundError(
            f"Could not find installed MMPose config: {CONFIG_NAME}"
        )

    row = read_first_manifest_row(manifest_path)

    path_column = find_existing_column(
        row,
        PATH_COLUMN_CANDIDATES,
    )
    if path_column is None:
        raise KeyError(
            "Could not identify the image-path column.\n"
            f"Manifest columns: {list(row.keys())}"
        )

    id_column = find_existing_column(
        row,
        ID_COLUMN_CANDIDATES,
    )
    sample_id = (
        row[id_column]
        if id_column is not None
        else Path(row[path_column]).stem
    )

    image_path = resolve_image_path(
        dataset_root,
        row[path_column],
    )

    image_gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )
    if image_gray is None:
        raise RuntimeError(f"OpenCV failed to read: {image_path}")

    image_bgr = cv2.cvtColor(
        image_gray,
        cv2.COLOR_GRAY2BGR,
    )

    height, width = image_bgr.shape[:2]

    # Use the entire FER image as the face bounding box.
    bbox = np.array(
        [[0.0, 0.0, float(width - 1), float(height - 1)]],
        dtype=np.float32,
    )

    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Loading RTMPose-Face on CPU...")

    model = init_model(
        str(config_path),
        str(checkpoint_path),
        device="cpu",
    )

    # Warm-up calls are not included in the diagnostic timing.
    for _ in range(max(0, args.warmup)):
        inference_topdown(
            model,
            image_bgr,
            bboxes=bbox,
            bbox_format="xyxy",
        )

    start_time = time.perf_counter()

    pose_results = inference_topdown(
        model,
        image_bgr,
        bboxes=bbox,
        bbox_format="xyxy",
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    keypoints, scores = extract_predictions(pose_results)

    output_dir = (
        project_root
        / "experiments"
        / "landmark_benchmark"
        / "outputs"
        / "rtmpose_face"
        / "single_test"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    points_overlay = create_overlay(
        image_bgr,
        keypoints,
        scores,
        args.score_threshold,
        draw_indices=False,
    )
    indexed_overlay = create_overlay(
        image_bgr,
        keypoints,
        scores,
        args.score_threshold,
        draw_indices=True,
    )

    points_path = output_dir / "overlay_points.png"
    indexed_path = output_dir / "overlay_indexed.png"
    csv_path = output_dir / "keypoints.csv"
    summary_path = output_dir / "summary.json"

    if not cv2.imwrite(str(points_path), points_overlay):
        raise RuntimeError(f"Failed to save: {points_path}")

    if not cv2.imwrite(str(indexed_path), indexed_overlay):
        raise RuntimeError(f"Failed to save: {indexed_path}")

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["keypoint_index", "x", "y", "score"])

        for index, ((x, y), score) in enumerate(
            zip(keypoints, scores)
        ):
            writer.writerow(
                [
                    index,
                    float(x),
                    float(y),
                    float(score),
                ]
            )

    visible_count = int(
        np.sum(scores >= args.score_threshold)
    )

    summary = {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "image_width": int(width),
        "image_height": int(height),
        "bbox_xyxy": bbox[0].tolist(),
        "device": "cpu",
        "model": "RTMPose-M Face6 256x256",
        "num_keypoints": int(keypoints.shape[0]),
        "score_threshold": float(args.score_threshold),
        "visible_keypoints": visible_count,
        "score_min": float(np.min(scores)),
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_max": float(np.max(scores)),
        "diagnostic_inference_ms": float(elapsed_ms),
        "note": (
            "This timing is diagnostic only and is not the final "
            "35/350-sample benchmark statistic."
        ),
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("RTMPose-Face single-image test: OK")
    print(f"Sample ID: {sample_id}")
    print(f"Image: {image_path}")
    print(f"Original size: {width} x {height}")
    print(f"Keypoints returned: {keypoints.shape[0]}")
    print(
        f"Visible keypoints: {visible_count}/106 "
        f"(threshold={args.score_threshold:.2f})"
    )
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Diagnostic inference time: {elapsed_ms:.2f} ms")
    print(f"Points overlay: {points_path}")
    print(f"Indexed overlay: {indexed_path}")
    print(f"Keypoint CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()