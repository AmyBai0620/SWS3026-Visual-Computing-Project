from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import face_alignment
import numpy as np

EXPECTED_KEYPOINTS = 68
PATH_COLUMNS = ("relative_path", "image_path", "file_path", "filepath", "path")
ID_COLUMNS = ("sample_id", "image_id", "id", "name")


def find_column(row: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    lowered = {key.lower(): key for key in row}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def read_first_row(manifest_path: Path) -> dict[str, str]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        row = next(csv.DictReader(file), None)
    if row is None:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    return row


def resolve_image_path(dataset_root: Path, manifest_value: str) -> Path:
    raw = Path(manifest_value.replace("\\", "/"))
    candidates = [raw] if raw.is_absolute() else [dataset_root / raw, dataset_root.parent / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not locate manifest image: {manifest_value}\nTried:\n{tried}"
    )


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def extract_predictions(result: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(result, tuple) or len(result) != 3:
        raise RuntimeError(
            "Unexpected FAN result; expected (landmarks, scores, detected_faces)."
        )
    landmarks_list, scores_list, _ = result
    if landmarks_list is None or len(landmarks_list) == 0:
        raise RuntimeError("FAN returned no landmarks.")
    if scores_list is None or len(scores_list) == 0:
        raise RuntimeError("FAN returned no landmark scores.")

    keypoints = to_numpy(landmarks_list[0]).astype(np.float32)
    scores = to_numpy(scores_list[0]).astype(np.float32).reshape(-1)

    if keypoints.shape != (EXPECTED_KEYPOINTS, 2):
        raise RuntimeError(
            f"Expected ({EXPECTED_KEYPOINTS}, 2) keypoints, got {keypoints.shape}."
        )
    if scores.shape != (EXPECTED_KEYPOINTS,):
        raise RuntimeError(
            f"Expected ({EXPECTED_KEYPOINTS},) scores, got {scores.shape}."
        )
    return keypoints, scores


def draw_overlay(
    image_bgr: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    draw_indices: bool,
    scale: int = 12,
) -> np.ndarray:
    overlay = cv2.resize(
        image_bgr,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
    for index, ((x, y), score) in enumerate(zip(keypoints, scores)):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        center = (int(round(float(x) * scale)), int(round(float(y) * scale)))
        color = (0, 255, 0) if float(score) >= threshold else (0, 165, 255)
        cv2.circle(overlay, center, 3, color, -1, lineType=cv2.LINE_AA)
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
        description="Run 2D FAN-4 on the first FER manifest sample."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "experiments/landmark_benchmark/manifests/sample_manifest_35.csv"
        ),
    )
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    manifest_path = (
        args.manifest
        if args.manifest.is_absolute()
        else project_root / args.manifest
    ).resolve()
    dataset_root = args.dataset_root.resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    row = read_first_row(manifest_path)
    path_column = find_column(row, PATH_COLUMNS)
    if path_column is None:
        raise KeyError(f"Image-path column not found. Columns: {list(row)}")
    id_column = find_column(row, ID_COLUMNS)
    sample_id = row[id_column] if id_column else Path(row[path_column]).stem
    image_path = resolve_image_path(dataset_root, row[path_column])

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"OpenCV failed to read: {image_path}")
    image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    detected_faces = [
        np.array([0.0, 0.0, float(width - 1), float(height - 1)], dtype=np.float32)
    ]

    print("Loading 2D FAN-4 on CPU...")
    fan = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device="cpu",
        flip_input=False,
        compile=False,
    )

    for _ in range(max(0, args.warmup)):
        fan.get_landmarks_from_image(
            image_rgb,
            detected_faces=detected_faces,
            return_landmark_score=True,
        )

    start = time.perf_counter()
    result = fan.get_landmarks_from_image(
        image_rgb,
        detected_faces=detected_faces,
        return_landmark_score=True,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    keypoints, scores = extract_predictions(result)
    finite = np.isfinite(keypoints).all(axis=1) & np.isfinite(scores)
    in_bounds = (
        finite
        & (keypoints[:, 0] >= 0.0)
        & (keypoints[:, 0] <= float(width - 1))
        & (keypoints[:, 1] >= 0.0)
        & (keypoints[:, 1] <= float(height - 1))
    )
    visible = finite & (scores >= args.score_threshold)

    output_dir = (
        project_root
        / "experiments"
        / "landmark_benchmark"
        / "outputs"
        / "fan_face"
        / "single_test"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    points_path = output_dir / "overlay_points.png"
    indexed_path = output_dir / "overlay_indexed.png"
    csv_path = output_dir / "keypoints.csv"
    summary_path = output_dir / "summary.json"

    if not cv2.imwrite(
        str(points_path),
        draw_overlay(image_bgr, keypoints, scores, args.score_threshold, False),
    ):
        raise RuntimeError(f"Failed to save: {points_path}")
    if not cv2.imwrite(
        str(indexed_path),
        draw_overlay(image_bgr, keypoints, scores, args.score_threshold, True),
    ):
        raise RuntimeError(f"Failed to save: {indexed_path}")

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["keypoint_index", "x", "y", "score"])
        for index, ((x, y), score) in enumerate(zip(keypoints, scores)):
            writer.writerow([index, float(x), float(y), float(score)])

    summary = {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "image_width": int(width),
        "image_height": int(height),
        "bbox_xyxy": detected_faces[0].tolist(),
        "device": "cpu",
        "model": "2D FAN-4",
        "num_keypoints": int(keypoints.shape[0]),
        "finite_keypoints": int(np.sum(finite)),
        "in_bounds_keypoints": int(np.sum(in_bounds)),
        "in_bounds_rate": float(np.mean(in_bounds)),
        "score_threshold": float(args.score_threshold),
        "visible_keypoints": int(np.sum(visible)),
        "score_min": float(np.min(scores)),
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_max": float(np.max(scores)),
        "diagnostic_inference_ms": float(elapsed_ms),
        "note": (
            "FAN scores are heatmap response scores. The visible count and "
            "single-image timing are diagnostic only."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nFAN single-image test: OK")
    print(f"Sample ID: {sample_id}")
    print(f"Image: {image_path}")
    print(f"Original size: {width} x {height}")
    print(f"Keypoints returned: {keypoints.shape[0]}")
    print(f"Finite keypoints: {int(np.sum(finite))}/{EXPECTED_KEYPOINTS}")
    print(f"In-bounds keypoints: {int(np.sum(in_bounds))}/{EXPECTED_KEYPOINTS}")
    print(
        "Visible by diagnostic threshold: "
        f"{int(np.sum(visible))}/{EXPECTED_KEYPOINTS} "
        f"(threshold={args.score_threshold:.2f})"
    )
    print(f"Mean heatmap score: {np.mean(scores):.4f}")
    print(f"Diagnostic inference time: {elapsed_ms:.2f} ms")
    print(f"Points overlay: {points_path}")
    print(f"Indexed overlay: {indexed_path}")
    print(f"Keypoint CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
