from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import face_alignment
import numpy as np
import torch


EXPECTED_KEYPOINTS = 68

PATH_COLUMNS = (
    "relative_path",
    "image_path",
    "file_path",
    "filepath",
    "path",
)

ID_COLUMNS = (
    "sample_id",
    "image_id",
    "id",
    "name",
)

CLASS_COLUMNS = (
    "class_name",
    "emotion",
    "label_name",
    "category",
    "class",
    "label",
)

HASH_COLUMNS = (
    "sha256",
    "sha_256",
    "file_sha256",
)

AUTO_COLUMNS = [
    "sample_id",
    "class_name",
    "relative_path",
    "manifest_sha256",
    "computed_sha256",
    "sha256_match",
    "image_width",
    "image_height",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "inference_success",
    "num_keypoints",
    "finite_keypoints",
    "finite_rate",
    "in_bounds_keypoints",
    "in_bounds_rate",
    "score_threshold",
    "visible_keypoints",
    "visible_rate",
    "score_min",
    "score_p10",
    "score_mean",
    "score_median",
    "score_p90",
    "score_max",
    "inference_ms",
    "overlay_path",
    "error_type",
    "error_message",
]

REVIEW_COLUMNS = AUTO_COLUMNS + [
    "review_label",
    "manual_valid",
    "eyes_fit",
    "eyebrows_fit",
    "mouth_fit",
    "open_mouth_response",
    "template_risk",
    "review_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark 2D FAN-4 on a FER manifest using the whole "
            "48x48 image as an externally supplied face bounding box."
        )
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
        required=True,
        help="CSV manifest, e.g. sample_manifest_35.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory, e.g. outputs/fan_face/smoke_35.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Use cpu for the official benchmark.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.20,
        help=(
            "Diagnostic heatmap-score threshold used only for "
            "visibility statistics and overlay color."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warm-up runs before timed inference.",
    )
    parser.add_argument(
        "--overlay-scale",
        type=int,
        default=10,
        help="Scale factor for the saved 48x48 overlays.",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Do not recompute image SHA-256.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debugging limit. Do not use for the official 35 run.",
    )
    return parser.parse_args()


def resolve_from_project(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def locate_column(
    fieldnames: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    lowered = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def read_manifest(
    path: Path,
) -> tuple[list[dict[str, str]], dict[str, str | None]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"Manifest has no header: {path}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")

    columns = {
        "path": locate_column(fieldnames, PATH_COLUMNS),
        "id": locate_column(fieldnames, ID_COLUMNS),
        "class": locate_column(fieldnames, CLASS_COLUMNS),
        "sha256": locate_column(fieldnames, HASH_COLUMNS),
    }

    if columns["path"] is None:
        raise KeyError(
            "Could not identify the image-path column. "
            f"Manifest columns: {fieldnames}"
        )

    return rows, columns


def infer_class_name(
    row: dict[str, str],
    class_column: str | None,
    relative_path: str,
) -> str:
    if class_column and row.get(class_column, "").strip():
        return row[class_column].strip()

    parts = Path(relative_path.replace("\\", "/")).parts
    if len(parts) >= 2:
        return parts[-2]

    return "unknown"


def resolve_image_path(
    dataset_root: Path,
    manifest_value: str,
) -> Path:
    raw = Path(manifest_value.replace("\\", "/"))

    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [
            dataset_root / raw,
            dataset_root.parent / raw,
            dataset_root.parent.parent / raw,
        ]

        if raw.parts and raw.parts[0].lower() == dataset_root.name.lower():
            candidates.insert(0, dataset_root.parent / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Image not found for manifest value: {manifest_value}\n"
        f"Tried:\n{tried}"
    )


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def extract_predictions(
    result: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if result is None:
        raise RuntimeError("FAN returned no result.")

    if not isinstance(result, tuple) or len(result) != 3:
        raise RuntimeError(
            "Unexpected FAN return value. Expected "
            "(landmarks, landmark_scores, detected_faces)."
        )

    landmarks_list, scores_list, _ = result

    if landmarks_list is None or len(landmarks_list) == 0:
        raise RuntimeError("FAN returned no landmark set.")

    if scores_list is None or len(scores_list) == 0:
        raise RuntimeError("FAN returned no landmark scores.")

    keypoints = to_numpy(landmarks_list[0]).astype(np.float32)
    scores = to_numpy(scores_list[0]).astype(np.float32).reshape(-1)

    if keypoints.shape != (EXPECTED_KEYPOINTS, 2):
        raise RuntimeError(
            f"Expected keypoints shape ({EXPECTED_KEYPOINTS}, 2), "
            f"received {keypoints.shape}."
        )

    if scores.shape != (EXPECTED_KEYPOINTS,):
        raise RuntimeError(
            f"Expected score shape ({EXPECTED_KEYPOINTS},), "
            f"received {scores.shape}."
        )

    return keypoints, scores


def percentile(
    values: np.ndarray,
    q: float,
) -> float:
    return float(np.percentile(values.astype(np.float64), q))


def make_overlay(
    image_bgr: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    scale: int,
) -> np.ndarray:
    overlay = cv2.resize(
        image_bgr,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    for (x, y), score in zip(keypoints, scores):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        center = (
            int(round(float(x) * scale)),
            int(round(float(y) * scale)),
        )

        color = (
            (0, 255, 0)
            if float(score) >= threshold
            else (0, 165, 255)
        )

        cv2.circle(
            overlay,
            center,
            radius=max(2, scale // 4),
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    return overlay


def make_failure_overlay(
    image_bgr: np.ndarray,
    message: str,
    scale: int,
) -> np.ndarray:
    overlay = cv2.resize(
        image_bgr,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    cv2.rectangle(
        overlay,
        (0, 0),
        (overlay.shape[1], 44),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        overlay,
        "INFERENCE FAILED",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        overlay,
        message[:80],
        (8, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return overlay


def add_caption(
    image: np.ndarray,
    sample_id: str,
    class_name: str,
) -> np.ndarray:
    caption_height = 32
    canvas = np.zeros(
        (image.shape[0] + caption_height, image.shape[1], 3),
        dtype=np.uint8,
    )
    canvas[caption_height:] = image

    cv2.putText(
        canvas,
        f"{sample_id} | {class_name}",
        (6, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas


def build_contact_sheet(
    overlay_paths: list[Path],
    labels: list[tuple[str, str]],
    output_path: Path,
    columns: int = 5,
    tile_size: int = 220,
) -> None:
    if not overlay_paths:
        return

    tiles: list[np.ndarray] = []

    for path, (sample_id, class_name) in zip(
        overlay_paths,
        labels,
    ):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        image = cv2.resize(
            image,
            (tile_size, tile_size),
            interpolation=cv2.INTER_AREA,
        )

        tiles.append(
            add_caption(
                image,
                sample_id,
                class_name,
            )
        )

    if not tiles:
        return

    rows = math.ceil(len(tiles) / columns)
    tile_h, tile_w = tiles[0].shape[:2]

    sheet = np.zeros(
        (rows * tile_h, columns * tile_w, 3),
        dtype=np.uint8,
    )

    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns

        sheet[
            row * tile_h : (row + 1) * tile_h,
            col * tile_w : (col + 1) * tile_w,
        ] = tile

    cv2.imwrite(str(output_path), sheet)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: row.get(column, "")
                    for column in columns
                }
            )


def safe_stats(
    values: list[float],
) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }

    array = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def build_summary(
    rows: list[dict[str, Any]],
    manifest_path: Path,
    device: str,
    score_threshold: float,
) -> dict[str, Any]:
    total = len(rows)
    success_rows = [
        row
        for row in rows
        if row["inference_success"] is True
    ]
    times = [
        float(row["inference_ms"])
        for row in success_rows
    ]

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        by_class[str(row["class_name"])].append(row)

    per_class: dict[str, Any] = {}

    for class_name, class_rows in sorted(by_class.items()):
        class_success = [
            row
            for row in class_rows
            if row["inference_success"] is True
        ]

        class_times = [
            float(row["inference_ms"])
            for row in class_success
        ]

        per_class[class_name] = {
            "samples": len(class_rows),
            "raw_inference_successes": len(class_success),
            "raw_inference_success_rate": (
                len(class_success) / len(class_rows)
                if class_rows
                else 0.0
            ),
            "timing_ms": safe_stats(class_times),
            "mean_keypoint_score": (
                float(
                    np.mean(
                        [
                            float(row["score_mean"])
                            for row in class_success
                        ]
                    )
                )
                if class_success
                else None
            ),
            "mean_in_bounds_rate": (
                float(
                    np.mean(
                        [
                            float(row["in_bounds_rate"])
                            for row in class_success
                        ]
                    )
                )
                if class_success
                else None
            ),
        }

    return {
        "model": "2D FAN-4",
        "expected_keypoints": EXPECTED_KEYPOINTS,
        "device": device,
        "manifest": str(manifest_path),
        "score_threshold": score_threshold,
        "total_samples": total,
        "raw_inference_successes": len(success_rows),
        "raw_inference_failures": total - len(success_rows),
        "raw_inference_success_rate": (
            len(success_rows) / total
            if total
            else 0.0
        ),
        "timing_ms": safe_stats(times),
        "mean_keypoint_score": (
            float(
                np.mean(
                    [
                        float(row["score_mean"])
                        for row in success_rows
                    ]
                )
            )
            if success_rows
            else None
        ),
        "mean_visible_rate": (
            float(
                np.mean(
                    [
                        float(row["visible_rate"])
                        for row in success_rows
                    ]
                )
            )
            if success_rows
            else None
        ),
        "mean_in_bounds_rate": (
            float(
                np.mean(
                    [
                        float(row["in_bounds_rate"])
                        for row in success_rows
                    ]
                )
            )
            if success_rows
            else None
        ),
        "per_class": per_class,
        "important_note": (
            "FAN is given the whole image as an external face bounding box. "
            "Returning 68 points is only raw inference success, not proof that "
            "the landmarks are anatomically correct. FAN heatmap scores can "
            "remain high even when landmarks are visibly misplaced."
        ),
    }


def write_environment(
    path: Path,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# FAN benchmark environment",
        "",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        f"- NumPy: {np.__version__}",
        f"- OpenCV: {cv2.__version__}",
        f"- PyTorch: {torch.__version__}",
        f"- Torch CUDA build: {torch.version.cuda}",
        f"- CUDA available: {torch.cuda.is_available()}",
        f"- Benchmark device: {args.device}",
        f"- Torch CPU threads: {torch.get_num_threads()}",
        "- face-alignment: 1.5.0",
        "- Model: 2D FAN-4",
        "- Landmarks: 68",
        f"- Score threshold: {args.score_threshold}",
        f"- Warm-up runs: {args.warmup}",
        "- flip_input: False",
        "- compile: False",
        "",
        "The full FER image was passed as an externally supplied face bounding box.",
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_notes(path: Path) -> None:
    path.write_text(
        """# Notes

- Input: original FER-2013 48x48 grayscale image converted to three-channel RGB.
- Bounding box: the entire image, `[0, 0, width-1, height-1]`.
- FAN's internal face detector is bypassed by passing `detected_faces`.
- `flip_input=False` and `compile=False`.
- Model loading, image reading, SHA-256 checking, CSV writing, and overlay saving are excluded from `inference_ms`.
- Warm-up calls are excluded from all timing statistics.
- `inference_success` only means that 68 finite keypoints were returned.
- It does **not** mean all landmarks are visually correct.
- FAN heatmap scores may remain high even when eyebrows or occluded-side landmarks are visibly misplaced.
- Complete manual review in `sample_metrics_reviewed.csv` using:
  - `good`
  - `acceptable`
  - `wrong`
- Recommended `manual_valid`:
  - `1` for good or acceptable
  - `0` for wrong
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]

    dataset_root = args.dataset_root.resolve()
    manifest_path = resolve_from_project(
        project_root,
        args.manifest,
    )
    output_dir = resolve_from_project(
        project_root,
        args.output_dir,
    )

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}"
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows, columns = read_manifest(manifest_path)

    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"Manifest: {manifest_path}")
    print(f"Samples: {len(rows)}")
    print(f"Device: {args.device}")
    print("Loading 2D FAN-4 once...")

    fan = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=args.device,
        flip_input=False,
        compile=False,
    )

    actual_device = str(
        next(fan.face_alignment_net.parameters()).device
    )
    print(f"Model parameter device: {actual_device}")

    path_column = str(columns["path"])

    warmup_image: np.ndarray | None = None
    warmup_faces: list[np.ndarray] | None = None

    for manifest_row in rows:
        try:
            path = resolve_image_path(
                dataset_root,
                manifest_row[path_column],
            )

            gray = cv2.imread(
                str(path),
                cv2.IMREAD_GRAYSCALE,
            )

            if gray is None:
                continue

            bgr = cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2BGR,
            )
            warmup_image = cv2.cvtColor(
                bgr,
                cv2.COLOR_BGR2RGB,
            )

            h, w = warmup_image.shape[:2]

            warmup_faces = [
                np.array(
                    [
                        0.0,
                        0.0,
                        float(w - 1),
                        float(h - 1),
                    ],
                    dtype=np.float32,
                )
            ]
            break

        except Exception:
            continue

    if warmup_image is None or warmup_faces is None:
        raise RuntimeError(
            "Could not find a readable image for warm-up."
        )

    print(f"Warm-up runs: {args.warmup}")

    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            fan.get_landmarks_from_image(
                warmup_image,
                detected_faces=warmup_faces,
                return_landmark_score=True,
            )

    result_rows: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []
    overlay_labels: list[tuple[str, str]] = []

    for index, manifest_row in enumerate(
        rows,
        start=1,
    ):
        relative_path = manifest_row[path_column]

        id_column = columns["id"]
        class_column = columns["class"]
        hash_column = columns["sha256"]

        sample_id = (
            manifest_row[str(id_column)].strip()
            if id_column
            and manifest_row.get(
                str(id_column),
                "",
            ).strip()
            else Path(relative_path).stem
        )

        class_name = infer_class_name(
            manifest_row,
            str(class_column)
            if class_column
            else None,
            relative_path,
        )

        manifest_hash = (
            manifest_row[str(hash_column)].strip().lower()
            if hash_column
            and manifest_row.get(
                str(hash_column),
                "",
            ).strip()
            else ""
        )

        print(
            f"[{index:03d}/{len(rows):03d}] "
            f"{sample_id} ({class_name})",
            end=" ... ",
        )

        base_row: dict[str, Any] = {
            "sample_id": sample_id,
            "class_name": class_name,
            "relative_path": relative_path,
            "manifest_sha256": manifest_hash,
            "computed_sha256": "",
            "sha256_match": "",
            "image_width": "",
            "image_height": "",
            "bbox_x1": 0.0,
            "bbox_y1": 0.0,
            "bbox_x2": "",
            "bbox_y2": "",
            "inference_success": False,
            "num_keypoints": 0,
            "finite_keypoints": 0,
            "finite_rate": 0.0,
            "in_bounds_keypoints": 0,
            "in_bounds_rate": 0.0,
            "score_threshold": args.score_threshold,
            "visible_keypoints": 0,
            "visible_rate": 0.0,
            "score_min": "",
            "score_p10": "",
            "score_mean": "",
            "score_median": "",
            "score_p90": "",
            "score_max": "",
            "inference_ms": "",
            "overlay_path": "",
            "error_type": "",
            "error_message": "",
        }

        image_bgr: np.ndarray | None = None

        try:
            image_path = resolve_image_path(
                dataset_root,
                relative_path,
            )

            computed_hash = (
                ""
                if args.skip_hash_check
                else sha256_file(image_path)
            )

            base_row["computed_sha256"] = computed_hash

            if manifest_hash and computed_hash:
                base_row["sha256_match"] = (
                    manifest_hash == computed_hash
                )

            image_gray = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if image_gray is None:
                raise RuntimeError(
                    f"OpenCV failed to read: {image_path}"
                )

            image_bgr = cv2.cvtColor(
                image_gray,
                cv2.COLOR_GRAY2BGR,
            )
            image_rgb = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2RGB,
            )

            height, width = image_rgb.shape[:2]

            base_row["image_width"] = width
            base_row["image_height"] = height
            base_row["bbox_x2"] = float(width - 1)
            base_row["bbox_y2"] = float(height - 1)

            detected_faces = [
                np.array(
                    [
                        0.0,
                        0.0,
                        float(width - 1),
                        float(height - 1),
                    ],
                    dtype=np.float32,
                )
            ]

            start = time.perf_counter()

            with torch.inference_mode():
                result = fan.get_landmarks_from_image(
                    image_rgb,
                    detected_faces=detected_faces,
                    return_landmark_score=True,
                )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000.0

            keypoints, scores = extract_predictions(result)

            finite_mask = (
                np.isfinite(keypoints).all(axis=1)
                & np.isfinite(scores)
            )

            in_bounds_mask = (
                finite_mask
                & (keypoints[:, 0] >= 0.0)
                & (
                    keypoints[:, 0]
                    <= float(width - 1)
                )
                & (keypoints[:, 1] >= 0.0)
                & (
                    keypoints[:, 1]
                    <= float(height - 1)
                )
            )

            visible_mask = (
                finite_mask
                & (scores >= args.score_threshold)
            )

            finite_count = int(np.sum(finite_mask))
            in_bounds_count = int(
                np.sum(in_bounds_mask)
            )
            visible_count = int(
                np.sum(visible_mask)
            )

            inference_success = (
                keypoints.shape
                == (EXPECTED_KEYPOINTS, 2)
                and finite_count
                == EXPECTED_KEYPOINTS
            )

            base_row.update(
                {
                    "inference_success": inference_success,
                    "num_keypoints": int(
                        keypoints.shape[0]
                    ),
                    "finite_keypoints": finite_count,
                    "finite_rate": (
                        finite_count
                        / EXPECTED_KEYPOINTS
                    ),
                    "in_bounds_keypoints": (
                        in_bounds_count
                    ),
                    "in_bounds_rate": (
                        in_bounds_count
                        / EXPECTED_KEYPOINTS
                    ),
                    "visible_keypoints": visible_count,
                    "visible_rate": (
                        visible_count
                        / EXPECTED_KEYPOINTS
                    ),
                    "score_min": float(
                        np.min(scores)
                    ),
                    "score_p10": percentile(
                        scores,
                        10,
                    ),
                    "score_mean": float(
                        np.mean(scores)
                    ),
                    "score_median": float(
                        np.median(scores)
                    ),
                    "score_p90": percentile(
                        scores,
                        90,
                    ),
                    "score_max": float(
                        np.max(scores)
                    ),
                    "inference_ms": elapsed_ms,
                }
            )

            overlay = make_overlay(
                image_bgr,
                keypoints,
                scores,
                args.score_threshold,
                args.overlay_scale,
            )

            overlay_path = (
                overlays_dir
                / f"{sample_id}.png"
            )

            if not cv2.imwrite(
                str(overlay_path),
                overlay,
            ):
                raise RuntimeError(
                    f"Failed to save overlay: "
                    f"{overlay_path}"
                )

            base_row["overlay_path"] = str(
                overlay_path.relative_to(
                    output_dir
                )
            )

            overlay_paths.append(overlay_path)
            overlay_labels.append(
                (
                    sample_id,
                    class_name,
                )
            )

            print(
                f"OK | {elapsed_ms:.2f} ms | "
                f"mean score "
                f"{float(np.mean(scores)):.3f}"
            )

        except Exception as error:
            base_row["error_type"] = (
                type(error).__name__
            )
            base_row["error_message"] = (
                str(error).replace(
                    "\n",
                    " | ",
                )
            )

            print(
                f"FAILED | "
                f"{type(error).__name__}: "
                f"{error}"
            )

            if image_bgr is not None:
                failure_overlay = (
                    make_failure_overlay(
                        image_bgr,
                        str(error),
                        args.overlay_scale,
                    )
                )

                failure_path = (
                    overlays_dir
                    / f"{sample_id}.png"
                )

                cv2.imwrite(
                    str(failure_path),
                    failure_overlay,
                )

                base_row["overlay_path"] = str(
                    failure_path.relative_to(
                        output_dir
                    )
                )

                overlay_paths.append(
                    failure_path
                )
                overlay_labels.append(
                    (
                        sample_id,
                        class_name,
                    )
                )

        result_rows.append(base_row)

    auto_csv_path = (
        output_dir
        / "sample_metrics_auto.csv"
    )
    reviewed_csv_path = (
        output_dir
        / "sample_metrics_reviewed.csv"
    )
    summary_path = (
        output_dir
        / "summary_auto.json"
    )
    environment_path = (
        output_dir
        / "environment.md"
    )
    notes_path = (
        output_dir
        / "notes.md"
    )
    contact_sheet_path = (
        output_dir
        / "contact_sheet.png"
    )

    write_csv(
        auto_csv_path,
        result_rows,
        AUTO_COLUMNS,
    )

    review_rows: list[dict[str, Any]] = []

    for row in result_rows:
        review_row = dict(row)

        review_row.update(
            {
                "review_label": "",
                "manual_valid": "",
                "eyes_fit": "",
                "eyebrows_fit": "",
                "mouth_fit": "",
                "open_mouth_response": "",
                "template_risk": "",
                "review_notes": "",
            }
        )

        review_rows.append(review_row)

    write_csv(
        reviewed_csv_path,
        review_rows,
        REVIEW_COLUMNS,
    )

    summary = build_summary(
        result_rows,
        manifest_path,
        args.device,
        args.score_threshold,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_environment(
        environment_path,
        args,
    )

    write_notes(notes_path)

    build_contact_sheet(
        overlay_paths,
        overlay_labels,
        contact_sheet_path,
        columns=5,
    )

    print()
    print("=" * 72)
    print("FAN benchmark completed")
    print(f"Samples: {summary['total_samples']}")
    print(
        "Raw inference success: "
        f"{summary['raw_inference_successes']}/"
        f"{summary['total_samples']} "
        f"({summary['raw_inference_success_rate'] * 100:.2f}%)"
    )

    timing = summary["timing_ms"]

    print(
        "Inference time: "
        f"mean={timing['mean']:.2f} ms, "
        f"median={timing['median']:.2f} ms, "
        f"P95={timing['p95']:.2f} ms, "
        f"max={timing['max']:.2f} ms"
    )

    print(f"Auto metrics: {auto_csv_path}")
    print(
        "Manual review template: "
        f"{reviewed_csv_path}"
    )
    print(f"Summary: {summary_path}")
    print(
        f"Contact sheet: "
        f"{contact_sheet_path}"
    )
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted by user.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
