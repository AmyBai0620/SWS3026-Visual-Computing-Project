#!/usr/bin/env python
"""
Run MediaPipe Face Mesh on a fixed FER-2013 manifest.

This script is designed for the 35-image smoke test first. It:
- reads images strictly from sample_manifest_35.csv / sample_manifest_350.csv;
- verifies each file hash;
- upscales FER images before MediaPipe inference;
- saves one original-vs-overlay image per sample;
- writes sample_metrics.csv, summary.json, environment.md, and notes.md;
- never modifies the source dataset or the existing baseline model.

Manual-review fields are initialized as "pending_review" and should be
updated after checking the generated overlays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


MANUAL_REVIEW_DEFAULT = "pending_review"

OUTPUT_FIELDS = [
    "sample_id",
    "label",
    "relative_path",
    "extract_success",
    "num_landmarks",
    "runtime_ms",
    "auto_geometry_valid",
    "face_bbox_area_ratio",
    "mouth_open_ratio",
    "landmark_valid",
    "eye_fit",
    "eyebrow_fit",
    "mouth_fit",
    "mouth_open_response",
    "template_risk",
    "failure_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MediaPipe Face Mesh on a fixed FER manifest."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="FER train directory containing angry/disgust/... folders.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to sample_manifest_35.csv or sample_manifest_350.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for metrics, summaries, and overlays.",
    )
    parser.add_argument(
        "--upscale-size",
        type=int,
        default=256,
        help="Resize each FER image to this square size before inference.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum detection confidence.",
    )
    parser.add_argument(
        "--refine-landmarks",
        action="store_true",
        help="Use iris refinement (478 points instead of 468 when supported).",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def point_distance(points: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(points[i] - points[j]))


def compute_mouth_open_ratio(points: np.ndarray) -> float | None:
    # MediaPipe semantic landmarks:
    # 13: upper inner lip, 14: lower inner lip
    # 78: left mouth corner, 308: right mouth corner
    required = [13, 14, 78, 308]
    if max(required) >= len(points):
        return None

    vertical = point_distance(points, 13, 14)
    horizontal = point_distance(points, 78, 308)
    if horizontal <= 1e-8:
        return None
    return vertical / horizontal


def automatic_geometry_check(points: np.ndarray) -> tuple[bool, float, str]:
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 100:
        return False, 0.0, "invalid_landmark_shape"

    if not np.isfinite(points).all():
        return False, 0.0, "non_finite_landmarks"

    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    width, height = max_xy - min_xy
    area_ratio = float(max(width, 0.0) * max(height, 0.0))

    # These are intentionally loose checks. They only catch collapsed or
    # severely out-of-frame meshes; visual validity still requires review.
    inside_ratio = float(
        np.mean(
            (points[:, 0] >= -0.05)
            & (points[:, 0] <= 1.05)
            & (points[:, 1] >= -0.05)
            & (points[:, 1] <= 1.05)
        )
    )

    if width < 0.12 or height < 0.12:
        return False, area_ratio, "collapsed_face_mesh"
    if area_ratio < 0.025:
        return False, area_ratio, "face_mesh_too_small_or_collapsed"
    if inside_ratio < 0.90:
        return False, area_ratio, "too_many_landmarks_outside_image"

    return True, area_ratio, ""


def draw_overlay(
    original_gray: np.ndarray,
    resized_bgr: np.ndarray,
    points: np.ndarray | None,
    sample_id: str,
    success: bool,
    runtime_ms: float,
) -> np.ndarray:
    left = cv2.resize(original_gray, (256, 256), interpolation=cv2.INTER_NEAREST)
    left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    right = cv2.resize(resized_bgr, (256, 256), interpolation=cv2.INTER_AREA)

    if points is not None:
        for x_norm, y_norm in points:
            x = int(round(x_norm * (right.shape[1] - 1)))
            y = int(round(y_norm * (right.shape[0] - 1)))
            if 0 <= x < right.shape[1] and 0 <= y < right.shape[0]:
                cv2.circle(right, (x, y), 1, (255, 0, 0), -1, cv2.LINE_AA)

        # Emphasize mouth landmarks used for the mouth-opening ratio.
        for index in [13, 14, 78, 308]:
            if index < len(points):
                x = int(round(points[index, 0] * (right.shape[1] - 1)))
                y = int(round(points[index, 1] * (right.shape[0] - 1)))
                cv2.circle(right, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)

    canvas = np.hstack([left, right])
    status = "SUCCESS" if success else "FAILED"
    cv2.putText(
        canvas,
        f"{sample_id} | {status} | {runtime_ms:.2f} ms",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 0) if success else (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Original 48x48 (shown enlarged)",
        (8, 246),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "MediaPipe overlay",
        (270, 246),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    overlays_dir = output_dir / "overlays"

    if not dataset_root.is_dir():
        print(f"ERROR: dataset root not found: {dataset_root}", file=sys.stderr)
        return 2
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    if args.upscale_size < 64:
        print("ERROR: --upscale-size should be at least 64.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        manifest_rows = list(csv.DictReader(file))

    if not manifest_rows:
        print("ERROR: manifest is empty.", file=sys.stderr)
        return 2

    mp_face_mesh = mp.solutions.face_mesh
    metrics_rows: list[dict[str, Any]] = []
    runtimes: list[float] = []
    class_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "extract_success": 0, "auto_geometry_valid": 0}
    )

    print(f"MediaPipe version: {mp.__version__}")
    print(f"Samples: {len(manifest_rows)}")
    print(f"Upscale size: {args.upscale_size}x{args.upscale_size}")
    print()

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=args.refine_landmarks,
        min_detection_confidence=args.min_detection_confidence,
    ) as face_mesh:
        for index, manifest_row in enumerate(manifest_rows, start=1):
            sample_id = manifest_row["sample_id"]
            label = manifest_row["label"]
            relative_path = manifest_row["relative_path"]
            image_path = dataset_root / Path(relative_path)
            expected_hash = manifest_row.get("sha256", "").strip()

            class_stats[label]["total"] += 1
            extract_success = False
            num_landmarks = 0
            runtime_ms = 0.0
            auto_geometry_valid = False
            face_bbox_area_ratio = 0.0
            mouth_open_ratio: float | None = None
            failure_reason = ""
            points: np.ndarray | None = None

            original_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if original_gray is None:
                failure_reason = "image_read_failed"
                placeholder = np.zeros((48, 48), dtype=np.uint8)
                resized_bgr = cv2.cvtColor(
                    cv2.resize(
                        placeholder,
                        (args.upscale_size, args.upscale_size),
                        interpolation=cv2.INTER_CUBIC,
                    ),
                    cv2.COLOR_GRAY2BGR,
                )
            else:
                if expected_hash:
                    actual_hash = sha256_file(image_path)
                    if actual_hash.lower() != expected_hash.lower():
                        failure_reason = "sha256_mismatch"

                resized_gray = cv2.resize(
                    original_gray,
                    (args.upscale_size, args.upscale_size),
                    interpolation=cv2.INTER_CUBIC,
                )
                resized_bgr = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2BGR)
                rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

                if not failure_reason:
                    start = time.perf_counter()
                    result = face_mesh.process(rgb)
                    runtime_ms = (time.perf_counter() - start) * 1000.0
                    runtimes.append(runtime_ms)

                    if result.multi_face_landmarks:
                        landmark_list = result.multi_face_landmarks[0].landmark
                        points = np.array(
                            [(landmark.x, landmark.y) for landmark in landmark_list],
                            dtype=np.float32,
                        )
                        num_landmarks = len(points)
                        extract_success = True
                        class_stats[label]["extract_success"] += 1

                        (
                            auto_geometry_valid,
                            face_bbox_area_ratio,
                            geometry_reason,
                        ) = automatic_geometry_check(points)
                        if auto_geometry_valid:
                            class_stats[label]["auto_geometry_valid"] += 1
                        elif not failure_reason:
                            failure_reason = geometry_reason

                        mouth_open_ratio = compute_mouth_open_ratio(points)
                    else:
                        failure_reason = "no_face_landmarks"

            overlay = draw_overlay(
                original_gray
                if original_gray is not None
                else np.zeros((48, 48), dtype=np.uint8),
                resized_bgr,
                points,
                sample_id,
                extract_success,
                runtime_ms,
            )
            overlay_path = overlays_dir / f"{sample_id}.jpg"
            if not cv2.imwrite(str(overlay_path), overlay):
                print(f"WARNING: failed to write overlay: {overlay_path}")

            metrics_rows.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "relative_path": relative_path,
                    "extract_success": str(extract_success).lower(),
                    "num_landmarks": num_landmarks,
                    "runtime_ms": f"{runtime_ms:.4f}",
                    "auto_geometry_valid": str(auto_geometry_valid).lower(),
                    "face_bbox_area_ratio": f"{face_bbox_area_ratio:.6f}",
                    "mouth_open_ratio": (
                        "" if mouth_open_ratio is None else f"{mouth_open_ratio:.6f}"
                    ),
                    "landmark_valid": MANUAL_REVIEW_DEFAULT,
                    "eye_fit": MANUAL_REVIEW_DEFAULT,
                    "eyebrow_fit": MANUAL_REVIEW_DEFAULT,
                    "mouth_fit": MANUAL_REVIEW_DEFAULT,
                    "mouth_open_response": MANUAL_REVIEW_DEFAULT,
                    "template_risk": MANUAL_REVIEW_DEFAULT,
                    "failure_reason": failure_reason,
                }
            )

            print(
                f"[{index:>3}/{len(manifest_rows)}] "
                f"{sample_id:<14} "
                f"success={str(extract_success):<5} "
                f"points={num_landmarks:<3} "
                f"time={runtime_ms:>7.2f} ms "
                f"reason={failure_reason or '-'}"
            )

    metrics_path = output_dir / "sample_metrics.csv"
    summary_path = output_dir / "summary.json"
    environment_path = output_dir / "environment.md"
    notes_path = output_dir / "notes.md"

    write_csv(metrics_path, metrics_rows)

    total = len(metrics_rows)
    success_count = sum(row["extract_success"] == "true" for row in metrics_rows)
    auto_valid_count = sum(
        row["auto_geometry_valid"] == "true" for row in metrics_rows
    )

    per_class_results: dict[str, Any] = {}
    for label, stats in sorted(class_stats.items()):
        class_total = stats["total"]
        per_class_results[label] = {
            "total": class_total,
            "extract_success_count": stats["extract_success"],
            "extract_success_rate": (
                stats["extract_success"] / class_total if class_total else 0.0
            ),
            "auto_geometry_valid_count": stats["auto_geometry_valid"],
            "auto_geometry_valid_rate": (
                stats["auto_geometry_valid"] / class_total if class_total else 0.0
            ),
        }

    summary = {
        "model_name": "MediaPipe Face Mesh",
        "model_version": mp.__version__,
        "manifest": str(manifest_path),
        "total_samples": total,
        "extract_success_count": success_count,
        "extract_success_rate": success_count / total if total else 0.0,
        "auto_geometry_valid_count": auto_valid_count,
        "auto_geometry_valid_rate": auto_valid_count / total if total else 0.0,
        "manual_landmark_valid_count": None,
        "manual_landmark_valid_rate": None,
        "num_landmarks": (
            478 if args.refine_landmarks else 468
        ),
        "mean_runtime_ms": statistics.fmean(runtimes) if runtimes else 0.0,
        "median_runtime_ms": statistics.median(runtimes) if runtimes else 0.0,
        "p95_runtime_ms": percentile(runtimes, 0.95),
        "max_runtime_ms": max(runtimes) if runtimes else 0.0,
        "upscale_size": args.upscale_size,
        "refine_landmarks": args.refine_landmarks,
        "min_detection_confidence": args.min_detection_confidence,
        "per_class_results": per_class_results,
        "hardware": platform.processor(),
        "software_environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "mediapipe": mp.__version__,
            "numpy": np.__version__,
        },
        "recommend_full_run": None,
        "notes": (
            "Manual quality fields remain pending_review until overlays are checked."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    environment_path.write_text(
        "\n".join(
            [
                "# MediaPipe 测试环境",
                "",
                f"- 操作系统：{platform.platform()}",
                f"- Python：{sys.version.split()[0]}",
                f"- OpenCV：{cv2.__version__}",
                f"- MediaPipe：{mp.__version__}",
                f"- NumPy：{np.__version__}",
                f"- CPU：{platform.processor() or '未自动识别'}",
                "- GPU：本脚本未显式使用 GPU",
                f"- 上采样尺寸：{args.upscale_size}×{args.upscale_size}",
                f"- refine_landmarks：{args.refine_landmarks}",
                f"- min_detection_confidence：{args.min_detection_confidence}",
                "",
                "## 运行命令",
                "",
                "请把实际运行命令补充在这里。",
            ]
        ),
        encoding="utf-8",
    )

    notes_path.write_text(
        """# MediaPipe 35 张试跑检查说明

1. 逐张查看 `overlays/`。
2. 在 `sample_metrics.csv` 中填写以下人工字段：
   - landmark_valid
   - eye_fit
   - eyebrow_fit
   - mouth_fit
   - mouth_open_response
   - template_risk
3. 质量字段统一填写：
   - good
   - acceptable
   - wrong
   - not_applicable
4. template_risk 统一填写：
   - low
   - medium
   - high
   - unknown
5. 自动几何检查只负责发现明显崩坏，不能替代人工检查。
6. 35 张全部检查后，再决定是否继续跑 350 张。
""",
        encoding="utf-8",
    )

    print()
    print("Completed.")
    print(f"Metrics:     {metrics_path}")
    print(f"Summary:     {summary_path}")
    print(f"Environment: {environment_path}")
    print(f"Notes:       {notes_path}")
    print(f"Overlays:    {overlays_dir}")
    print()
    print(
        f"Extraction success: {success_count}/{total} "
        f"({(success_count / total * 100.0 if total else 0.0):.2f}%)"
    )
    print(
        f"Auto geometry valid: {auto_valid_count}/{total} "
        f"({(auto_valid_count / total * 100.0 if total else 0.0):.2f}%)"
    )
    if runtimes:
        print(
            "Runtime: "
            f"mean={statistics.fmean(runtimes):.2f} ms, "
            f"median={statistics.median(runtimes):.2f} ms, "
            f"p95={percentile(runtimes, 0.95):.2f} ms"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
