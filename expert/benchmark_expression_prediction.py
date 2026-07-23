from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

if __package__:
    from .rtmpose_emotion_recognizer import (
        RTMPoseEmotionRecognizer,
    )
    from .runtime_metrics import (
        PREDICTION_TARGET_MS,
        percentile,
    )
else:
    from rtmpose_emotion_recognizer import (
        RTMPoseEmotionRecognizer,
    )
    from runtime_metrics import (
        PREDICTION_TARGET_MS,
        percentile,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the complete expression prediction path used by the "
            "live demo: input conversion, RTMPose, feature construction, "
            "scaling, classification and probability estimation."
        )
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="auto",
        help="Inference device. 'auto' selects CUDA when available.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of untimed complete predictions before measurement.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of timed complete predictions.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help=(
            "Optional face-crop image. Without this option, a deterministic "
            "synthetic 256 x 256 face-like input is used."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path for per-run timing records.",
    )
    return parser.parse_args()


def create_synthetic_face() -> np.ndarray:
    """Create a deterministic face-like BGR crop for a portable benchmark."""
    image = np.full((256, 256, 3), 210, dtype=np.uint8)

    cv2.ellipse(
        image,
        (128, 128),
        (82, 104),
        0,
        0,
        360,
        (176, 196, 218),
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(image, (96, 111), 9, (45, 45, 45), -1, cv2.LINE_AA)
    cv2.circle(image, (160, 111), 9, (45, 45, 45), -1, cv2.LINE_AA)
    cv2.ellipse(
        image,
        (128, 171),
        (34, 18),
        0,
        0,
        180,
        (55, 55, 55),
        4,
        cv2.LINE_AA,
    )

    return image


def load_input(image_path: Optional[Path]) -> tuple[np.ndarray, str]:
    if image_path is None:
        return create_synthetic_face(), "synthetic 256 x 256 face-like crop"

    resolved = image_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Benchmark image not found: {resolved}")

    image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise RuntimeError(f"Could not read benchmark image: {resolved}")

    return image, str(resolved)


def summarize(name: str, values: list[float]) -> None:
    p95_ms = percentile(values, 0.95)
    if p95_ms is None:
        raise ValueError("Cannot summarize an empty timing list.")

    print(
        f"{name:<18} "
        f"mean {statistics.fmean(values):7.2f} ms   "
        f"median {statistics.median(values):7.2f} ms   "
        f"P95 {p95_ms:7.2f} ms   "
        f"max {max(values):7.2f} ms"
    )


def main() -> int:
    args = parse_args()

    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("--warmup must be >= 0 and --runs must be > 0.")

    face_bgr, input_description = load_input(args.image)

    print("=" * 88)
    print("Complete expression prediction benchmark")
    print("=" * 88)
    print(f"Requested device: {args.device}")
    print(f"Input:            {input_description}")
    print(f"Input shape:      {tuple(face_bgr.shape)}")
    print(f"Warm-up runs:     {args.warmup}")
    print(f"Timed runs:       {args.runs}")
    print(f"Project target:   <= {PREDICTION_TARGET_MS:.1f} ms per prediction")
    print()

    load_started = time.perf_counter()
    recognizer = RTMPoseEmotionRecognizer(
        device=args.device,
        warmup_runs=0,
    )
    load_ms = (time.perf_counter() - load_started) * 1000.0

    print(f"Resolved device:  {recognizer.device}")
    print(f"Pipeline loaded:  {load_ms:.1f} ms")
    print()

    if args.warmup:
        print("Warming up complete predictions...")

    for index in range(args.warmup):
        prediction = recognizer.predict(face_bgr)
        if not prediction.success:
            raise RuntimeError(
                "Warm-up prediction failed: "
                f"{prediction.error or 'unknown error'}"
            )
        print(
            f"  warm-up {index + 1:02d}/{args.warmup:02d}",
            end="\r",
        )

    if args.warmup:
        print()

    records: list[dict[str, object]] = []
    print("Benchmarking complete predictions...")

    for index in range(args.runs):
        external_started = time.perf_counter()
        prediction = recognizer.predict(face_bgr)
        external_ms = (time.perf_counter() - external_started) * 1000.0

        if not prediction.success:
            raise RuntimeError(
                f"Timed prediction {index + 1} failed: "
                f"{prediction.error or 'unknown error'}"
            )

        total_ms = recognizer.last_prediction_ms
        pose_ms = recognizer.last_pose_ms
        classifier_ms = recognizer.last_classifier_ms

        if total_ms is None or pose_ms is None or classifier_ms is None:
            raise RuntimeError(
                "Recognizer did not expose all required timing values."
            )

        other_ms = max(0.0, total_ms - pose_ms - classifier_ms)
        over_target = total_ms > PREDICTION_TARGET_MS

        records.append(
            {
                "run": index + 1,
                "prediction_ms": total_ms,
                "rtmpose_ms": pose_ms,
                "classifier_ms": classifier_ms,
                "other_pipeline_ms": other_ms,
                "external_call_ms": external_ms,
                "over_30ms": over_target,
                "label": prediction.label,
                "confidence": prediction.confidence,
            }
        )

        print(
            f"  run {index + 1:03d}/{args.runs:03d}: "
            f"total {total_ms:7.2f} ms   "
            f"pose {pose_ms:7.2f} ms   "
            f"classifier {classifier_ms:6.2f} ms",
            end="\r",
        )

    print()
    print()

    total_values = [float(row["prediction_ms"]) for row in records]
    pose_values = [float(row["rtmpose_ms"]) for row in records]
    classifier_values = [float(row["classifier_ms"]) for row in records]
    other_values = [float(row["other_pipeline_ms"]) for row in records]
    external_values = [float(row["external_call_ms"]) for row in records]

    print("Steady-state timing summary")
    print("-" * 88)
    summarize("Prediction total", total_values)
    summarize("RTMPose stage", pose_values)
    summarize("Classifier stage", classifier_values)
    summarize("Other pipeline", other_values)
    summarize("External call", external_values)
    print()

    average_ms = statistics.fmean(total_values)
    p95_ms = percentile(total_values, 0.95)
    if p95_ms is None:
        raise RuntimeError("No prediction timings were recorded.")
    maximum_ms = max(total_values)
    over_count = sum(
        1 for value in total_values if value > PREDICTION_TARGET_MS
    )
    approximate_rate = 1000.0 / average_ms if average_ms > 0.0 else float("inf")

    print(f"Approx. prediction capacity from the mean: {approximate_rate:.2f} predictions/s")
    print(
        f"Runs over {PREDICTION_TARGET_MS:.1f} ms: "
        f"{over_count}/{len(total_values)}"
    )
    print()

    if over_count == 0:
        print(
            "[PASS] Every measured complete prediction finished within "
            f"{PREDICTION_TARGET_MS:.1f} ms."
        )
    elif average_ms <= PREDICTION_TARGET_MS:
        print(
            "[MIXED] The mean meets the 30 ms target, but some runs exceed it. "
            f"Report mean={average_ms:.2f} ms, P95={p95_ms:.2f} ms and "
            f"max={maximum_ms:.2f} ms rather than claiming every run passes."
        )
    else:
        print(
            "[OVER] The complete prediction mean exceeds the 30 ms project "
            f"target: mean={average_ms:.2f} ms, P95={p95_ms:.2f} ms."
        )

    if args.csv is not None:
        csv_path = args.csv.expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        print(f"Per-run timings saved to: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
