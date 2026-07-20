#!/usr/bin/env python
"""Manifest-aligned MediaPipe Face Detection evidence runner.

This is a face-presence QC experiment, not a landmark extractor.  It reuses
the prior benchmark's deterministic 48px grayscale -> 256px cubic RGB
preprocessing, but deliberately uses ``mp.solutions.face_detection.FaceDetection``
instead of the benchmark's FaceMesh API.  A no-detection is raw evidence only;
it is not a declaration that the source image is invalid.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd


FIELDS = [
    "official_row_index", "sample_id", "split", "class_name", "relative_path",
    "image_read_success", "face_detected", "face_count", "max_face_confidence",
    "best_bbox_x1", "best_bbox_y1", "best_bbox_x2", "best_bbox_y2", "bbox_area_ratio",
    "processing_seconds", "error_message", "mediapipe_version", "mediapipe_api",
    "min_detection_confidence", "resize_width", "resize_height", "interpolation",
]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("experiments/expression_classification/manifests/official_protocol_manifest.csv"))
    p.add_argument("--dataset-root", type=Path, default=Path("expert/facial_expression_dataset"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--indices-csv", type=Path, help="CSV containing official_row_index for a deterministic subset")
    p.add_argument("--limit", type=int, help="Process only the first N selected rows")
    p.add_argument("--resume", action="store_true", help="Skip sample IDs already written to raw_detections.csv")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    return p.parse_args()


def write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = a.output_dir / "raw_detections.csv"
    manifest = pd.read_csv(a.manifest).reset_index(names="official_row_index")
    if a.indices_csv:
        indices = pd.read_csv(a.indices_csv)["official_row_index"].astype(int).tolist()
        wanted = set(indices)
        manifest = manifest[manifest.official_row_index.isin(wanted)].sort_values("official_row_index")
    if a.limit is not None:
        manifest = manifest.head(a.limit)
    rows: list[dict] = []
    if a.resume and output_csv.exists():
        rows = pd.read_csv(output_csv).to_dict("records")
        done = set(str(r["sample_id"]) for r in rows)
        manifest = manifest[~manifest.sample_id.astype(str).isin(done)]
    metadata = {"manifest": str(a.manifest), "dataset_root": str(a.dataset_root), "api": "mp.solutions.face_detection.FaceDetection", "resize": "48px grayscale -> 256x256 INTER_CUBIC -> RGB", "min_detection_confidence": a.min_detection_confidence, "mediapipe_version": mp.__version__, "resume": a.resume}
    (a.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    detector_api = "mp.solutions.face_detection.FaceDetection(model_selection=0)"
    pending = manifest.to_dict("records")
    with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=a.min_detection_confidence) as detector:
        for n, item in enumerate(pending, 1):
            row = {"official_row_index": int(item["official_row_index"]), "sample_id": item["sample_id"], "split": item["split"], "class_name": item["class_name"], "relative_path": item["relative_path"], "image_read_success": False, "face_detected": False, "face_count": 0, "max_face_confidence": "", "best_bbox_x1": "", "best_bbox_y1": "", "best_bbox_x2": "", "best_bbox_y2": "", "bbox_area_ratio": "", "processing_seconds": 0.0, "error_message": "", "mediapipe_version": mp.__version__, "mediapipe_api": detector_api, "min_detection_confidence": a.min_detection_confidence, "resize_width": 256, "resize_height": 256, "interpolation": "INTER_CUBIC"}
            try:
                image = cv2.imread(str(a.dataset_root / item["relative_path"]), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    row["error_message"] = "image_read_failed"
                else:
                    row["image_read_success"] = True
                    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
                    started = time.perf_counter()
                    result = detector.process(cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB))
                    row["processing_seconds"] = time.perf_counter() - started
                    detections = list(result.detections or [])
                    row["face_count"] = len(detections)
                    row["face_detected"] = bool(detections)
                    if detections:
                        best = max(detections, key=lambda d: float(d.score[0]) if d.score else -1.0)
                        b = best.location_data.relative_bounding_box
                        x1, y1 = float(b.xmin), float(b.ymin)
                        x2, y2 = x1 + float(b.width), y1 + float(b.height)
                        row.update(max_face_confidence=float(best.score[0]), best_bbox_x1=x1, best_bbox_y1=y1, best_bbox_x2=x2, best_bbox_y2=y2, bbox_area_ratio=max(0.0, b.width) * max(0.0, b.height))
            except Exception as exc:  # preserve failure evidence and continue
                row["error_message"] = f"{type(exc).__name__}: {exc}"[:1000]
            rows.append(row)
            if n % a.checkpoint_every == 0 or n == len(pending):
                write_rows(output_csv, rows)
                print(f"checkpoint {n}/{len(pending)} rows={len(rows)}")
    times = [float(r["processing_seconds"]) for r in rows if float(r["processing_seconds"]) > 0]
    summary = {"rows_written": len(rows), "face_detected": sum(str(r["face_detected"]).lower() == "true" for r in rows), "no_detection": sum(str(r["image_read_success"]).lower() == "true" and str(r["face_detected"]).lower() != "true" for r in rows), "processing_errors": sum(not pd.isna(r["error_message"]) and bool(str(r["error_message"]).strip()) for r in rows), "mean_ms": statistics.fmean(times) * 1000 if times else None, "median_ms": statistics.median(times) * 1000 if times else None, "p95_ms": sorted(times)[max(0, int(.95 * len(times)) - 1)] * 1000 if times else None}
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
