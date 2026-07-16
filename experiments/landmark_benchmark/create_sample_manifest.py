#!/usr/bin/env python
"""
Create reproducible FER-2013 sample manifests for landmark-model benchmarking.

Example:
python create_sample_manifest.py ^
  --dataset-root "E:\NUS_Visual_Computing\project\emotion\data\facial_expression_dataset\train" ^
  --output-dir ".\manifests" ^
  --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Iterable

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 35-image and 350-image FER sample manifests."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to FER train directory containing the seven class folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where manifest CSV files will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
        help="Number of images per class in the full feasibility sample.",
    )
    parser.add_argument(
        "--smoke-per-class",
        type=int,
        default=5,
        help="Number of images per class in the smoke-test subset.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def write_manifest(
    output_path: Path,
    rows: Iterable[dict[str, str | int]],
) -> None:
    fieldnames = [
        "sample_id",
        "label",
        "relative_path",
        "filename",
        "sha256",
        "seed",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if args.smoke_per_class > args.samples_per_class:
        print(
            "ERROR: --smoke-per-class cannot exceed --samples-per-class.",
            file=sys.stderr,
        )
        return 2

    if not dataset_root.is_dir():
        print(f"ERROR: dataset directory does not exist: {dataset_root}", file=sys.stderr)
        return 2

    missing_classes = [
        emotion for emotion in EMOTIONS if not (dataset_root / emotion).is_dir()
    ]
    if missing_classes:
        print(
            "ERROR: missing class folders: " + ", ".join(missing_classes),
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    full_rows: list[dict[str, str | int]] = []
    smoke_rows: list[dict[str, str | int]] = []
    class_counts: dict[str, int] = {}

    for emotion in EMOTIONS:
        class_dir = dataset_root / emotion
        images = list_images(class_dir)
        class_counts[emotion] = len(images)

        if len(images) < args.samples_per_class:
            print(
                f"ERROR: class '{emotion}' has only {len(images)} images; "
                f"{args.samples_per_class} required.",
                file=sys.stderr,
            )
            return 2

        # Sampling happens once per class. The smoke test is the first N rows
        # of the same sampled set, so the 35-image set is contained in the
        # 350-image set.
        selected = rng.sample(images, args.samples_per_class)

        class_rows: list[dict[str, str | int]] = []
        for index, image_path in enumerate(selected, start=1):
            relative_path = image_path.relative_to(dataset_root).as_posix()
            row = {
                "sample_id": f"{emotion}_{index:03d}",
                "label": emotion,
                "relative_path": relative_path,
                "filename": image_path.name,
                "sha256": sha256_file(image_path),
                "seed": args.seed,
            }
            class_rows.append(row)

        full_rows.extend(class_rows)
        smoke_rows.extend(class_rows[: args.smoke_per_class])

    full_manifest = output_dir / "sample_manifest_350.csv"
    smoke_manifest = output_dir / "sample_manifest_35.csv"
    metadata_path = output_dir / "manifest_metadata.json"

    write_manifest(full_manifest, full_rows)
    write_manifest(smoke_manifest, smoke_rows)

    metadata = {
        "dataset_root_used_for_generation": str(dataset_root),
        "seed": args.seed,
        "classes": EMOTIONS,
        "available_images_per_class": class_counts,
        "samples_per_class": args.samples_per_class,
        "smoke_per_class": args.smoke_per_class,
        "full_manifest_rows": len(full_rows),
        "smoke_manifest_rows": len(smoke_rows),
        "notes": [
            "The 35-image smoke-test set is a subset of the 350-image set.",
            "Teammates should use relative_path and their own local dataset root.",
            "SHA-256 hashes verify that all teammates use identical source images.",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Manifest generation completed.")
    print(f"Dataset root: {dataset_root}")
    print(f"Seed: {args.seed}")
    print(f"35-image manifest:  {smoke_manifest}")
    print(f"350-image manifest: {full_manifest}")
    print(f"Metadata:           {metadata_path}")
    print()
    print("Rows per class:")
    for emotion in EMOTIONS:
        print(
            f"  {emotion:<8} smoke={args.smoke_per_class:>2} "
            f"full={args.samples_per_class:>2} "
            f"available={class_counts[emotion]}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
