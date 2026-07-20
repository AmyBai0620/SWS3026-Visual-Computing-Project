#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

CLASS_ORDER = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]
SOURCE_SPLITS = ["train", "test"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".webp"]
IMAGE_EXTENSION_SET = set(IMAGE_EXTENSIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit duplicate image content in FER-2013 using SHA-256 grouping."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-groups-csv", type=Path, required=True)
    parser.add_argument("--output-conflicts-csv", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def relative_posix_path(path: Path, dataset_root: Path) -> str:
    rel = path.relative_to(dataset_root).as_posix()
    if "\\" in rel:
        raise RuntimeError(f"Backslash found in relative path: {rel}")
    if Path(rel).is_absolute():
        raise RuntimeError(f"Absolute path found where relative path was expected: {rel}")
    return rel


def verify_image_readable_and_size(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (48, 48):
                raise RuntimeError(
                    f"Expected 48x48 image, found {image.size[0]}x{image.size[1]}"
                )
    except Exception as exc:
        raise RuntimeError(f"Unreadable or corrupted image: {path}") from exc


def ensure_expected_layout(dataset_root: Path) -> None:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    for split in SOURCE_SPLITS:
        if not (dataset_root / split).is_dir():
            raise RuntimeError(f"Missing split directory: {dataset_root / split}")

    if (dataset_root / "val").exists():
        raise RuntimeError(f"Unexpected existing validation split found: {dataset_root / 'val'}")

    unexpected_root_dirs = sorted(
        path.name for path in dataset_root.iterdir() if path.is_dir() and path.name not in SOURCE_SPLITS
    )
    if unexpected_root_dirs:
        raise RuntimeError(
            "Unexpected top-level directories in dataset root: " + ", ".join(unexpected_root_dirs)
        )


def scan_dataset(dataset_root: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    """
    Scan all source images and compute SHA-256 once per file.
    Return one record per file.
    """
    ensure_expected_layout(dataset_root)

    records: list[dict[str, str]] = []
    counts = {"train": 0, "test": 0}

    for split in SOURCE_SPLITS:
        split_dir = dataset_root / split
        actual_classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        missing_classes = [name for name in CLASS_ORDER if name not in actual_classes]
        unexpected_classes = [name for name in actual_classes if name not in CLASS_ORDER]

        if missing_classes:
            raise RuntimeError(f"{split_dir} is missing class folders: {', '.join(missing_classes)}")
        if unexpected_classes:
            raise RuntimeError(
                f"{split_dir} has unexpected class folders: {', '.join(unexpected_classes)}"
            )

        for class_name in CLASS_ORDER:
            class_dir = split_dir / class_name
            all_files = sorted(path for path in class_dir.iterdir() if path.is_file())

            non_images = [path.name for path in all_files if path.suffix.lower() not in IMAGE_EXTENSION_SET]
            if non_images:
                raise RuntimeError(
                    f"{class_dir} contains unexpected non-image files: " + ", ".join(non_images[:20])
                )

            image_paths = sorted(path for path in all_files if path.suffix.lower() in IMAGE_EXTENSION_SET)
            if not image_paths:
                raise RuntimeError(f"No images found in {class_dir}")

            for image_path in image_paths:
                verify_image_readable_and_size(image_path)
                rel = relative_posix_path(image_path, dataset_root)
                file_sha256 = sha256_file(image_path)

                records.append(
                    {
                        "source_split": split,
                        "class_name": class_name,
                        "relative_path": rel,
                        "filename": image_path.name,
                        "sha256": file_sha256,
                    }
                )
                counts[split] += 1

    return records, counts


def write_groups_csv(path: Path, grouped: dict[str, list[dict[str, str]]]) -> None:
    fieldnames = [
        "sha256",
        "group_size",
        "source_split",
        "class_name",
        "relative_path",
        "spans_source_splits",
        "has_label_conflict",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for sha256_value in sorted(grouped):
            group = sorted(grouped[sha256_value], key=lambda row: row["relative_path"])
            source_splits = {row["source_split"] for row in group}
            class_names = {row["class_name"] for row in group}
            spans_source_splits = len(source_splits) > 1
            has_label_conflict = len(class_names) > 1

            for row in group:
                writer.writerow(
                    {
                        "sha256": sha256_value,
                        "group_size": len(group),
                        "source_split": row["source_split"],
                        "class_name": row["class_name"],
                        "relative_path": row["relative_path"],
                        "spans_source_splits": str(spans_source_splits).lower(),
                        "has_label_conflict": str(has_label_conflict).lower(),
                    }
                )


def write_conflicts_csv(path: Path, grouped: dict[str, list[dict[str, str]]]) -> None:
    fieldnames = ["sha256", "group_size", "source_splits", "class_names", "relative_paths"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for sha256_value in sorted(grouped):
            group = sorted(grouped[sha256_value], key=lambda row: row["relative_path"])
            class_names = sorted({row["class_name"] for row in group})
            if len(class_names) <= 1:
                continue

            writer.writerow(
                {
                    "sha256": sha256_value,
                    "group_size": len(group),
                    "source_splits": ";".join(sorted({row["source_split"] for row in group})),
                    "class_names": ";".join(class_names),
                    "relative_paths": ";".join(row["relative_path"] for row in group),
                }
            )


def main() -> int:
    args = parse_args()
    start_time = time.perf_counter()

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_groups_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_conflicts_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        records, counts = scan_dataset(args.dataset_root)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    filename_map: dict[str, list[str]] = defaultdict(list)

    for record in records:
        grouped[record["sha256"]].append(record)
        filename_map[record["filename"]].append(record["relative_path"])

    duplicate_groups = {sha256_value: group for sha256_value, group in grouped.items() if len(group) > 1}

    only_train_groups = 0
    only_test_groups = 0
    spanning_train_test_groups = 0
    one_label_groups = 0
    conflicting_label_groups = 0
    split_and_label_spanning_groups = 0
    train_files_also_in_test = 0
    test_files_also_in_train = 0
    multiplicity_distribution: Counter[int] = Counter()

    for sha256_value, group in duplicate_groups.items():
        source_splits = {row["source_split"] for row in group}
        class_names = {row["class_name"] for row in group}
        multiplicity_distribution[len(group)] += 1

        if source_splits == {"train"}:
            only_train_groups += 1
        elif source_splits == {"test"}:
            only_test_groups += 1
        elif source_splits == {"train", "test"}:
            spanning_train_test_groups += 1

        if len(class_names) == 1:
            one_label_groups += 1
        else:
            conflicting_label_groups += 1

        if len(source_splits) > 1 and len(class_names) > 1:
            split_and_label_spanning_groups += 1

        if source_splits == {"train", "test"}:
            train_files_also_in_test += sum(1 for row in group if row["source_split"] == "train")
            test_files_also_in_train += sum(1 for row in group if row["source_split"] == "test")

    duplicate_filename_groups = {name: locations for name, locations in filename_map.items() if len(locations) > 1}

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_layout": "train/<class>/*.jpg and test/<class>/*.jpg",
        "class_order": CLASS_ORDER,
        "image_extension_policy": IMAGE_EXTENSIONS,
        "total_files": len(records),
        "source_train_count": counts["train"],
        "source_test_count": counts["test"],
        "unique_content_hashes": len(grouped),
        "duplicate_hash_group_count": len(duplicate_groups),
        "files_in_duplicate_groups": sum(len(group) for group in duplicate_groups.values()),
        "duplicate_groups_only_within_original_train": only_train_groups,
        "duplicate_groups_only_within_original_test": only_test_groups,
        "duplicate_groups_spanning_original_train_and_test": spanning_train_test_groups,
        "duplicate_groups_with_one_class_label": one_label_groups,
        "duplicate_groups_with_conflicting_class_labels": conflicting_label_groups,
        "duplicate_groups_spanning_both_source_split_and_class_labels": split_and_label_spanning_groups,
        "train_files_whose_content_also_appears_in_test": train_files_also_in_test,
        "test_files_whose_content_also_appears_in_train": test_files_also_in_train,
        "duplicate_multiplicity_distribution": {
            str(size): count for size, count in sorted(multiplicity_distribution.items())
        },
        "duplicate_filename_count": len(duplicate_filename_groups),
        "duplicate_filename_examples": dict(list(duplicate_filename_groups.items())[:20]),
        "runtime_seconds": round(time.perf_counter() - start_time, 6),
    }

    with args.output_report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    write_groups_csv(args.output_groups_csv, grouped)
    write_conflicts_csv(args.output_conflicts_csv, grouped)

    print("Duplicate image audit completed.")
    print(f"Report: {args.output_report.as_posix()}")
    print(f"Groups CSV: {args.output_groups_csv.as_posix()}")
    print(f"Conflicts CSV: {args.output_conflicts_csv.as_posix()}")
    print(f"Total files: {report['total_files']}")
    print(f"Unique content hashes: {report['unique_content_hashes']}")
    print(f"Duplicate hash groups: {report['duplicate_hash_group_count']}")
    print(f"Files in duplicate groups: {report['files_in_duplicate_groups']}")
    print(
        "Train/test spanning duplicate groups: "
        f"{report['duplicate_groups_spanning_original_train_and_test']}"
    )
    print(
        "Conflicting-label duplicate groups: "
        f"{report['duplicate_groups_with_conflicting_class_labels']}"
    )
    print(f"Runtime: {time.perf_counter() - start_time:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())