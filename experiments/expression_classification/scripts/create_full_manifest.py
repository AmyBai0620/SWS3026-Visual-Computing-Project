#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import StringIO
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
SPLIT_ORDER = ["train", "val", "test"]
SOURCE_SPLITS = ["train", "test"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".webp"]
IMAGE_EXTENSION_SET = set(IMAGE_EXTENSIONS)
MANIFEST_COLUMNS = ["sample_id", "class_name", "split", "relative_path", "sha256"]

EXCLUDED_TRAIN_TEST_DUPLICATES_FILENAME = "excluded_train_test_duplicates.csv"
EXCLUDED_LABEL_CONFLICTS_FILENAME = "excluded_label_conflicts.csv"
DEFAULT_DUPLICATE_AUDIT_FILENAME = "duplicate_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a leakage-safe deterministic FER-2013 full manifest."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        raise RuntimeError(
            f"Unexpected existing validation split found: {dataset_root / 'val'}"
        )

    unexpected_root_dirs = sorted(
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name not in SOURCE_SPLITS
    )
    if unexpected_root_dirs:
        raise RuntimeError(
            "Unexpected top-level directories in dataset root: "
            + ", ".join(unexpected_root_dirs)
        )


def get_repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def safe_git_commit(repo_root: Path) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={repo_root.as_posix()}",
        "rev-parse",
        "HEAD",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def sanitize_command_for_metadata(argv: list[str]) -> str:
    sanitized: list[str] = []
    for item in argv:
        try:
            if Path(item).is_absolute():
                sanitized.append("<absolute-path-omitted>")
            else:
                sanitized.append(item)
        except Exception:
            sanitized.append(item)
    return "python " + " ".join(shlex.quote(part) for part in sanitized)


def sorted_unique_join(values: list[str]) -> str:
    return ";".join(sorted(set(values)))


def scan_dataset(dataset_root: Path) -> tuple[list[dict[str, str]], dict]:
    """
    Scan every source file once.
    No file is silently skipped.
    """
    ensure_expected_layout(dataset_root)

    records: list[dict[str, str]] = []
    source_counts = {"train": 0, "test": 0}
    per_source_split_class_counts = {"train": {}, "test": {}}
    filename_map: dict[str, list[str]] = defaultdict(list)

    for split in SOURCE_SPLITS:
        split_dir = dataset_root / split
        actual_classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        missing_classes = [name for name in CLASS_ORDER if name not in actual_classes]
        unexpected_classes = [name for name in actual_classes if name not in CLASS_ORDER]

        if missing_classes:
            raise RuntimeError(
                f"{split_dir} is missing class folders: {', '.join(missing_classes)}"
            )
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
                    f"{class_dir} contains unexpected non-image files: "
                    + ", ".join(non_images[:20])
                )

            image_paths = sorted(
                path for path in all_files if path.suffix.lower() in IMAGE_EXTENSION_SET
            )
            if not image_paths:
                raise RuntimeError(f"No images found in {class_dir}")

            per_source_split_class_counts[split][class_name] = len(image_paths)
            source_counts[split] += len(image_paths)

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
                filename_map[image_path.name].append(rel)

    duplicate_filename_groups = {
        name: paths for name, paths in filename_map.items() if len(paths) > 1
    }

    summary = {
        "source_total_count": len(records),
        "source_train_count": source_counts["train"],
        "source_test_count": source_counts["test"],
        "per_source_split_class_counts": per_source_split_class_counts,
        "duplicate_filename_count": len(duplicate_filename_groups),
        "duplicate_filename_examples": dict(list(duplicate_filename_groups.items())[:20]),
    }
    return records, summary


def group_by_sha256(records: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record["sha256"]].append(record)
    return grouped


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validation_targets_from_eligible_counts(
    eligible_train_groups_by_class: dict[str, list[dict[str, object]]],
    validation_ratio: float,
) -> dict[str, int]:
    targets: dict[str, int] = {}
    for class_name in CLASS_ORDER:
        eligible_count = sum(group["size"] for group in eligible_train_groups_by_class[class_name])
        targets[class_name] = int(math.floor(eligible_count * validation_ratio))
    return targets


def deterministic_group_order(
    groups: list[dict[str, object]],
    seed: int,
    class_name: str,
) -> list[dict[str, object]]:
    """
    Deterministic order for tie-breaking during group-aware validation selection.
    The base content order is stable. A seeded pseudo-random decoration is used
    only to break otherwise arbitrary subset ties.
    """
    base = sorted(groups, key=lambda g: (g["relative_paths"][0], g["sha256"]))
    class_seed_text = f"{seed}:{class_name}"
    class_seed = int(hashlib.sha256(class_seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(class_seed)
    decorated = [(rng.random(), index, group) for index, group in enumerate(base)]
    decorated.sort(key=lambda item: (item[0], item[1]))
    return [group for _, _, group in decorated]


def choose_validation_groups_for_class(
    groups: list[dict[str, object]],
    target_count: int,
    seed: int,
    class_name: str,
) -> tuple[list[dict[str, object]], int]:
    """
    Group-aware validation selection.

    Rules:
    - groups are never broken
    - selection is deterministic
    - choose an achievable validation size closest to the target
    - if two sizes are equally close, prefer the smaller one (conservative)
    """
    if not groups or target_count <= 0:
        return [], 0

    ordered_groups = deterministic_group_order(groups, seed, class_name)
    sizes = [int(group["size"]) for group in ordered_groups]
    total = sum(sizes)

    reachable = [False] * (total + 1)
    parent_sum = [-1] * (total + 1)
    parent_group = [-1] * (total + 1)
    reachable[0] = True

    for group_index, size in enumerate(sizes):
        for current_sum in range(total - size, -1, -1):
            if reachable[current_sum] and not reachable[current_sum + size]:
                reachable[current_sum + size] = True
                parent_sum[current_sum + size] = current_sum
                parent_group[current_sum + size] = group_index

    candidate_sums = [index for index, ok in enumerate(reachable) if ok]
    best_sum = min(
        candidate_sums,
        key=lambda value: (abs(value - target_count), value > target_count, value),
    )

    chosen_group_indices: set[int] = set()
    trace_sum = best_sum
    while trace_sum != 0:
        group_index = parent_group[trace_sum]
        chosen_group_indices.add(group_index)
        trace_sum = parent_sum[trace_sum]

    chosen_groups = [ordered_groups[index] for index in sorted(chosen_group_indices)]
    achieved_count = sum(int(group["size"]) for group in chosen_groups)
    return chosen_groups, achieved_count


def build_manifest_rows(
    split_to_entries: dict[str, dict[str, list[dict[str, str]]]],
) -> list[dict[str, str]]:
    """
    Final output order:
    1. train
    2. val
    3. test

    Within each split:
    1. fixed class order
    2. sorted relative_path

    Sample IDs are regenerated after final exclusions and ordering.
    """
    rows: list[dict[str, str]] = []

    for split in SPLIT_ORDER:
        for class_name in CLASS_ORDER:
            entries = sorted(
                split_to_entries[split][class_name],
                key=lambda row: row["relative_path"],
            )
            for index_within_split_class, entry in enumerate(entries, start=1):
                sample_id = f"{split}_{class_name}_{index_within_split_class:06d}"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "class_name": class_name,
                        "split": split,
                        "relative_path": entry["relative_path"],
                        "sha256": entry["sha256"],
                    }
                )

    return rows


def manifest_csv_text(rows: list[dict[str, str]]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_manifest(path: Path, rows: list[dict[str, str]]) -> str:
    csv_text = manifest_csv_text(rows)
    path.write_text(csv_text, encoding="utf-8", newline="")
    return sha256_text(csv_text)


def load_optional_duplicate_audit_hash(metadata_dir: Path) -> str:
    audit_path = metadata_dir / DEFAULT_DUPLICATE_AUDIT_FILENAME
    if not audit_path.is_file():
        return ""
    return sha256_file(audit_path)


def main() -> int:
    args = parse_args()
    start_time = time.perf_counter()

    if not (0.0 <= args.validation_ratio < 1.0):
        print("ERROR: --validation-ratio must be in [0, 1).", file=sys.stderr)
        return 2

    output_manifest = args.output_manifest
    metadata_output = args.metadata_output
    metadata_dir = metadata_output.parent

    excluded_train_test_duplicates_path = metadata_dir / EXCLUDED_TRAIN_TEST_DUPLICATES_FILENAME
    excluded_label_conflicts_path = metadata_dir / EXCLUDED_LABEL_CONFLICTS_FILENAME

    outputs_to_protect = [
        output_manifest,
        metadata_output,
        excluded_train_test_duplicates_path,
        excluded_label_conflicts_path,
    ]

    for output_path in outputs_to_protect:
        if output_path.exists() and not args.overwrite:
            print(
                f"ERROR: Output already exists: {output_path}. Use --overwrite to replace it.",
                file=sys.stderr,
            )
            return 2

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    try:
        records, source_summary = scan_dataset(args.dataset_root)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    grouped = group_by_sha256(records)

    original_test_entries = []
    for group in grouped.values():
        original_test_entries.extend(row for row in group if row["source_split"] == "test")

    test_hashes = {
        sha256_value
        for sha256_value, group in grouped.items()
        if any(row["source_split"] == "test" for row in group)
    }

    excluded_train_test_duplicate_rows: list[dict[str, str]] = []
    excluded_train_test_duplicate_paths: set[str] = set()

    for sha256_value in sorted(grouped):
        group = sorted(grouped[sha256_value], key=lambda row: row["relative_path"])
        test_rows = [row for row in group if row["source_split"] == "test"]
        train_rows = [row for row in group if row["source_split"] == "train"]

        if not test_rows or not train_rows:
            continue

        matching_test_paths = [row["relative_path"] for row in test_rows]
        matching_test_classes = [row["class_name"] for row in test_rows]

        for train_row in train_rows:
            excluded_train_test_duplicate_paths.add(train_row["relative_path"])
            label_conflict = len({train_row["class_name"], *matching_test_classes}) > 1
            excluded_train_test_duplicate_rows.append(
                {
                    "relative_path": train_row["relative_path"],
                    "class_name": train_row["class_name"],
                    "sha256": train_row["sha256"],
                    "reason": "train_content_matches_official_test",
                    "matching_test_paths": ";".join(sorted(matching_test_paths)),
                    "matching_test_classes": ";".join(sorted(set(matching_test_classes))),
                    "label_conflict": str(label_conflict).lower(),
                }
            )

    # Remaining train files after removing any train content that also appears in test.
    remaining_train_entries = [
        row
        for row in records
        if row["source_split"] == "train"
        and row["relative_path"] not in excluded_train_test_duplicate_paths
    ]

    remaining_train_grouped = group_by_sha256(remaining_train_entries)

    excluded_label_conflict_rows: list[dict[str, str]] = []
    excluded_label_conflict_paths: set[str] = set()

    eligible_train_groups_by_class: dict[str, list[dict[str, object]]] = {
        class_name: [] for class_name in CLASS_ORDER
    }

    for sha256_value in sorted(remaining_train_grouped):
        group = sorted(remaining_train_grouped[sha256_value], key=lambda row: row["relative_path"])
        class_names = sorted({row["class_name"] for row in group})

        if len(class_names) > 1:
            excluded_label_conflict_rows.append(
                {
                    "sha256": sha256_value,
                    "relative_paths": ";".join(row["relative_path"] for row in group),
                    "class_names": ";".join(class_names),
                    "reason": "conflicting_labels_within_train_sha256_group",
                }
            )
            for row in group:
                excluded_label_conflict_paths.add(row["relative_path"])
            continue

        class_name = class_names[0]
        eligible_train_groups_by_class[class_name].append(
            {
                "sha256": sha256_value,
                "class_name": class_name,
                "entries": group,
                "relative_paths": [row["relative_path"] for row in group],
                "size": len(group),
            }
        )

    eligible_train_count = sum(
        sum(int(group["size"]) for group in eligible_train_groups_by_class[class_name])
        for class_name in CLASS_ORDER
    )

    validation_target_counts = validation_targets_from_eligible_counts(
        eligible_train_groups_by_class=eligible_train_groups_by_class,
        validation_ratio=args.validation_ratio,
    )

    validation_achieved_counts: dict[str, int] = {}
    chosen_validation_hashes: set[str] = set()

    for class_name in CLASS_ORDER:
        chosen_groups, achieved_count = choose_validation_groups_for_class(
            groups=eligible_train_groups_by_class[class_name],
            target_count=validation_target_counts[class_name],
            seed=args.seed,
            class_name=class_name,
        )
        validation_achieved_counts[class_name] = achieved_count
        chosen_validation_hashes.update(group["sha256"] for group in chosen_groups)

    final_split_entries = {
        "train": {class_name: [] for class_name in CLASS_ORDER},
        "val": {class_name: [] for class_name in CLASS_ORDER},
        "test": {class_name: [] for class_name in CLASS_ORDER},
    }

    # Put all eligible remaining train groups wholly in train or val.
    for class_name in CLASS_ORDER:
        for group in eligible_train_groups_by_class[class_name]:
            target_split = "val" if group["sha256"] in chosen_validation_hashes else "train"
            final_split_entries[target_split][class_name].extend(group["entries"])

    # Official original test is always preserved.
    for test_row in original_test_entries:
        final_split_entries["test"][test_row["class_name"]].append(test_row)

    manifest_rows = build_manifest_rows(final_split_entries)
    manifest_sha256 = write_manifest(output_manifest, manifest_rows)

    write_csv(
        excluded_train_test_duplicates_path,
        [
            "relative_path",
            "class_name",
            "sha256",
            "reason",
            "matching_test_paths",
            "matching_test_classes",
            "label_conflict",
        ],
        sorted(excluded_train_test_duplicate_rows, key=lambda row: row["relative_path"]),
    )

    write_csv(
        excluded_label_conflicts_path,
        ["sha256", "relative_paths", "class_names", "reason"],
        sorted(excluded_label_conflict_rows, key=lambda row: row["sha256"]),
    )

    per_split_class_counts = {
        split: {
            class_name: len(final_split_entries[split][class_name])
            for class_name in CLASS_ORDER
        }
        for split in SPLIT_ORDER
    }

    final_train_count = sum(per_split_class_counts["train"].values())
    final_val_count = sum(per_split_class_counts["val"].values())
    final_test_count = sum(per_split_class_counts["test"].values())
    final_total_count = final_train_count + final_val_count + final_test_count

    repo_root = get_repo_root_from_script()
    generator_script_rel = Path(__file__).resolve().relative_to(repo_root).as_posix()

    duplicate_policy = {
        "official_test_preserved": True,
        "train_files_matching_test_content_excluded": True,
        "train_sha256_groups_split_across_train_and_val": False,
        "conflicting_label_train_sha256_groups_excluded": True,
        "within_split_duplicate_content_allowed_by_default": True,
    }

    metadata = {
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "validation_target_ratio": args.validation_ratio,
        "stratified": True,
        "class_order": CLASS_ORDER,
        "dataset_root_layout": "train/<class> and test/<class>",
        "source_total_count": source_summary["source_total_count"],
        "source_train_count": source_summary["source_train_count"],
        "source_test_count": source_summary["source_test_count"],
        "eligible_train_count": eligible_train_count,
        "excluded_train_test_duplicate_count": len(excluded_train_test_duplicate_rows),
        "excluded_train_label_conflict_count": sum(
            len(group["entries"])
            for class_name in CLASS_ORDER
            for group in []
        ),
        "excluded_train_label_conflict_group_count": len(excluded_label_conflict_rows),
        "final_train_count": final_train_count,
        "final_val_count": final_val_count,
        "final_test_count": final_test_count,
        "final_total_count": final_total_count,
        "per_split_class_counts": per_split_class_counts,
        "validation_target_counts": validation_target_counts,
        "validation_achieved_counts": validation_achieved_counts,
        "duplicate_policy": duplicate_policy,
        "group_aware_split": True,
        "manifest_sha256": manifest_sha256,
        "duplicate_audit_sha256": load_optional_duplicate_audit_hash(metadata_dir),
        "git_commit": safe_git_commit(repo_root),
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "generator_script_path": generator_script_rel,
        "generation_command": sanitize_command_for_metadata(sys.argv),
        "image_extension_policy": IMAGE_EXTENSIONS,
        "sample_id_convention": "<split>_<class_name>_<zero-padded 1-based index within final deterministic split/class ordering>",
        "validation_rounding_rule": "floor(eligible_class_count * validation_ratio) separately for each class",
        "duplicate_filename_count": source_summary["duplicate_filename_count"],
        "duplicate_filename_examples": source_summary["duplicate_filename_examples"],
        "runtime_seconds": round(time.perf_counter() - start_time, 6),
        "notes": [
            "All original test files are preserved as final test entries.",
            "Any original train file whose SHA-256 appears in the original test split is excluded from both final train and final val.",
            "Remaining original-train SHA-256 groups are assigned wholly to final train or final val.",
            "Remaining original-train SHA-256 groups with conflicting class labels are excluded.",
            "Final total may be lower than the raw source total because excluded train files are documented rather than silently dropped.",
        ],
    }

    # Count excluded label-conflict files for metadata.
    excluded_label_conflict_count = 0
    for row in excluded_label_conflict_rows:
        rel_paths = [item for item in row["relative_paths"].split(";") if item]
        excluded_label_conflict_count += len(rel_paths)
    metadata["excluded_train_label_conflict_count"] = excluded_label_conflict_count

    with metadata_output.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print("Leakage-safe manifest generation completed.")
    print(f"Manifest: {output_manifest.as_posix()}")
    print(f"Metadata: {metadata_output.as_posix()}")
    print(f"Excluded train/test duplicate report: {excluded_train_test_duplicates_path.as_posix()}")
    print(f"Excluded label-conflict report: {excluded_label_conflicts_path.as_posix()}")
    print(f"Source total: {source_summary['source_total_count']}")
    print(f"Eligible train after exclusions: {eligible_train_count}")
    print(f"Excluded train files matching test content: {len(excluded_train_test_duplicate_rows)}")
    print(f"Excluded train files in conflicting-label groups: {excluded_label_conflict_count}")
    print(f"Final train: {final_train_count}")
    print(f"Final val:   {final_val_count}")
    print(f"Final test:  {final_test_count}")
    print(f"Final total: {final_total_count}")
    print(f"Manifest SHA-256: {manifest_sha256}")
    print("Validation target vs achieved counts by class:")
    for class_name in CLASS_ORDER:
        print(
            f"  {class_name:<8} target={validation_target_counts[class_name]:>4} "
            f"achieved={validation_achieved_counts[class_name]:>4}"
        )
    print(f"Runtime: {time.perf_counter() - start_time:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
