#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
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

SAMPLE_ID_PATTERN = re.compile(
    r"^(train|val|test)_(angry|disgust|fear|happy|neutral|sad|surprise)_(\d{6})$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a leakage-safe deterministic FER-2013 full manifest."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--check-determinism", action="store_true")
    parser.add_argument(
        "--deduplicate-within-split",
        action="store_true",
        help=(
            "Report projected counts after collapsing identical content within the same split "
            "and label. This does not modify the manifest."
        ),
    )
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
            image.verify()
        with Image.open(path) as image:
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


def scan_dataset(dataset_root: Path) -> tuple[list[dict[str, str]], dict]:
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


def build_expected_state(
    source_records: list[dict[str, str]],
    validation_ratio: float,
    seed: int,
) -> dict:
    grouped = group_by_sha256(source_records)

    original_test_entries = []
    for group in grouped.values():
        original_test_entries.extend(row for row in group if row["source_split"] == "test")

    excluded_train_test_duplicates: list[dict[str, str]] = []
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
            excluded_train_test_duplicates.append(
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

    remaining_train_entries = [
        row
        for row in source_records
        if row["source_split"] == "train"
        and row["relative_path"] not in excluded_train_test_duplicate_paths
    ]

    remaining_train_grouped = group_by_sha256(remaining_train_entries)

    excluded_label_conflicts: list[dict[str, str]] = []
    eligible_train_groups_by_class: dict[str, list[dict[str, object]]] = {
        class_name: [] for class_name in CLASS_ORDER
    }

    for sha256_value in sorted(remaining_train_grouped):
        group = sorted(remaining_train_grouped[sha256_value], key=lambda row: row["relative_path"])
        class_names = sorted({row["class_name"] for row in group})

        if len(class_names) > 1:
            excluded_label_conflicts.append(
                {
                    "sha256": sha256_value,
                    "relative_paths": ";".join(row["relative_path"] for row in group),
                    "class_names": ";".join(class_names),
                    "reason": "conflicting_labels_within_train_sha256_group",
                }
            )
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

    validation_target_counts = validation_targets_from_eligible_counts(
        eligible_train_groups_by_class=eligible_train_groups_by_class,
        validation_ratio=validation_ratio,
    )

    validation_achieved_counts: dict[str, int] = {}
    chosen_validation_hashes: set[str] = set()

    for class_name in CLASS_ORDER:
        chosen_groups, achieved_count = choose_validation_groups_for_class(
            groups=eligible_train_groups_by_class[class_name],
            target_count=validation_target_counts[class_name],
            seed=seed,
            class_name=class_name,
        )
        validation_achieved_counts[class_name] = achieved_count
        chosen_validation_hashes.update(group["sha256"] for group in chosen_groups)

    final_split_entries = {
        "train": {class_name: [] for class_name in CLASS_ORDER},
        "val": {class_name: [] for class_name in CLASS_ORDER},
        "test": {class_name: [] for class_name in CLASS_ORDER},
    }

    for class_name in CLASS_ORDER:
        for group in eligible_train_groups_by_class[class_name]:
            split_name = "val" if group["sha256"] in chosen_validation_hashes else "train"
            final_split_entries[split_name][class_name].extend(group["entries"])

    for test_row in original_test_entries:
        final_split_entries["test"][test_row["class_name"]].append(test_row)

    expected_manifest_rows: list[dict[str, str]] = []
    for split in SPLIT_ORDER:
        for class_name in CLASS_ORDER:
            entries = sorted(
                final_split_entries[split][class_name],
                key=lambda row: row["relative_path"],
            )
            for index_within_split_class, entry in enumerate(entries, start=1):
                sample_id = f"{split}_{class_name}_{index_within_split_class:06d}"
                expected_manifest_rows.append(
                    {
                        "sample_id": sample_id,
                        "class_name": class_name,
                        "split": split,
                        "relative_path": entry["relative_path"],
                        "sha256": entry["sha256"],
                    }
                )

    per_split_class_counts = {
        split: {
            class_name: len(final_split_entries[split][class_name])
            for class_name in CLASS_ORDER
        }
        for split in SPLIT_ORDER
    }

    excluded_label_conflict_count = 0
    for row in excluded_label_conflicts:
        excluded_label_conflict_count += len(
            [item for item in row["relative_paths"].split(";") if item]
        )

    return {
        "expected_manifest_rows": expected_manifest_rows,
        "expected_excluded_train_test_duplicates": sorted(
            excluded_train_test_duplicates, key=lambda row: row["relative_path"]
        ),
        "expected_excluded_label_conflicts": sorted(
            excluded_label_conflicts, key=lambda row: row["sha256"]
        ),
        "per_split_class_counts": per_split_class_counts,
        "validation_target_counts": validation_target_counts,
        "validation_achieved_counts": validation_achieved_counts,
        "excluded_train_test_duplicate_count": len(excluded_train_test_duplicates),
        "excluded_train_label_conflict_count": excluded_label_conflict_count,
    }


def load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return rows, fieldnames


def print_group(title: str, items: list[str], max_examples: int = 10) -> None:
    print(f"\n[{title}]")
    if not items:
        print("None")
        return
    for item in items[:max_examples]:
        print(f"- {item}")
    if len(items) > max_examples:
        print(f"- ... {len(items) - max_examples} additional entries omitted; see CSV/JSON reports for complete details.")


def main() -> int:
    args = parse_args()
    start_time = time.perf_counter()

    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    if not args.dataset_root.is_dir():
        print(f"ERROR: dataset root not found: {args.dataset_root}", file=sys.stderr)
        return 2
    if not args.manifest.is_file():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    if not args.metadata.is_file():
        print(f"ERROR: metadata not found: {args.metadata}", file=sys.stderr)
        return 2

    excluded_train_test_duplicates_path = args.metadata.parent / EXCLUDED_TRAIN_TEST_DUPLICATES_FILENAME
    excluded_label_conflicts_path = args.metadata.parent / EXCLUDED_LABEL_CONFLICTS_FILENAME

    if not excluded_train_test_duplicates_path.is_file():
        print(
            f"ERROR: exclusion report not found: {excluded_train_test_duplicates_path}",
            file=sys.stderr,
        )
        return 2
    if not excluded_label_conflicts_path.is_file():
        print(
            f"ERROR: exclusion report not found: {excluded_label_conflicts_path}",
            file=sys.stderr,
        )
        return 2

    try:
        source_records, source_summary = scan_dataset(args.dataset_root)
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        manifest_rows, manifest_fieldnames = load_csv(args.manifest)
        excluded_tt_rows, excluded_tt_fieldnames = load_csv(excluded_train_test_duplicates_path)
        excluded_conflict_rows, excluded_conflict_fieldnames = load_csv(excluded_label_conflicts_path)
        expected_state = build_expected_state(
            source_records=source_records,
            validation_ratio=float(metadata["validation_ratio"]),
            seed=int(metadata["seed"]),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Exact manifest schema.
    if manifest_fieldnames != MANIFEST_COLUMNS:
        errors.append(
            f"Manifest columns must be exactly {MANIFEST_COLUMNS}, found {manifest_fieldnames}"
        )

    expected_tt_fields = [
        "relative_path",
        "class_name",
        "sha256",
        "reason",
        "matching_test_paths",
        "matching_test_classes",
        "label_conflict",
    ]
    if excluded_tt_fieldnames != expected_tt_fields:
        errors.append(
            f"{EXCLUDED_TRAIN_TEST_DUPLICATES_FILENAME} columns must be exactly {expected_tt_fields}, found {excluded_tt_fieldnames}"
        )

    expected_conflict_fields = ["sha256", "relative_paths", "class_names", "reason"]
    if excluded_conflict_fieldnames != expected_conflict_fields:
        errors.append(
            f"{EXCLUDED_LABEL_CONFLICTS_FILENAME} columns must be exactly {expected_conflict_fields}, found {excluded_conflict_fieldnames}"
        )

    # Basic manifest properties.
    sample_ids = [row["sample_id"] for row in manifest_rows]
    relative_paths = [row["relative_path"] for row in manifest_rows]

    duplicate_sample_ids = [item for item, count in Counter(sample_ids).items() if count > 1]
    if duplicate_sample_ids:
        errors.append(f"Duplicate sample_id values found: {duplicate_sample_ids[:20]}")

    duplicate_relative_paths = [item for item, count in Counter(relative_paths).items() if count > 1]
    if duplicate_relative_paths:
        errors.append(f"Duplicate relative_path values found: {duplicate_relative_paths[:20]}")

    invalid_classes = sorted(
        set(row["class_name"] for row in manifest_rows if row["class_name"] not in CLASS_ORDER)
    )
    if invalid_classes:
        errors.append(f"Invalid class_name values found: {invalid_classes}")

    invalid_splits = sorted(
        set(row["split"] for row in manifest_rows if row["split"] not in SPLIT_ORDER)
    )
    if invalid_splits:
        errors.append(f"Invalid split values found: {invalid_splits}")

    bad_sample_ids = [sid for sid in sample_ids if not SAMPLE_ID_PATTERN.match(sid)]
    if bad_sample_ids:
        errors.append(
            f"Sample IDs do not match the documented convention: {bad_sample_ids[:20]}"
        )

    absolute_or_nonportable_paths = [
        path_text
        for path_text in relative_paths
        if Path(path_text).is_absolute() or "\\" in path_text
    ]
    if absolute_or_nonportable_paths:
        errors.append(
            f"Absolute or non-portable relative_path values found: {absolute_or_nonportable_paths[:20]}"
        )

    # Exclusion files must match the expected state exactly.
    expected_tt_rows = expected_state["expected_excluded_train_test_duplicates"]
    expected_conflict_rows = expected_state["expected_excluded_label_conflicts"]

    if expected_tt_rows != excluded_tt_rows:
        errors.append(
            f"{EXCLUDED_TRAIN_TEST_DUPLICATES_FILENAME} does not match the expected exclusions derived from the source dataset."
        )

    if expected_conflict_rows != excluded_conflict_rows:
        errors.append(
            f"{EXCLUDED_LABEL_CONFLICTS_FILENAME} does not match the expected exclusions derived from the source dataset."
        )

    # Deterministic manifest rows should match exactly.
    if args.check_determinism:
        expected_manifest_rows = expected_state["expected_manifest_rows"]
        if expected_manifest_rows != manifest_rows:
            errors.append(
                "Manifest does not match the deterministic regeneration from the source dataset."
            )

    # File existence, readability, hash checks, split-source consistency.
    manifest_split_path_sets = {"train": set(), "val": set(), "test": set()}
    manifest_sha_to_split_set: defaultdict[str, set[str]] = defaultdict(set)
    manifest_sha_to_split_label_set: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    per_split_class_counts = {
        "train": {name: 0 for name in CLASS_ORDER},
        "val": {name: 0 for name in CLASS_ORDER},
        "test": {name: 0 for name in CLASS_ORDER},
    }

    unreadable_count = 0
    missing_count = 0
    sha_mismatch_count = 0

    for row in manifest_rows:
        sample_id = row["sample_id"]
        class_name = row["class_name"]
        split = row["split"]
        relative_path = row["relative_path"]
        sha256_value = row["sha256"]

        per_split_class_counts[split][class_name] += 1
        manifest_split_path_sets[split].add(relative_path)
        manifest_sha_to_split_set[sha256_value].add(split)
        manifest_sha_to_split_label_set[sha256_value].add((split, class_name))

        full_path = args.dataset_root / Path(relative_path)
        if not full_path.is_file():
            missing_count += 1
            errors.append(f"Referenced file does not exist: {relative_path}")
            continue

        try:
            verify_image_readable_and_size(full_path)
        except Exception as exc:
            unreadable_count += 1
            errors.append(str(exc))
            continue

        actual_sha = sha256_file(full_path)
        if actual_sha != sha256_value:
            sha_mismatch_count += 1
            errors.append(f"SHA-256 mismatch for {relative_path}")

        parts = Path(relative_path).parts
        if len(parts) != 3:
            errors.append(f"Unexpected relative path layout: {relative_path}")
            continue

        source_split, folder_class_name, _filename = parts

        if folder_class_name != class_name:
            errors.append(
                f"class_name does not match folder name for {sample_id}: "
                f"class_name={class_name}, folder={folder_class_name}"
            )

        if split == "test" and source_split != "test":
            errors.append(
                f"Final test row points outside original test split: {relative_path}"
            )

        if split in ("train", "val") and source_split != "train":
            errors.append(
                f"Final {split} row points outside original train split: {relative_path}"
            )

    # No path overlap.
    if manifest_split_path_sets["train"] & manifest_split_path_sets["val"]:
        errors.append("Train and val contain overlapping relative paths.")
    if manifest_split_path_sets["train"] & manifest_split_path_sets["test"]:
        errors.append("Train and test contain overlapping relative paths.")
    if manifest_split_path_sets["val"] & manifest_split_path_sets["test"]:
        errors.append("Val and test contain overlapping relative paths.")

    # Cross-split duplicate content is always an error.
    duplicate_sha_across_multiple_final_splits = {
        sha_value: sorted(splits)
        for sha_value, splits in manifest_sha_to_split_set.items()
        if len(splits) > 1
    }
    if duplicate_sha_across_multiple_final_splits:
        errors.append(
            f"Duplicate image content hashes found across final splits: {len(duplicate_sha_across_multiple_final_splits)}"
        )
        for sha_value, splits in list(duplicate_sha_across_multiple_final_splits.items())[:20]:
            errors.append(f"SHA {sha_value} appears across final splits: {splits}")

    # Conflicting labels among train/val entries are an error because those
    # source groups must have been excluded.  Test-only conflicts are retained
    # intentionally: the policy preserves every original official-test file.
    included_sha_conflicting_labels = {}
    test_only_conflicting_labels = {}
    for sha_value, split_label_pairs in manifest_sha_to_split_label_set.items():
        class_names = {class_name for _split, class_name in split_label_pairs}
        if len(class_names) > 1:
            splits = {split for split, _class_name in split_label_pairs}
            if splits == {"test"}:
                test_only_conflicting_labels[sha_value] = sorted(class_names)
            else:
                included_sha_conflicting_labels[sha_value] = sorted(class_names)

    if included_sha_conflicting_labels:
        errors.append(
            f"Included SHA-256 groups with conflicting labels found: {len(included_sha_conflicting_labels)}"
        )
        for sha_value, classes in list(included_sha_conflicting_labels.items())[:20]:
            errors.append(f"SHA {sha_value} has conflicting included labels: {classes}")

    if test_only_conflicting_labels:
        warnings.append(
            "Official test contains conflicting-label SHA-256 groups preserved by policy: "
            f"{len(test_only_conflicting_labels)}"
        )
        for sha_value, classes in list(test_only_conflicting_labels.items())[:10]:
            warnings.append(f"Test-only SHA {sha_value} has conflicting labels: {classes}")

    # Same-split duplicates are allowed when labels match, but warn.
    same_split_duplicate_groups = {}
    for sha_value, split_label_pairs in manifest_sha_to_split_label_set.items():
        splits = {split for split, _class_name in split_label_pairs}
        if len(splits) == 1:
            same_split_duplicate_groups[sha_value] = sorted(split_label_pairs)

    same_split_duplicate_warnings = 0
    for sha_value, split_label_pairs in same_split_duplicate_groups.items():
        count = sum(1 for row in manifest_rows if row["sha256"] == sha_value)
        if count > 1:
            same_split_duplicate_warnings += 1
            split_name = split_label_pairs[0][0]
            class_names = sorted({class_name for _split, class_name in split_label_pairs})
            warnings.append(
                f"Same-split duplicate content kept in final {split_name}: SHA {sha_value} count={count} classes={class_names}"
            )

    if same_split_duplicate_warnings:
        warnings.append(
            "Keeping multiple identical files in the same training split can overweight those samples during learning."
        )

    if args.deduplicate_within_split:
        projected_counts = {"train": 0, "val": 0, "test": 0}
        seen = {"train": set(), "val": set(), "test": set()}
        for row in manifest_rows:
            split = row["split"]
            key = (row["sha256"], row["class_name"])
            if key not in seen[split]:
                seen[split].add(key)
                projected_counts[split] += 1

        info.append(
            "Projected counts after optional within-split deduplication "
            f"(report-only): train={projected_counts['train']} "
            f"val={projected_counts['val']} test={projected_counts['test']}"
        )
        warnings.append(
            "--deduplicate-within-split is report-only here. It does not modify the manifest."
        )

    # Source files must reconcile exactly with final manifest + exclusions.
    source_path_set = {row["relative_path"] for row in source_records}
    manifest_path_set = {row["relative_path"] for row in manifest_rows}
    excluded_tt_path_set = {row["relative_path"] for row in excluded_tt_rows}

    excluded_conflict_path_set: set[str] = set()
    for row in excluded_conflict_rows:
        for rel in row["relative_paths"].split(";"):
            if rel:
                excluded_conflict_path_set.add(rel)

    union_all = manifest_path_set | excluded_tt_path_set | excluded_conflict_path_set
    if union_all != source_path_set:
        missing_from_accounting = sorted(source_path_set - union_all)
        extra_in_accounting = sorted(union_all - source_path_set)
        if missing_from_accounting:
            errors.append(
                f"Source files not accounted for by final manifest + exclusions: {len(missing_from_accounting)}"
            )
            for rel in missing_from_accounting[:20]:
                errors.append(f"Unaccounted source file: {rel}")
        if extra_in_accounting:
            errors.append(
                f"Accounting contains non-source paths: {len(extra_in_accounting)}"
            )
            for rel in extra_in_accounting[:20]:
                errors.append(f"Unexpected accounted path: {rel}")

    if manifest_path_set & excluded_tt_path_set:
        errors.append("Some files appear in both the final manifest and train/test-duplicate exclusions.")
    if manifest_path_set & excluded_conflict_path_set:
        errors.append("Some files appear in both the final manifest and label-conflict exclusions.")
    if excluded_tt_path_set & excluded_conflict_path_set:
        errors.append("Some files appear in both exclusion reports.")

    # Metadata checks.
    split_totals = {
        "train": sum(per_split_class_counts["train"].values()),
        "val": sum(per_split_class_counts["val"].values()),
        "test": sum(per_split_class_counts["test"].values()),
    }
    final_total_count = split_totals["train"] + split_totals["val"] + split_totals["test"]

    if metadata.get("source_total_count") != source_summary["source_total_count"]:
        errors.append("Metadata source_total_count mismatch.")
    if metadata.get("source_train_count") != source_summary["source_train_count"]:
        errors.append("Metadata source_train_count mismatch.")
    if metadata.get("source_test_count") != source_summary["source_test_count"]:
        errors.append("Metadata source_test_count mismatch.")
    if metadata.get("final_train_count") != split_totals["train"]:
        errors.append("Metadata final_train_count mismatch.")
    if metadata.get("final_val_count") != split_totals["val"]:
        errors.append("Metadata final_val_count mismatch.")
    if metadata.get("final_test_count") != split_totals["test"]:
        errors.append("Metadata final_test_count mismatch.")
    if metadata.get("final_total_count") != final_total_count:
        errors.append("Metadata final_total_count mismatch.")
    if metadata.get("per_split_class_counts") != per_split_class_counts:
        errors.append("Metadata per_split_class_counts mismatch.")

    if metadata.get("excluded_train_test_duplicate_count") != len(excluded_tt_rows):
        errors.append("Metadata excluded_train_test_duplicate_count mismatch.")

    expected_conflict_excluded_count = 0
    for row in excluded_conflict_rows:
        expected_conflict_excluded_count += len([item for item in row["relative_paths"].split(";") if item])
    if metadata.get("excluded_train_label_conflict_count") != expected_conflict_excluded_count:
        errors.append("Metadata excluded_train_label_conflict_count mismatch.")

    if metadata.get("validation_target_counts") != expected_state["validation_target_counts"]:
        errors.append("Metadata validation_target_counts mismatch.")
    if metadata.get("validation_achieved_counts") != expected_state["validation_achieved_counts"]:
        errors.append("Metadata validation_achieved_counts mismatch.")

    manifest_text = args.manifest.read_text(encoding="utf-8")
    manifest_sha = sha256_text(manifest_text)
    if metadata.get("manifest_sha256") != manifest_sha:
        errors.append("Metadata manifest_sha256 mismatch.")

    duplicate_audit_path = args.metadata.parent / "duplicate_audit.json"
    if duplicate_audit_path.is_file():
        actual_duplicate_audit_sha = sha256_file(duplicate_audit_path)
        recorded_duplicate_audit_sha = metadata.get("duplicate_audit_sha256", "")
        if recorded_duplicate_audit_sha and recorded_duplicate_audit_sha != actual_duplicate_audit_sha:
            errors.append("Metadata duplicate_audit_sha256 mismatch.")

    # Informational notes.
    if source_summary["duplicate_filename_count"] > 0:
        warnings.append(
            f"Duplicate filenames across different folders detected: {source_summary['duplicate_filename_count']}"
        )
        for name, paths in source_summary["duplicate_filename_examples"].items():
            warnings.append(f"Filename '{name}' appears in: {paths}")

    info.extend(
        [
            f"Manifest: {args.manifest.as_posix()}",
            f"Metadata: {args.metadata.as_posix()}",
            f"Excluded train/test duplicates: {excluded_train_test_duplicates_path.as_posix()}",
            f"Excluded label conflicts: {excluded_label_conflicts_path.as_posix()}",
            f"Source total: {source_summary['source_total_count']}",
            f"Final train: {split_totals['train']}",
            f"Final val: {split_totals['val']}",
            f"Final test: {split_totals['test']}",
            f"Final total: {final_total_count}",
            f"Unreadable files: {0 if not unreadable_count else unreadable_count}",
            f"Missing files: {missing_count}",
            f"SHA-256 mismatches: {sha_mismatch_count}",
            f"Manifest SHA-256: {manifest_sha}",
        ]
    )

    print_group("Info", info)

    print("\n[Per-class counts]")
    for class_name in CLASS_ORDER:
        print(
            f"{class_name:<8} "
            f"train={per_split_class_counts['train'][class_name]:>4} "
            f"val={per_split_class_counts['val'][class_name]:>4} "
            f"test={per_split_class_counts['test'][class_name]:>4}"
        )

    print_group("Warnings", warnings)
    print_group("Errors", errors)

    elapsed = time.perf_counter() - start_time
    if errors:
        print(f"\nValidation FAILED in {elapsed:.2f}s")
        return 1

    print(f"\nValidation PASSED in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
