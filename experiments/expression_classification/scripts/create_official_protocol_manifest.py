#!/usr/bin/env python
"""Create the instructor-preserving, group-aware FER official-protocol manifest."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from create_full_manifest import (
    CLASS_ORDER,
    SPLIT_ORDER,
    build_manifest_rows,
    choose_validation_groups_for_class,
    scan_dataset,
    sha256_file,
    validation_targets_from_eligible_counts,
    write_csv,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.validation_ratio < 1.0:
        raise SystemExit("--validation-ratio must be in [0, 1).")

    warning_csv = args.metadata_output.parent / "official_protocol_warnings.csv"
    outputs = [args.output_manifest, args.metadata_output, warning_csv]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise SystemExit("An output already exists; use --overwrite to replace it.")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)

    records, source_summary = scan_dataset(args.dataset_root)
    train_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        if row["source_split"] == "train":
            train_groups[row["sha256"]].append(row)

    eligible_by_class: dict[str, list[dict[str, object]]] = {name: [] for name in CLASS_ORDER}
    conflict_groups: list[dict[str, str]] = []
    for sha, rows in sorted(train_groups.items()):
        rows = sorted(rows, key=lambda row: row["relative_path"])
        labels = sorted({row["class_name"] for row in rows})
        if len(labels) != 1:
            conflict_groups.append({
                "warning_type": "original_train_conflicting_label_sha256_group",
                "sha256": sha,
                "relative_paths": ";".join(row["relative_path"] for row in rows),
                "class_names": ";".join(labels),
            })
            continue
        eligible_by_class[labels[0]].append({
            "sha256": sha, "entries": rows,
            "relative_paths": [row["relative_path"] for row in rows], "size": len(rows),
        })

    # Targets are defined from every original-train file.  Conflicting-label
    # groups are preserved (and deterministically remain in train), so they do
    # not silently reduce the instructor protocol's requested validation size.
    original_train_counts = {
        class_name: sum(
            row["source_split"] == "train" and row["class_name"] == class_name
            for row in records
        )
        for class_name in CLASS_ORDER
    }
    targets = {name: int(original_train_counts[name] * args.validation_ratio) for name in CLASS_ORDER}
    chosen_hashes: set[str] = set()
    achieved: dict[str, int] = {}
    for class_name in CLASS_ORDER:
        selected, achieved[class_name] = choose_validation_groups_for_class(
            eligible_by_class[class_name], targets[class_name], args.seed, class_name
        )
        chosen_hashes.update(group["sha256"] for group in selected)

    final = {split: {name: [] for name in CLASS_ORDER} for split in SPLIT_ORDER}
    for sha, rows in train_groups.items():
        split = "val" if sha in chosen_hashes else "train"
        for row in rows:
            final[split][row["class_name"]].append(row)
    for row in records:
        if row["source_split"] == "test":
            final["test"][row["class_name"]].append(row)

    rows = build_manifest_rows(final)
    manifest_sha = write_manifest(args.output_manifest, rows)
    write_csv(warning_csv, ["warning_type", "sha256", "relative_paths", "class_names"], conflict_groups)
    train_test_groups = []
    for sha, grouped_rows in sorted(_group_by_sha(records).items()):
        source_splits = {row["source_split"] for row in grouped_rows}
        if source_splits == {"train", "test"}:
            train_test_groups.append({
                "warning_type": "inherited_train_test_sha256_overlap", "sha256": sha,
                "relative_paths": ";".join(sorted(row["relative_path"] for row in grouped_rows)),
                "class_names": ";".join(sorted({row["class_name"] for row in grouped_rows})),
            })
    if train_test_groups:
        with warning_csv.open("a", encoding="utf-8", newline="") as handle:
            import csv
            csv.DictWriter(handle, fieldnames=["warning_type", "sha256", "relative_paths", "class_names"]).writerows(train_test_groups)

    counts = {split: sum(len(final[split][name]) for name in CLASS_ORDER) for split in SPLIT_ORDER}
    metadata = {
        "policy": "official_protocol", "validation_ratio": args.validation_ratio, "seed": args.seed,
        **source_summary, "final_train_count": counts["train"], "final_val_count": counts["val"],
        "final_test_count": counts["test"], "final_total_count": sum(counts.values()),
        "per_split_class_counts": {split: {name: len(final[split][name]) for name in CLASS_ORDER} for split in SPLIT_ORDER},
        "validation_target_counts": targets, "validation_achieved_counts": achieved,
        "manifest_sha256": manifest_sha, "warning_csv_sha256": sha256_file(warning_csv),
        "warning_count": len(conflict_groups) + len(train_test_groups),
        "conflicting_label_group_warning_count": len(conflict_groups),
        "inherited_train_test_overlap_warning_count": len(train_test_groups),
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"official protocol: train={counts['train']} val={counts['val']} test={counts['test']} total={sum(counts.values())}")
    print(f"warnings: conflicts={len(conflict_groups)} inherited_train_test_overlap_groups={len(train_test_groups)}")
    return 0


def _group_by_sha(records: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        grouped[row["sha256"]].append(row)
    return grouped


if __name__ == "__main__":
    raise SystemExit(main())
