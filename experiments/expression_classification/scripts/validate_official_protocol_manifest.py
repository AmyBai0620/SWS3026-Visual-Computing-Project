#!/usr/bin/env python
"""Strictly validate the instructor-preserving official-protocol manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from create_full_manifest import CLASS_ORDER, MANIFEST_COLUMNS, scan_dataset


def sha_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    source, summary = scan_dataset(args.dataset_root)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle); rows = list(reader)
        if reader.fieldnames != MANIFEST_COLUMNS: raise SystemExit("FAILED: manifest schema")
    errors = []
    source_paths = {row["relative_path"] for row in source}
    paths = [row["relative_path"] for row in rows]
    if len(rows) != 35887 or set(paths) != source_paths or len(set(paths)) != len(paths): errors.append("source preservation/path accounting")
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    if counts != {"train": 24406, "val": 4303, "test": 7178}: errors.append(f"counts={counts}")
    sha_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sha_splits[row["sha256"]].add(row["split"])
    train_val = sum(splits == {"train", "val"} for splits in sha_splits.values())
    if train_val: errors.append(f"train/val SHA overlap={train_val}")
    if sha_text(args.manifest) != metadata.get("manifest_sha256"): errors.append("manifest SHA metadata")
    if metadata.get("policy") != "official_protocol": errors.append("policy metadata")
    if metadata.get("source_total_count") != summary["source_total_count"]: errors.append("source metadata")
    warning_csv = args.metadata.parent / "official_protocol_warnings.csv"
    if not warning_csv.is_file() or sha_text(warning_csv) != metadata.get("warning_csv_sha256"): errors.append("warning report")
    inherited = sum({row["source_split"] for row in group} == {"train", "test"} for group in _groups(source).values())
    conflicts = sum(len({row["class_name"] for row in group}) > 1 for group in _groups([r for r in source if r["source_split"] == "train"]).values())
    print(f"counts: train={counts['train']} val={counts['val']} test={counts['test']} total={len(rows)}")
    print(f"warnings: inherited_train_test_overlap_groups={inherited}; train_conflicting_label_groups={conflicts}")
    if errors:
        print("FAILED: " + "; ".join(errors)); return 1
    print("PASSED: strict official-protocol validation; train/val SHA overlap=0")
    return 0


def _groups(rows):
    grouped = defaultdict(list)
    for row in rows: grouped[row["sha256"]].append(row)
    return grouped


if __name__ == "__main__":
    raise SystemExit(main())
