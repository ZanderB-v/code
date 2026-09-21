#!/usr/bin/env python3
"""Apply newly adjudicated exclusions to frozen Clean Dev V3 as Clean Dev V4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


PROTOCOL_ID = "CLEAN_DEV_V4_ADJUDICATED"
LANGUAGES = ("zh", "ug", "kk")
EXPECTED_NEW_EXCLUSIONS = {
    ("ug", "misogyny_125_ug_L002"),
    ("ug", "misogyny_237_ug_L002"),
    ("ug", "misogyny_495_ug_L006"),
    ("ug", "misogyny_866_ug_L001"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("01_data_preparation/clean_dev_v4_adjudicated"),
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"Output exists; use --replace: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    source = root / "01_data_preparation/clean_dev_v3_adjudicated"
    frozen_path = source / "frozen_clean_dev_v3_manifest.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8-sig"))
    if frozen.get("status") != "CLEAN_DEV_V3_ADJUDICATED_FROZEN":
        raise ValueError("Clean Dev V3 is not frozen")
    if frozen.get("test_evaluated") is not False:
        raise ValueError("Clean Dev V3 violates the no-Test policy")
    for relative, expected in frozen["artifact_sha256"].items():
        path = source / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen V3 artifact mismatch: {relative}")

    overrides_path = root / "01_data_preparation/clean_dev_v4_protocol/exclusion_overrides.csv"
    overrides = read_csv(overrides_path)
    override_by_key = {(row["language"], row["sample_id"]): row for row in overrides}
    if len(override_by_key) != len(overrides):
        raise ValueError("Duplicate V4 exclusion overrides")
    if set(override_by_key) != EXPECTED_NEW_EXCLUSIONS:
        raise ValueError("V4 overrides do not match the four frozen manual exclusions")
    if any(row["action"] != "exclude" for row in overrides):
        raise ValueError("V4 supports exclusions only")

    source_manifest = source / "clean_dev_v3_adjudicated_manifest.jsonl"
    source_rows = read_jsonl(source_manifest)
    if len(source_rows) != 951:
        raise ValueError(f"Expected 951 V3 source rows, got {len(source_rows)}")
    keys = {(row["language"], row["sample_id"]) for row in source_rows}
    if not set(override_by_key).issubset(keys):
        raise ValueError("V4 override references an unknown V3 row")

    final_rows = []
    applied = set()
    for row in source_rows:
        item = dict(row)
        key = (item["language"], item["sample_id"])
        override = override_by_key.get(key)
        if override is not None:
            if item["excluded"]:
                raise ValueError(f"V4 override is already excluded in V3: {key}")
            item["excluded"] = True
            item["exclusion_reason"] = override["reason"]
            applied.add(key)
        final_rows.append(item)
    if applied != set(override_by_key):
        raise ValueError("Not all V4 exclusions were applied")

    active_rows = [row for row in final_rows if not row["excluded"]]
    excluded_rows = [row for row in final_rows if row["excluded"]]
    expected_active = {"zh": 342, "ug": 290, "kk": 308}
    active_counts = dict(Counter(row["language"] for row in active_rows))
    if active_counts != expected_active:
        raise ValueError(f"Unexpected V4 active language counts: {active_counts}")

    manifest_path = output / "clean_dev_v4_adjudicated_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    labels_dir = output / "labels"
    labels_dir.mkdir()
    for language in LANGUAGES:
        with (labels_dir / f"{language}.txt").open("w", encoding="utf-8") as handle:
            for row in active_rows:
                if row["language"] == language:
                    handle.write(f"{row['image_path']}\t{row['normalized_gt']}\n")
    with (labels_dir / "all.txt").open("w", encoding="utf-8") as handle:
        for row in active_rows:
            handle.write(f"{row['image_path']}\t{row['normalized_gt']}\n")

    write_csv(
        output / "v4_exclusion_ledger.csv",
        [
            {
                "sample_id": row["sample_id"],
                "language": row["language"],
                "image_path": row["image_path"],
                "raw_gt": row["raw_gt"],
                "excluded": row["excluded"],
                "exclusion_reason": row["exclusion_reason"],
                "new_v4_exclusion": (row["language"], row["sample_id"]) in applied,
            }
            for row in excluded_rows
        ],
        [
            "sample_id",
            "language",
            "image_path",
            "raw_gt",
            "excluded",
            "exclusion_reason",
            "new_v4_exclusion",
        ],
    )
    shutil.copyfile(source / "label_patch_v3.csv", output / "label_patch_v4.csv")
    shutil.copyfile(source / "normalization_v2.yaml", output / "normalization_v2.yaml")

    artifacts = {
        "clean_dev_v4_adjudicated_manifest.jsonl": manifest_path,
        "v4_exclusion_ledger.csv": output / "v4_exclusion_ledger.csv",
        "label_patch_v4.csv": output / "label_patch_v4.csv",
        "normalization_v2.yaml": output / "normalization_v2.yaml",
        "labels/all.txt": labels_dir / "all.txt",
        "labels/zh.txt": labels_dir / "zh.txt",
        "labels/ug.txt": labels_dir / "ug.txt",
        "labels/kk.txt": labels_dir / "kk.txt",
    }
    summary = {
        "status": "CLEAN_DEV_V4_ADJUDICATED_FROZEN",
        "protocol_id": PROTOCOL_ID,
        "normalization_protocol": frozen["normalization_protocol"],
        "source_rows": len(final_rows),
        "evaluated_rows": len(active_rows),
        "language_evaluated_counts": active_counts,
        "active_patch_rows": frozen["active_patch_rows"],
        "inherited_v3_exclusions": frozen["excluded_rows"],
        "new_v4_exclusions": len(applied),
        "excluded_rows": len(excluded_rows),
        "source": {
            "frozen_clean_dev_v3_manifest_sha256": sha256(frozen_path),
            "clean_dev_v3_adjudicated_manifest_sha256": sha256(source_manifest),
            "v4_exclusion_overrides_sha256": sha256(overrides_path),
        },
        "artifact_sha256": {name: sha256(path) for name, path in artifacts.items()},
        "all_source_rows_preserved": True,
        "training_run": False,
        "corrupted_dev_used": False,
        "test_evaluated": False,
        "supersedes_for_future_development": "CLEAN_DEV_V3_ADJUDICATED",
        "checkpoint_reselection_required": True,
    }
    (output / "frozen_clean_dev_v4_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
