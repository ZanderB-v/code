#!/usr/bin/env python3
"""Verify every frozen Clean Dev V2 artifact and protocol invariant."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from text_normalization_v2 import normalize_text_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("01_data_preparation/clean_dev_v2_verified"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol = args.protocol_dir if args.protocol_dir.is_absolute() else root / args.protocol_dir
    frozen = json.loads(
        (protocol / "frozen_clean_dev_v2_manifest.json").read_text(encoding="utf-8-sig")
    )
    assert frozen["status"] == "CLEAN_DEV_V2_VERIFIED_FROZEN"
    assert frozen["rows"] == 951
    assert frozen["patch_rows"] == 15
    assert frozen["all_original_rows_preserved"] is True
    assert frozen["ambiguous_or_crop_rows_excluded"] is False
    assert frozen["model_predictions_used_for_label_audit"] is False
    assert frozen["formal_new_module_training_allowed"] is False
    assert frozen["test_evaluated"] is False
    for relative, expected in frozen["artifact_sha256"].items():
        path = protocol / relative
        assert path.is_file(), path
        assert sha256(path) == expected, relative

    rows = []
    with (protocol / "clean_dev_v2_verified_manifest.jsonl").open(
        "r", encoding="utf-8-sig"
    ) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    assert len(rows) == 951
    assert len({(row["language"], row["sample_id"]) for row in rows}) == 951
    assert Counter(row["language"] for row in rows) == {"zh": 346, "ug": 295, "kk": 310}
    assert all(row["excluded"] is False for row in rows)
    assert all(normalize_text_v2(row["normalized_gt"]) == row["normalized_gt"] for row in rows)
    assert sum(row["raw_gt"] != row["patched_gt"] for row in rows) == 15
    assert sum(row["patched_gt"] != row["normalized_gt"] for row in rows) == 8
    print(
        json.dumps(
            {
                "status": "CLEAN_DEV_V2_VERIFICATION_PASSED",
                "rows": len(rows),
                "patch_rows": 15,
                "normalization_changed_rows": 8,
                "test_evaluated": False,
                "errors": [],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
