#!/usr/bin/env python3
"""Freeze the completed model-driven S50 M3 error review as diagnostic evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_ID = "S50_M3_ERROR_REVIEW_V1"
REQUIRED_FILES = (
    "review_summary.json",
    "all_errors.csv",
    "zh_reviewed_errors.csv",
    "ug_reviewed_errors.csv",
    "kk_reviewed_errors.csv",
    "label_patch_v1.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    review_dir = args.review_dir.resolve()
    freeze_path = review_dir / "frozen_review_manifest.json"
    if freeze_path.exists():
        raise FileExistsError(f"Review is already frozen: {freeze_path}")

    paths = {name: review_dir / name for name in REQUIRED_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required review artifacts: {missing}")

    summary = read_json(paths["review_summary.json"])
    if summary.get("status") != "S50_M3_ERROR_REVIEW_V1_COMPLETE":
        raise ValueError("Review summary is not complete")
    if summary.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected review protocol")
    if summary.get("review_complete") is not True:
        raise ValueError("Review is incomplete")
    if summary.get("invalid_patch_rows"):
        raise ValueError("Review contains invalid patch rows")
    if summary.get("test_evaluated") is not False:
        raise ValueError("Review does not preserve the Test prohibition")

    all_rows = csv_rows(paths["all_errors.csv"])
    patch_rows = csv_rows(paths["label_patch_v1.csv"])
    if len(all_rows) != 108 or len(patch_rows) != 24:
        raise ValueError(
            f"Unexpected review sizes: all={len(all_rows)}, patches={len(patch_rows)}"
        )
    if any(not row.get("manual_decision") for row in all_rows):
        raise ValueError("At least one review row has no manual decision")
    patch_keys = [(row["language"], row["sample_id"]) for row in patch_rows]
    if len(patch_keys) != len(set(patch_keys)):
        raise ValueError("Duplicate label-patch keys")
    if any(not row.get("new_gt") or row["new_gt"] == row["old_gt"] for row in patch_rows):
        raise ValueError("Every patch must contain a changed, non-empty GT")

    hashes = {name: sha256(path) for name, path in paths.items()}
    for language in ("zh", "ug", "kk"):
        expected = summary["review_csv_sha256"][language]
        actual = hashes[f"{language}_reviewed_errors.csv"]
        if actual != expected:
            raise ValueError(f"{language} review hash differs from completed summary")
    if hashes["label_patch_v1.csv"] != summary["label_patch_sha256"]:
        raise ValueError("Label-patch hash differs from completed summary")

    frozen = {
        "status": "S50_M3_ERROR_REVIEW_V1_FROZEN",
        "protocol_id": PROTOCOL_ID,
        "scope": "model-driven diagnostic review; not the final Clean Dev V2 audit",
        "selection_bias_disclosure": (
            "Rows were selected because frozen M3 predicted them incorrectly. "
            "Corrections must not define Clean Dev V2 without a model-blind full audit."
        ),
        "error_rows": len(all_rows),
        "patch_rows": len(patch_rows),
        "artifact_sha256": hashes,
        "source_prediction_sha256": summary["source_prediction_sha256"],
        "original_clean_dev_modified": False,
        "test_evaluated": False,
    }
    freeze_path.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(frozen, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
