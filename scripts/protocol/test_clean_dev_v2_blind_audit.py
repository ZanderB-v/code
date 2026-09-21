#!/usr/bin/env python3
"""Regression checks for the Clean Dev V2 model-blind review package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LANGUAGE_COUNTS = {"zh": 346, "ug": 295, "kk": 310}
FORBIDDEN_REVIEW_FIELDS = {
    "prediction",
    "pred",
    "confidence",
    "edit_distance",
    "model",
    "checkpoint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("01_data_preparation/clean_dev_v2_protocol/blind_audit_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    audit = args.audit_dir if args.audit_dir.is_absolute() else root / args.audit_dir
    summary = json.loads(
        (audit / "blind_audit_summary.json").read_text(encoding="utf-8-sig")
    )
    assert summary["protocol_id"] == "CLEAN_DEV_V2_MODEL_BLIND_AUDIT_V1"
    assert summary["model_blind"] is True
    assert summary["model_predictions_loaded"] is False
    assert summary["rows"] == 951
    assert summary["language_counts"] == LANGUAGE_COUNTS
    assert summary["clean_dev_v2_generated"] is False
    assert summary["formal_model_development_allowed"] is False
    assert summary["test_evaluated"] is False

    keys = set()
    for language, expected in LANGUAGE_COUNTS.items():
        csv_path = audit / f"{language}_blind_review.csv"
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            rows = list(reader)
        assert not (fields & FORBIDDEN_REVIEW_FIELDS)
        assert len(rows) == expected
        for row in rows:
            key = (row["language"], row["sample_id"])
            assert key not in keys
            keys.add(key)
            image = (
                root
                / "01_data_preparation/real_line_dataset_eval_reviewed"
                / row["image_path"]
            )
            assert image.is_file(), image
        html = (audit / f"{language}_blind_review.html").read_text(encoding="utf-8")
        lowered = html.lower()
        assert "prediction" not in lowered
        assert "confidence" not in lowered
        assert "edit distance" not in lowered
        assert "view==='pending'&&(!done||bad)" in html

    assert len(keys) == 951
    print(
        json.dumps(
            {
                "status": "CLEAN_DEV_V2_BLIND_AUDIT_REGRESSION_OK",
                "rows": len(keys),
                "model_predictions_loaded": False,
                "test_evaluated": False,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
