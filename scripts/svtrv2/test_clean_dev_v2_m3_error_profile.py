#!/usr/bin/env python3
"""Regression checks for the Clean Dev V2 M3 error-profile package."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "05_evaluation/clean_dev_v2_m3_error_profile_v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    summary = read_json(OUTPUT / "error_profile_summary.json")
    gate = read_json(OUTPUT / "decision_gate.json")
    rows = read_csv(OUTPUT / "all_error_lines.csv")
    operations = read_csv(OUTPUT / "error_operations.csv")

    assert summary["protocol_id"] == "CLEAN_DEV_V2_M3_ERROR_PROFILE_V1"
    assert summary["model"] == "M3 / SOAR-SVTR"
    assert summary["alpha"] == 0.15
    assert summary["selected_epoch"] == 34
    assert summary["rows"] == {"zh": 346, "ug": 295, "kk": 310}
    assert summary["error_lines"] == {"zh": 51, "ug": 24, "kk": 23}
    assert len(rows) == 98
    assert len({row["review_id"] for row in rows}) == 98
    assert Counter(row["language"] for row in rows) == Counter(
        {"zh": 51, "ug": 24, "kk": 23}
    )
    assert len(operations) == 319
    assert abs(summary["metrics"]["macro"]["cer"] - 0.020234821995435567) < 1e-12
    assert abs(
        summary["metrics"]["macro"]["line_accuracy"] - 0.8990172251596258
    ) < 1e-12
    assert gate["status"] == "MANUAL_REVIEW_REQUIRED"
    assert gate["sldr"]["sldr_authorized"] is False
    assert gate["segment_count_or_duplicate_alignment"]["authorized"] is False
    assert gate["segment_count_or_duplicate_alignment"][
        "kk_adjacent_duplicate_candidates"
    ] == 1
    assert summary["selected_predictions_modified"] is False
    assert summary["corrupted_dev_used"] is False
    assert summary["test_evaluated"] is False

    print(
        json.dumps(
            {
                "status": "CLEAN_DEV_V2_M3_ERROR_PROFILE_TEST_OK",
                "error_lines": len(rows),
                "edit_operations": len(operations),
                "test_evaluated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
