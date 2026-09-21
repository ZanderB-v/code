#!/usr/bin/env python3
"""Regression checks for Clean Dev V4 exclusions."""

from __future__ import annotations

import csv
import json
from pathlib import Path


EXPECTED_NEW = {
    ("ug", "misogyny_125_ug_L002"),
    ("ug", "misogyny_237_ug_L002"),
    ("ug", "misogyny_495_ug_L006"),
    ("ug", "misogyny_866_ug_L001"),
}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    protocol = root / "01_data_preparation/clean_dev_v4_adjudicated"
    frozen = json.loads(
        (protocol / "frozen_clean_dev_v4_manifest.json").read_text(encoding="utf-8")
    )
    assert frozen["status"] == "CLEAN_DEV_V4_ADJUDICATED_FROZEN"
    assert frozen["source_rows"] == 951
    assert frozen["evaluated_rows"] == 940
    assert frozen["language_evaluated_counts"] == {"zh": 342, "ug": 290, "kk": 308}
    assert frozen["inherited_v3_exclusions"] == 7
    assert frozen["new_v4_exclusions"] == 4
    assert frozen["excluded_rows"] == 11
    assert frozen["test_evaluated"] is False

    with (protocol / "v4_exclusion_ledger.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    new_keys = {
        (row["language"], row["sample_id"])
        for row in rows
        if row["new_v4_exclusion"] == "True"
    }
    assert new_keys == EXPECTED_NEW
    assert len(rows) == 11

    for language, expected in {"zh": 342, "ug": 290, "kk": 308}.items():
        lines = (protocol / "labels" / f"{language}.txt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == expected

    print(json.dumps({
        "status": "CLEAN_DEV_V4_ADJUDICATION_REGRESSION_OK",
        "source_rows": 951,
        "evaluated_rows": 940,
        "new_exclusions": 4,
        "test_evaluated": False,
        "errors": [],
    }))


if __name__ == "__main__":
    main()
