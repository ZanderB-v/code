#!/usr/bin/env python3
"""Regression checks for inherited Clean Dev V3 adjudication."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "01_data_preparation/clean_dev_v3_adjudicated"
    frozen = json.loads(
        (output / "frozen_clean_dev_v3_manifest.json").read_text(encoding="utf-8")
    )
    assert frozen["status"] == "CLEAN_DEV_V3_ADJUDICATED_FROZEN"
    assert frozen["source_rows"] == 951
    assert frozen["evaluated_rows"] == 944
    assert frozen["language_evaluated_counts"] == {"zh": 342, "kk": 308, "ug": 294}
    assert frozen["active_patch_rows"] == 25
    assert frozen["unique_patch_candidate_rows"] == 27
    assert frozen["patch_candidates_suppressed_by_exclusion"] == 1
    assert frozen["patch_candidates_equivalent_after_normalization"] == 1
    assert frozen["excluded_rows"] == 7
    assert frozen["model_selected_patch_rows_inherited"] == 24
    assert frozen["blind_patch_rows_inherited"] == 14
    assert frozen["blind_patch_rows_accounted"] == 15
    assert frozen["blind_patch_rows_superseded_by_adjudication"] == 1
    assert frozen["unresolved_conflicts"] == 0
    assert frozen["test_evaluated"] is False

    ledger = {(row["language"], row["sample_id"]): row for row in csv_rows(
        output / "review_inheritance_ledger.csv"
    )}
    assert len(ledger) == 951
    assert ledger[("zh", "misogyny_530_zh_L001")]["final_gt"] == "不是考研没用"
    assert ledger[("zh", "misogyny_994_zh_L003")]["final_gt"] == "我内心的平静"
    assert ledger[("kk", "misogyny_299_kk_L001")]["final_gt"] == "қайырлы таң"
    assert ledger[("ug", "misogyny_495_ug_L008")]["final_gt"] == "بولدى، مەن تېخى"
    assert ledger[("kk", "misogyny_705_kk_L002")]["final_gt"] == "болды!"
    assert ledger[("kk", "misogyny_1227_kk_L001")]["excluded"] == "True"
    assert ledger[("kk", "misogyny_125_kk_L002")]["excluded"] == "True"

    excluded = {(row["language"], row["sample_id"]) for row in csv_rows(
        output / "excluded_samples_v3.csv"
    )}
    assert len(excluded) == 7
    assert ("kk", "misogyny_1227_kk_L001") in excluded
    assert ("kk", "misogyny_125_kk_L002") in excluded

    label_counts = {}
    for language in ("zh", "ug", "kk"):
        lines = (output / "labels" / f"{language}.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        label_counts[language] = len(lines)
        assert all("\t" in line for line in lines)
    assert label_counts == {"zh": 342, "ug": 294, "kk": 308}

    print(json.dumps({
        "status": "CLEAN_DEV_V3_ADJUDICATION_REGRESSION_OK",
        "source_rows": 951,
        "evaluated_rows": 944,
        "active_patch_rows": 25,
        "excluded_rows": 7,
        "test_evaluated": False,
        "errors": [],
    }))


if __name__ == "__main__":
    main()
