#!/usr/bin/env python3
"""Freeze one explicit, dataset-wide punctuation task definition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from analyze_m3_ug_kk_errors import PUNCTUATION_REVIEW_CATEGORIES


CONFIRMATION = "I_CONFIRM_ONE_PUNCTUATION_POLICY_FOR_ALL_LANGUAGES_AND_MODELS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("ignore_frozen_set", "full_transcription"),
        required=True,
    )
    parser.add_argument("--confirmation", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.confirmation != CONFIRMATION:
        raise ValueError(f"Use the exact confirmation string: {CONFIRMATION}")
    root = args.root.resolve()
    audit_dir = root / "04_model_training/eval_reports/s50_m3_targeted_diagnostics_v1"
    candidate_path = audit_dir / "punctuation_protocol_candidate_v1.json"
    review_path = audit_dir / "punctuation_mismatch_review.csv"
    if not candidate_path.is_file() or not review_path.is_file():
        raise FileNotFoundError("Run the multilingual M3 error audit first")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Punctuation mismatch review is empty")
    invalid = [
        row
        for row in rows
        if row.get("manual_category") not in PUNCTUATION_REVIEW_CATEGORIES
    ]
    if invalid:
        raise ValueError(
            f"Punctuation review is incomplete or invalid: {len(invalid)} rows"
        )
    counts = Counter(row["manual_category"] for row in rows)

    output = root / "00_docs/frozen_punctuation_protocol_v1"
    protocol_path = output / "punctuation_protocol_v1.json"
    if protocol_path.exists():
        raise FileExistsError(
            f"Frozen punctuation protocol is immutable: {protocol_path}"
        )
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol_id": "PUNCTUATION_PROTOCOL_V1",
        "status": "frozen",
        "task_definition": args.policy,
        "scope": {
            "languages": ["zh", "ug", "kk"],
            "models": "all_models_without_exception",
            "splits": ["train", "dev", "test"],
        },
        "evaluation": (
            {
                "raw_prediction_preserved": True,
                "normalization": "remove exact set from both GT and prediction",
                "characters": candidate["candidate_characters"],
                "codepoints": candidate["candidate_codepoints"],
                "all_existing_model_metrics_must_be_recomputed": True,
            }
            if args.policy == "ignore_frozen_set"
            else {
                "raw_prediction_preserved": True,
                "normalization": "none",
                "missing_visible_GT_punctuation_is_annotation_error": True,
                "GT_repair_required_before_new_training": True,
            }
        ),
        "manual_review_counts": dict(counts),
        "audit": {
            "candidate": str(candidate_path),
            "candidate_sha256": sha256(candidate_path),
            "review": str(review_path),
            "review_sha256": sha256(review_path),
        },
        "selection_data": "Clean Dev audit only; no model selected by normalized metrics",
        "corrupted_dev_used": False,
        "test_evaluated": False,
    }
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "PUNCTUATION_PROTOCOL_V1_FROZEN",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "next_action": (
            "recompute_all_saved_dev_epochs_and_reselect_checkpoints_with_frozen_normalization"
            if args.policy == "ignore_frozen_set"
            else "repair_and_reaudit_all_split_annotations_before_training"
        ),
        "formal_new_module_training_allowed": False,
        "blocking_reason": (
            "Checkpoint selection must be repeated because the primary metric definition changed."
            if args.policy == "ignore_frozen_set"
            else "Frozen GT annotations require repair and a new data audit."
        ),
        "test_evaluated": False,
    }
    (output / "freeze_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
