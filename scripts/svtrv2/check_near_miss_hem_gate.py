#!/usr/bin/env python3
"""Evaluate the frozen eligibility gate for Near-Miss HEM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_ID = "near_miss_hem_protocol_v1"
GENUINE_ERROR_THRESHOLD = 0.70
EXPECTED_REVIEW_ROWS = 64
EXPECTED_M3_ALPHA = 0.15
SAMPLING_WEIGHTS = {"ED=0": 1.0, "ED=1": 2.0, "ED=2": 1.5, "ED>=3": 1.0}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_gate(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if summary.get("status") != "STAGE6_ERROR_MIGRATION_COMPLETE":
        errors.append("Stage 6 analysis is not complete")
    if summary.get("test_evaluated") is not False:
        errors.append("Stage 6 must not use Test")
    if summary.get("corrupted_dev_used") is not False:
        errors.append("Stage 6 must not use Corrupted Dev for selection")

    review = summary.get("hem_gate", {}).get("manual_review", {})
    if review.get("complete") is not True:
        errors.append("ED=1 manual review is incomplete")
    if review.get("sample_count") != EXPECTED_REVIEW_ROWS:
        errors.append(
            f"Expected {EXPECTED_REVIEW_ROWS} reviewed ED=1 rows, "
            f"got {review.get('sample_count')!r}"
        )
    counts = review.get("decision_counts", {})
    if sum(int(value) for value in counts.values()) != EXPECTED_REVIEW_ROWS:
        errors.append("Manual decision counts do not sum to the reviewed sample count")

    ratio = review.get("genuine_ocr_error_ratio")
    if not isinstance(ratio, (int, float)):
        errors.append("Missing genuine OCR error ratio")
    elif float(ratio) < GENUINE_ERROR_THRESHOLD:
        errors.append(
            "Genuine OCR error ratio is below the frozen HEM threshold: "
            f"{float(ratio):.4%} < {GENUINE_ERROR_THRESHOLD:.0%}"
        )

    m3 = summary.get("source_metrics", {}).get("M3", {})
    if m3.get("alpha") != EXPECTED_M3_ALPHA:
        errors.append(
            f"M3 alpha mismatch: expected {EXPECTED_M3_ALPHA}, got {m3.get('alpha')!r}"
        )
    if summary.get("hem_gate", {}).get("quantitative_candidate") is not True:
        errors.append("ED-bucket quantitative condition did not pass")
    return not errors, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage6-summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-eligible",
        action="store_true",
        help="Exit non-zero when the frozen 70% eligibility gate does not pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    stage6 = (
        args.stage6_summary
        or root
        / "04_model_training/eval_reports/stage6_error_migration_v1/stage6_summary.json"
    ).resolve()
    output = (
        args.output
        or root / "04_model_training/hem_v1/eligibility_report.json"
    ).resolve()
    if not stage6.is_file():
        raise FileNotFoundError(stage6)

    summary = json.loads(stage6.read_text(encoding="utf-8"))
    eligible, errors = evaluate_gate(summary)
    review = summary["hem_gate"]["manual_review"]
    report = {
        "status": "HEM_ELIGIBLE" if eligible else "HEM_NOT_AUTHORIZED",
        "protocol_id": PROTOCOL_ID,
        "formal_training_allowed": eligible,
        "stage6_summary": str(stage6),
        "stage6_summary_sha256": sha256(stage6),
        "eligibility": {
            "reviewed_rows": review.get("sample_count"),
            "genuine_ocr_error_rows": review.get("decision_counts", {}).get(
                "genuine_ocr_error", 0
            ),
            "genuine_ocr_error_ratio": review.get("genuine_ocr_error_ratio"),
            "required_ratio": GENUINE_ERROR_THRESHOLD,
            "errors": errors,
        },
        "frozen_training_design": {
            "mining_split": "target_domain_train_only",
            "physical_gpu_ids": [0, 1],
            "distributed_world_size": 2,
            "batch_size_per_card": 16,
            "global_batch_size": 32,
            "batch_policy": "match frozen B1 and SOAR-SVTR controls",
            "memory_policy": (
                "maximize useful GPU utilization without changing the frozen batch schedule"
            ),
            "selection_metric": "Clean Dev Macro CER",
            "corrupted_dev_for_selection": False,
            "test_evaluated": False,
            "sampling_weights": SAMPLING_WEIGHTS,
            "new_formal_runs": ["B1+HEM", "SOAR-SVTR+HEM"],
            "fixed_m3_alpha": EXPECTED_M3_ALPHA,
            "no_concurrent_changes": [
                "alpha",
                "architecture",
                "augmentation",
                "direction_conditioning",
            ],
        },
        "next_action": (
            "mine_and_audit_target_train_then_run_controlled_HEM"
            if eligible
            else "clean_target_train_labels_and_normalization_before_HEM"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_eligible and not eligible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
