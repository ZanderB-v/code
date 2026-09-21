#!/usr/bin/env python3
"""Regression tests for the frozen Near-Miss HEM eligibility gate."""

from __future__ import annotations

from check_near_miss_hem_gate import evaluate_gate


def fixture(ratio: float) -> dict:
    genuine = round(64 * ratio)
    return {
        "status": "STAGE6_ERROR_MIGRATION_COMPLETE",
        "test_evaluated": False,
        "corrupted_dev_used": False,
        "source_metrics": {"M3": {"alpha": 0.15}},
        "hem_gate": {
            "quantitative_candidate": True,
            "manual_review": {
                "complete": True,
                "sample_count": 64,
                "decision_counts": {
                    "genuine_ocr_error": genuine,
                    "normalization_issue": 64 - genuine,
                },
                "genuine_ocr_error_ratio": ratio,
            },
        },
    }


def main() -> None:
    eligible, errors = evaluate_gate(fixture(0.70))
    assert eligible and not errors
    eligible, errors = evaluate_gate(fixture(0.6999))
    assert not eligible
    assert any("below the frozen HEM threshold" in error for error in errors)
    print("NEAR_MISS_HEM_GATE_REGRESSION_OK")


if __name__ == "__main__":
    main()
