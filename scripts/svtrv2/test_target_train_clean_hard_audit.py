#!/usr/bin/env python3
"""CPU-only regression tests for the target-train clean-hard audit."""

from __future__ import annotations

import tempfile
from pathlib import Path

from target_train_clean_hard_audit import (
    ALLOWED_DECISIONS,
    assign_sample_weight,
    normalize_for_risk,
    read_resumable_jsonl,
    read_csv,
    select_stratified_review,
    stable_fingerprint,
    text_risk_flags,
    write_csv,
)


def main() -> None:
    dictionary = set("汉字abcАБاە123")
    assert text_risk_flags("汉字", dictionary) == []
    assert "dictionary_oov" in text_risk_flags("汉?", dictionary)
    assert "unicode_control_or_surrogate" in text_risk_flags("a\u200db", dictionary)
    assert "suspicious_mixed_scripts" in text_risk_flags("汉a", dictionary)
    assert normalize_for_risk("Ａ  B") == "A B"

    records = []
    for language in ("zh", "ug", "kk"):
        for distance in (1, 2):
            for index in range(8):
                records.append(
                    {
                        "sample_id": f"{language}-{distance}-{index}",
                        "language": language,
                        "ed": distance,
                        "ed_bucket": str(distance),
                        "image_path": f"train_reviewed/{language}/{index}.jpg",
                        "gt_text": "gt",
                        "prediction": "pred",
                        "confidence": 0.9,
                        "width": 100,
                        "height": 32,
                        "contrast_span": 80.0,
                        "blur_variance": 100.0,
                        "edge_contact": 0.1,
                        "risk_flags": [],
                        "eligible_for_review": index != 0,
                    }
                )
    first, population = select_stratified_review(records, per_stratum=3, seed=7)
    second, _ = select_stratified_review(records, per_stratum=3, seed=7)
    assert len(first) == 18
    assert set(population.values()) == {7}
    assert stable_fingerprint(first, ("sample_id",)) == stable_fingerprint(
        second, ("sample_id",)
    )
    assert all(not row["manual_decision"] for row in first)

    assert assign_sample_weight(
        {"sample_id": "a", "eligible_for_review": True, "ed": 1}, set()
    ) == (2.0, "clean_ed1")
    assert assign_sample_weight(
        {"sample_id": "b", "eligible_for_review": True, "ed": 2}, set()
    ) == (1.5, "clean_ed2")
    assert assign_sample_weight(
        {"sample_id": "c", "eligible_for_review": False, "ed": 1}, set()
    ) == (1.0, "ordinary")
    assert assign_sample_weight(
        {"sample_id": "a", "eligible_for_review": True, "ed": 1}, {"a"}
    ) == (1.0, "manual_review_excluded")
    assert len(ALLOWED_DECISIONS) == 4
    with tempfile.TemporaryDirectory() as directory:
        partial = Path(directory) / "partial.jsonl"
        partial.write_text('{"sample_id":"ok"}\n{"sample', encoding="utf-8")
        assert read_resumable_jsonl(partial) == [{"sample_id": "ok"}]
        assert partial.read_text(encoding="utf-8") == '{"sample_id":"ok"}\n'
        review_csv = Path(directory) / "review.csv"
        write_csv(review_csv, first, list(first[0]))
        _, roundtrip = read_csv(review_csv)
        assert stable_fingerprint(first, ("sample_id", "ed")) == stable_fingerprint(
            roundtrip, ("sample_id", "ed")
        )
    print(
        {
            "status": "TARGET_TRAIN_CLEAN_HARD_AUDIT_TESTS_OK",
            "strata": population,
            "review_rows": len(first),
            "formal_manifest_before_review": False,
            "test_evaluated": False,
        }
    )


if __name__ == "__main__":
    main()
