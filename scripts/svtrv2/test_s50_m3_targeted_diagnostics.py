#!/usr/bin/env python3
"""Regression tests for the targeted M3 diagnostic protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_m3_ug_kk_errors import (  # noqa: E402
    blank_separated_evidence,
    duplicate_kind,
    indexed_alignment,
    strip_candidate_punctuation,
    ug_run_position,
)
from dual_order_protocol import METHODS, make_method_config  # noqa: E402
from export_ctc_raw_paths import compress_path  # noqa: E402
from build_s50_m3_error_review import (  # noqa: E402
    MANUAL_DECISIONS,
    auto_category,
    strip_punctuation,
)


def main() -> None:
    spec = METHODS["m3_nococ"]
    assert spec["ctc_order"] == "visual"
    assert spec["sgm_order"] == "logical"
    assert spec["use_script_adapter"] is True
    assert spec["consistency_weight"] == 0.0
    generated = make_method_config(
        method="m3_nococ",
        root=Path("/tmp/project"),
        run_dir=Path("/tmp/project/run"),
        project_name="m3_nococ_test",
        train_lmdbs=[Path("/tmp/project/train")],
        eval_lmdbs=[Path("/tmp/project/dev")],
        pretrained_model=Path("/tmp/project/source.pth"),
        max_epoch=50,
        first_batch_size=32,
        num_workers=8,
        max_ratio=40,
        lr=2.5e-5,
        internal_eval_every=100000,
        seed=20260731,
    )
    assert "method_variant: m3_nococ" in generated
    assert "consistency_weight: 0.0" in generated
    assert "script_weight: 0.1" in generated
    assert "use_script_adapter: True" in generated

    alignment = indexed_alignment("abc", "abbc")
    insertions = [item for item in alignment if item["operation"] == "ins"]
    assert len(insertions) == 1
    insertion = insertions[0]
    assert duplicate_kind("abbc", insertion["pred_index"]) in {
        "duplicate_left",
        "duplicate_right",
        "duplicate_both",
    }

    ids = np.asarray([1, 1, 0, 2, 2, 0, 2, 2, 3], dtype=np.int64)
    probabilities = np.asarray([0.9] * len(ids), dtype=np.float32)
    raw = compress_path(ids, probabilities, ["blank", "a", "b", "c"])
    assert raw["decoded_visual_text"] == "abbc"
    assert blank_separated_evidence(raw, 2, "duplicate_left")

    assert ug_run_position("ئ", 0) == "isolated"
    assert ug_run_position("ئۇيغۇر", 0) == "initial"
    assert ug_run_position("ئۇيغۇر", 2) == "medial"
    assert ug_run_position("ئۇيغۇر", 5) == "final"
    assert ug_run_position("ئۇيغۇر 1", 7) == "non_arabic_or_insertion"
    assert strip_candidate_punctuation("今天真开心！") == "今天真开心"
    assert strip_candidate_punctuation("ئۇيغۇرچە؟") == "ئۇيغۇرچە"
    assert strip_candidate_punctuation("қазақ.") == "қазақ"
    assert strip_punctuation("《今天真开心！》") == "今天真开心"
    punctuation_only = indexed_alignment("今天真开心", "今天真开心！")
    punctuation_only = [row for row in punctuation_only if row["operation"] != "match"]
    assert auto_category("今天真开心", "今天真开心！", punctuation_only) == "punctuation_only_candidate"
    duplicate = indexed_alignment("қазақ", "қазаққ")
    duplicate = [row for row in duplicate if row["operation"] != "match"]
    assert auto_category("қазақ", "қазаққ", duplicate) == "duplicate_insertion_candidate"
    assert set(MANUAL_DECISIONS) == {
        "genuine_char_error",
        "punctuation_gt_missing",
        "gt_annotation_error",
        "normalization_issue",
        "duplicate_error",
        "image_or_crop_issue",
        "ambiguous_image",
    }

    print(
        {
            "status": "S50_M3_TARGETED_DIAGNOSTIC_TESTS_OK",
            "m3_nococ_only_changed_factor": "consistency_weight",
            "blank_separated_duplicate_peak": True,
            "ug_context_positions": True,
        }
    )


if __name__ == "__main__":
    main()
