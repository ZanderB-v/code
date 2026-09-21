#!/usr/bin/env python3
"""Apply the legacy U2 logical-output heuristic and score its predictions.

The U2 direction experiment trains CTC labels in visual left-to-right order.
Calling the bidi display transform a second time is exact for the frozen dev
set, but it is not a general inverse for mixed RTL/LTR text. Reports therefore
record both the heuristic output and its ground-truth round-trip diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from metrics_v1 import (
    edit_distance,
    finalize_metric_bucket,
    update_metric_bucket,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="U2 predictions.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-python-bidi",
        action="store_true",
        help="Fail when python-bidi is unavailable instead of using simple reverse.",
    )
    parser.add_argument(
        "--metric-normalization",
        choices=("none", "normalization_v2"),
        default="none",
        help="Apply the same frozen normalization to logical GT and prediction text.",
    )
    return parser.parse_args()


def get_bidi_display():
    try:
        from bidi.algorithm import get_display  # type: ignore

        return get_display, "python-bidi"
    except Exception:
        return None, "simple_reverse_fallback"


GET_DISPLAY, CONVERSION_METHOD = get_bidi_display()


def normalize_spaces(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def load_metric_normalizer(name: str):
    if name == "none":
        return lambda text: text, None, None
    protocol_dir = Path(__file__).resolve().parents[1] / "protocol"
    sys.path.insert(0, str(protocol_dir))
    from text_normalization_v2 import PROTOCOL_ID, normalize_text_v2

    implementation = protocol_dir / "text_normalization_v2.py"
    return (
        normalize_text_v2,
        PROTOCOL_ID,
        hashlib.sha256(implementation.read_bytes()).hexdigest(),
    )


def visual_to_logical(text: str) -> str:
    text = normalize_spaces(text)
    if GET_DISPLAY is not None:
        # This is a compatibility heuristic, not a mathematical inverse of the
        # Unicode bidi algorithm for arbitrary mixed-direction strings.
        return GET_DISPLAY(text, base_dir="R")
    return text[::-1]


def length_bucket(text: str) -> str:
    length = len(text)
    if length <= 10:
        return "short<=10"
    if length <= 32:
        return "medium11-32"
    return "long>32"


def text_kind(text: str) -> str:
    arabic = sum(0x0600 <= ord(ch) <= 0x06FF for ch in text)
    latin = sum(ch.isascii() and ch.isalpha() for ch in text)
    digit = sum(ch.isdigit() for ch in text)
    if latin and arabic == 0:
        return "latin_only"
    if digit and arabic == 0:
        return "digit_only_or_symbol"
    if latin or digit:
        return "ug_mixed_ascii_digit"
    return "ug_arabic_only"


def main() -> None:
    args = parse_args()
    normalize_metric_text, normalization_protocol, normalization_sha256 = (
        load_metric_normalizer(args.metric_normalization)
    )
    if args.require_python_bidi and CONVERSION_METHOD != "python-bidi":
        raise RuntimeError("python-bidi is required. Install with: python -m pip install python-bidi")

    rows = []
    with args.input.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_predictions = args.output_dir / "predictions_logical.jsonl"
    metrics = defaultdict(lambda: defaultdict(float))
    status_counts = Counter()
    gt_roundtrip_mismatches = []

    with out_predictions.open("w", encoding="utf-8") as out:
        for row in rows:
            status = row.get("status", "ok")
            status_counts[status] += 1
            gt_logical = normalize_metric_text(row.get("gt_text", ""))
            pred_visual = normalize_metric_text(row.get("pred_text", ""))
            pred_logical = normalize_metric_text(visual_to_logical(pred_visual))
            gt_visual = row.get("eval_gt_text", "")
            recovered_gt = normalize_metric_text(visual_to_logical(gt_visual))
            if recovered_gt != gt_logical:
                gt_roundtrip_mismatches.append(
                    {
                        "image": row.get("image"),
                        "gt_logical": gt_logical,
                        "gt_visual": gt_visual,
                        "gt_recovered": recovered_gt,
                    }
                )
            result = dict(row)
            result["gt_logical_text"] = gt_logical
            result["pred_visual_text"] = pred_visual
            result["pred_logical_text"] = pred_logical
            result["logical_char_edit_distance"] = edit_distance(gt_logical, pred_logical)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")

            if status == "ok":
                language = row.get("language", "unknown")
                update_metric_bucket(metrics["all"], gt_logical, pred_logical)
                update_metric_bucket(metrics[language], gt_logical, pred_logical)
                update_metric_bucket(
                    metrics[f"length:{length_bucket(gt_logical)}"],
                    gt_logical,
                    pred_logical,
                )
                update_metric_bucket(
                    metrics[f"kind:{text_kind(gt_logical)}"],
                    gt_logical,
                    pred_logical,
                )

    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "metric_normalization": normalization_protocol,
        "normalization_implementation_sha256": normalization_sha256,
        "conversion": {
            "from": "u2_visual_order_prediction",
            "to": "logical_like_prediction",
            "method": (
                "python_bidi_second_display_pass_heuristic"
                if CONVERSION_METHOD == "python-bidi"
                else CONVERSION_METHOD
            ),
            "is_general_inverse": False,
        },
        "ground_truth_roundtrip": {
            "mismatches": len(gt_roundtrip_mismatches),
            "examples": gt_roundtrip_mismatches[:20],
        },
        "rows": len(rows),
        "status_counts": dict(status_counts),
        "metrics": {
            name: finalize_metric_bucket(bucket)
            for name, bucket in sorted(metrics.items())
        },
        "outputs": {
            "predictions_logical": str(out_predictions),
            "summary": str(args.output_dir / "metrics_summary_logical.json"),
        },
    }
    (args.output_dir / "metrics_summary_logical.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
