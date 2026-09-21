#!/usr/bin/env python3
"""Combine completed model summaries into one controlled comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from build_corruption_protocol import DEFAULT_ROOT, PROTOCOL_ID


MODEL_ORDER = ("e0", "e1", "b0", "b1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_root = (
        args.protocol_root.resolve()
        if args.protocol_root
        else root / "05_evaluation" / "corruption_protocol_v1"
    )
    summaries = []
    for model in args.models:
        path = protocol_root / "eval_reports" / model / "model_summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = read_json(path)
        if summary.get("status") != "passed":
            raise ValueError(f"Model summary did not pass: {path}")
        if summary.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"Protocol mismatch: {path}")
        summaries.append(summary)

    comparison = {
        "status": "passed",
        "errors": [],
        "protocol_id": PROTOCOL_ID,
        "selection_policy": (
            "checkpoints were selected by clean target Dev Macro CER before "
            "corruption evaluation"
        ),
        "corruption_role": "diagnostic_only_not_checkpoint_selection",
        "test_evaluated": False,
        "models": summaries,
    }
    output_dir = protocol_root / "eval_reports"
    json_path = output_dir / "clean_corrupted_dev_comparison.json"
    json_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = output_dir / "clean_corrupted_dev_comparison.csv"
    fields = (
        "model",
        "clean_macro_cer",
        "clean_macro_wer",
        "clean_macro_1_ned",
        "clean_macro_line_accuracy",
        "mean_corrupted_macro_cer",
        "absolute_macro_cer_increase",
        "relative_macro_cer_increase",
        "curve",
        "distort",
        "stretch",
        "rotate",
        "perspective",
        "shrink",
        "translate_x",
        "translate_y",
        "contrast",
        "brightness",
        "jpeg_compression",
        "pixelate",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            clean = summary["clean"]
            row = {
                "model": summary["model_key"],
                "clean_macro_cer": clean["macro_cer"],
                "clean_macro_wer": clean["macro_wer"],
                "clean_macro_1_ned": clean["macro_one_minus_ned"],
                "clean_macro_line_accuracy": clean["macro_line_accuracy"],
                "mean_corrupted_macro_cer": summary[
                    "mean_corrupted_macro_cer"
                ],
                "absolute_macro_cer_increase": summary[
                    "absolute_macro_cer_increase"
                ],
                "relative_macro_cer_increase": summary[
                    "relative_macro_cer_increase"
                ],
            }
            for corruption, values in summary["per_corruption"].items():
                row[corruption] = values["mean_macro_cer"]
            writer.writerow(row)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
