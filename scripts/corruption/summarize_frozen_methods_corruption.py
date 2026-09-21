#!/usr/bin/env python3
"""Build publication-ready Clean/Corrupted Dev tables for frozen methods."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METHODS = ("b0", "b1", "b2", "m1", "m2", "m3", "full")
LANGUAGES = ("zh", "ug", "kk")
OUTPUT_PREFIX = "method_binding_v1_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty list")
    return sum(values) / len(values)


def flattened_conditions(summary: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for corruption in summary["per_corruption"].values():
        result.extend(corruption["levels"].values())
    expected = int(summary["corrupted_conditions"])
    if len(result) != expected:
        raise ValueError(f"Expected {expected} conditions, found {len(result)}")
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_root = (
        args.protocol_root.resolve()
        if args.protocol_root
        else root / "05_evaluation" / "corruption_protocol_v1"
    )
    reports = protocol_root / "eval_reports"
    rows = []
    corruption_rows = []
    detailed: dict[str, Any] = {}
    for method in METHODS:
        path = reports / f"{OUTPUT_PREFIX}{method}" / "model_summary.json"
        summary = read_json(path)
        if summary.get("status") != "passed" or summary.get("test_evaluated") is not False:
            raise ValueError(f"Invalid method corruption summary: {path}")
        conditions = flattened_conditions(summary)
        clean = summary["clean"]
        language_stats = {}
        for language in LANGUAGES:
            clean_cer = float(clean["languages"][language]["cer"])
            corrupted_cer = mean(
                [float(item["languages"][language]["cer"]) for item in conditions]
            )
            delta = corrupted_cer - clean_cer
            language_stats[language] = {
                "clean_cer": clean_cer,
                "mean_corrupted_cer": corrupted_cer,
                "absolute_cer_increase": delta,
                "relative_cer_increase": delta / clean_cer if clean_cer else None,
            }
        per_corruption = {}
        for name, group in summary["per_corruption"].items():
            levels = list(group["levels"].values())
            per_corruption[name] = {
                "mean_macro_cer": mean([float(item["macro_cer"]) for item in levels]),
                "languages": {
                    language: mean(
                        [float(item["languages"][language]["cer"]) for item in levels]
                    )
                    for language in LANGUAGES
                },
            }
            corruption_rows.append(
                {
                    "method": method,
                    "corruption": name,
                    "mean_macro_cer": per_corruption[name]["mean_macro_cer"],
                    **{
                        f"{language}_cer": per_corruption[name]["languages"][language]
                        for language in LANGUAGES
                    },
                }
            )
        row = {
            "method": method,
            "clean_macro_cer": float(clean["macro_cer"]),
            "mean_corrupted_macro_cer": float(summary["mean_corrupted_macro_cer"]),
            "absolute_macro_cer_increase": float(summary["absolute_macro_cer_increase"]),
            "relative_macro_cer_increase": float(summary["relative_macro_cer_increase"]),
            **{
                f"{language}_{metric}": values[metric]
                for language, values in language_stats.items()
                for metric in (
                    "clean_cer",
                    "mean_corrupted_cer",
                    "absolute_cer_increase",
                    "relative_cer_increase",
                )
            },
        }
        rows.append(row)
        detailed[method] = {
            **row,
            "per_corruption": per_corruption,
            "checkpoint_sha256": summary["checkpoint_sha256"],
        }

    b1 = detailed["b1"]
    m3 = detailed["m3"]
    comparison = {
        "m3_minus_b1_clean_macro_cer": m3["clean_macro_cer"] - b1["clean_macro_cer"],
        "m3_minus_b1_mean_corrupted_macro_cer": (
            m3["mean_corrupted_macro_cer"] - b1["mean_corrupted_macro_cer"]
        ),
        "m3_absolute_gain_clean": b1["clean_macro_cer"] - m3["clean_macro_cer"],
        "m3_absolute_gain_corrupted": (
            b1["mean_corrupted_macro_cer"] - m3["mean_corrupted_macro_cer"]
        ),
    }
    comparison["m3_gain_expands_under_corruption"] = (
        comparison["m3_absolute_gain_corrupted"]
        > comparison["m3_absolute_gain_clean"]
    )
    comparison["m3_improves_corrupted_dev"] = (
        comparison["m3_absolute_gain_corrupted"] > 0
    )
    comparison["robustness_claim_supported"] = (
        comparison["m3_improves_corrupted_dev"]
        and comparison["m3_gain_expands_under_corruption"]
    )
    comparison["languages"] = {
        language: {
            "m3_minus_b1_clean_cer": (
                m3[f"{language}_clean_cer"] - b1[f"{language}_clean_cer"]
            ),
            "m3_minus_b1_mean_corrupted_cer": (
                m3[f"{language}_mean_corrupted_cer"]
                - b1[f"{language}_mean_corrupted_cer"]
            ),
        }
        for language in LANGUAGES
    }
    result = {
        "status": "passed",
        "models": list(METHODS),
        "protocol": "corruption_protocol_v1",
        "test_evaluated": False,
        "methods": detailed,
        "b1_vs_m3": comparison,
    }
    json_path = reports / "frozen_methods_clean_corrupted_dev_summary.json"
    csv_path = reports / "frozen_methods_clean_corrupted_dev_summary.csv"
    corruption_csv_path = reports / "frozen_methods_corruption_breakdown.csv"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with corruption_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(corruption_rows[0]))
        writer.writeheader()
        writer.writerows(corruption_rows)
    print(
        json.dumps(
            {
                "status": "FROZEN_METHOD_CORRUPTION_SUMMARY_OK",
                "json": str(json_path),
                "summary_csv": str(csv_path),
                "corruption_csv": str(corruption_csv_path),
                "robustness_claim_supported": comparison["robustness_claim_supported"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
