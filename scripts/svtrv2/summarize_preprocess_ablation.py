#!/usr/bin/env python3
"""Summarize P0/P1 dev metrics by language, length, and source aspect ratio."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from metrics_v1 import finalize_metric_bucket, update_metric_bucket


LANGUAGES = ("zh", "ug", "kk")
IPS_PATTERN = re.compile(r"\bips:\s*([0-9]+(?:\.[0-9]+)?)\s+samples/s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/data_home/wudayu/experiments/multilingual_meme_ocr/"
            "svtrv2_line_recognition"
        ),
    )
    parser.add_argument("--p0-report", type=Path, required=True)
    parser.add_argument("--p1-report", type=Path, required=True)
    parser.add_argument("--p0-log", type=Path, required=True)
    parser.add_argument("--p1-log", type=Path, required=True)
    parser.add_argument("--p0-gpu-csv", type=Path, required=True)
    parser.add_argument("--p1-gpu-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalize_path(value: str) -> str:
    return (value or "").replace("\\", "/")


def length_bucket(text: str) -> str:
    length = len(text)
    if length <= 10:
        return "short<=10"
    if length <= 32:
        return "medium11-32"
    return "long>32"


def aspect_bucket(aspect: float) -> str:
    if aspect < 4:
        return "aspect<4"
    if aspect < 8:
        return "aspect4-8"
    if aspect < 12:
        return "aspect8-12"
    return "ultra_wide>=12"


def load_metadata(path: Path) -> dict[str, dict]:
    result = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image = normalize_path(row.get("image") or "")
            if image:
                result[image] = row
                result[Path(image).name] = row
    return result


def load_rows(report_dir: Path, language: str) -> list[dict]:
    if language == "ug":
        path = report_dir / "ug_logical" / "predictions_logical.jsonl"
    else:
        path = report_dir / language / "predictions.jsonl"
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metric_texts(row: dict, language: str) -> tuple[str, str]:
    if language == "ug":
        return (
            row.get("gt_logical_text") or "",
            row.get("pred_logical_text") or "",
        )
    return row.get("eval_gt_text") or "", row.get("eval_pred_text") or ""


def parse_training_ips(path: Path) -> dict:
    values = [
        float(match.group(1))
        for match in IPS_PATTERN.finditer(
            path.read_text(encoding="utf-8-sig", errors="replace")
        )
    ]
    if not values:
        return {"measurements": 0, "mean_samples_per_second": None}
    values.sort()
    return {
        "measurements": len(values),
        "mean_samples_per_second": sum(values) / len(values),
        "median_samples_per_second": values[len(values) // 2],
        "min_samples_per_second": min(values),
        "max_samples_per_second": max(values),
    }


def parse_gpu_csv(path: Path) -> dict:
    memory_values = []
    utilization_values = []
    if not path.is_file():
        return {"samples": 0, "max_memory_used_mib": None}
    for line in path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        try:
            memory_values.append(float(parts[2]))
            utilization_values.append(float(parts[3]))
        except ValueError:
            continue
    return {
        "samples": len(memory_values),
        "max_memory_used_mib": max(memory_values, default=None),
        "mean_memory_used_mib": (
            sum(memory_values) / len(memory_values)
            if memory_values
            else None
        ),
        "mean_gpu_utilization_percent": (
            sum(utilization_values) / len(utilization_values)
            if utilization_values
            else None
        ),
    }


def summarize_one(
    report_dir: Path,
    metadata: dict[str, dict],
    train_log: Path,
    gpu_csv: Path,
) -> dict:
    buckets = defaultdict(lambda: defaultdict(float))
    language_metrics = {}
    inference_timing = {}
    for language in LANGUAGES:
        rows = load_rows(report_dir, language)
        for row in rows:
            if row.get("status") != "ok":
                continue
            gt, pred = metric_texts(row, language)
            image = normalize_path(row.get("image") or "")
            metadata_row = metadata.get(image) or metadata.get(Path(image).name)
            try:
                aspect = float((metadata_row or {}).get("bbox_aspect") or 0)
            except (TypeError, ValueError):
                aspect = 0.0
            update_metric_bucket(buckets[f"language:{language}"], gt, pred)
            update_metric_bucket(
                buckets[f"length:{language}:{length_bucket(gt)}"], gt, pred
            )
            update_metric_bucket(
                buckets[f"aspect:{language}:{aspect_bucket(aspect)}"], gt, pred
            )
            update_metric_bucket(buckets["all"], gt, pred)

        if language == "ug":
            timing_path = (
                report_dir
                / "ug_visual"
                / "metrics_summary.json"
            )
        else:
            timing_path = report_dir / language / "metrics_summary.json"
        timing_summary = json.loads(
            timing_path.read_text(encoding="utf-8-sig")
        )
        inference_timing[language] = timing_summary.get("timing", {})

    finalized = {
        name: finalize_metric_bucket(bucket)
        for name, bucket in sorted(buckets.items())
    }
    for language in LANGUAGES:
        language_metrics[language] = finalized[f"language:{language}"]
    macro_cer = sum(
        language_metrics[language]["cer"] for language in LANGUAGES
    ) / len(LANGUAGES)
    return {
        "target_dev_macro_cer": macro_cer,
        "language_metrics": language_metrics,
        "length_metrics": {
            key: value
            for key, value in finalized.items()
            if key.startswith("length:")
        },
        "aspect_metrics": {
            key: value
            for key, value in finalized.items()
            if key.startswith("aspect:")
        },
        "training_throughput": parse_training_ips(train_log),
        "gpu": parse_gpu_csv(gpu_csv),
        "inference_timing": inference_timing,
        "report_dir": str(report_dir),
    }


def main() -> None:
    args = parse_args()
    target_metadata = (
        args.root
        / "01_data_preparation"
        / "real_line_dataset_eval_reviewed"
        / "metadata.jsonl"
    )
    metadata = load_metadata(target_metadata)
    p0 = summarize_one(
        args.p0_report, metadata, args.p0_log, args.p0_gpu_csv
    )
    p1 = summarize_one(
        args.p1_report, metadata, args.p1_log, args.p1_gpu_csv
    )
    preferred = (
        "p0_fixed"
        if p0["target_dev_macro_cer"] <= p1["target_dev_macro_cer"]
        else "p1_msr"
    )
    result = {
        "experiment": "preprocess_ablation_v1",
        "selection_split": "target_dev_only",
        "test_used": False,
        "selection_rule": (
            "Lower target-dev macro CER; inspect long and ultra-wide CER, "
            "GPU memory, and throughput before protocol freeze."
        ),
        "p0_fixed": p0,
        "p1_msr": p1,
        "automatic_preference_by_macro_cer": preferred,
        "manual_confirmation_required": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
