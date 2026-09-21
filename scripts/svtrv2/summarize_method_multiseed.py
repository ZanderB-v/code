#!/usr/bin/env python3
"""Summarize target-finetune seeds and paired-bootstrap B1 versus M3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any


METHODS = ("b1", "m1", "m2", "m3")
LANGUAGES = ("zh", "ug", "kk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remap_server_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    normalized = value.replace("\\", "/")
    marker = "/04_model_training/"
    if marker in normalized:
        return root / "04_model_training" / normalized.split(marker, 1)[1]
    return path


def metric_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std_sample": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def collect_summaries(root: Path, plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for method in METHODS:
        spec = plan["methods"][method]
        items = [(int(spec["canonical"]["seed"]), Path(spec["canonical"]["final_summary"]))]
        items.extend((int(item["seed"]), Path(item["final_summary"])) for item in spec["new_runs"])
        summaries = []
        for seed, path in items:
            path = remap_server_path(root, str(path))
            summary = read_json(path)
            if summary.get("test_policy") != "not_evaluated_during_model_development":
                raise ValueError(f"Test policy violation: {path}")
            if summary.get("preprocess_protocol") != "P1_MSR_V3":
                raise ValueError(f"Preprocessing protocol drift: {path}")
            if summary.get("dev", {}).get("selection_metric") != "clean_target_dev_macro_CER":
                raise ValueError(f"Checkpoint selection metric drift: {path}")
            if seed == int(plan["canonical_seed"]):
                config_path = remap_server_path(root, spec["canonical"]["config"])
                checkpoint_path = remap_server_path(root, spec["canonical"]["checkpoint"])
                if sha256(config_path) != spec["canonical"]["config_sha256"]:
                    raise ValueError(f"Canonical config hash drift: {config_path}")
                if sha256(checkpoint_path) != spec["canonical"]["checkpoint_sha256"]:
                    raise ValueError(f"Canonical checkpoint hash drift: {checkpoint_path}")
                if summary.get("config_sha256") != spec["canonical"]["config_sha256"]:
                    raise ValueError(f"Canonical summary/config mismatch: {path}")
            else:
                planned = next(item for item in spec["new_runs"] if int(item["seed"]) == seed)
                config_path = remap_server_path(root, planned["config"])
                if sha256(config_path) != planned["config_sha256"]:
                    raise ValueError(f"New-seed config hash drift: {config_path}")
                if summary.get("config_sha256") != planned["config_sha256"]:
                    raise ValueError(f"Summary/config mismatch: {path}")
            checkpoint_path = remap_server_path(root, summary["best_checkpoint"])
            if not checkpoint_path.is_file() or checkpoint_path.name != "best_clean_dev_macro_cer.pth":
                raise ValueError(f"Missing or invalid selected checkpoint: {checkpoint_path}")
            summaries.append({"seed": seed, "path": str(path), "summary": summary})
        if len(summaries) != 3:
            raise ValueError(f"{method}: expected 3 seeds, found {len(summaries)}")
        result[method] = sorted(summaries, key=lambda item: item["seed"])
    return result


def read_predictions(root: Path, summary: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    output = remap_server_path(root, summary["dev"]["output_dir"])
    paths = {
        "zh": output / "zh" / "predictions.jsonl",
        "ug": output / "ug_logical" / "predictions_logical.jsonl",
        "kk": output / "kk" / "predictions.jsonl",
    }
    result = {}
    for language, path in paths.items():
        rows = {}
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                row = json.loads(line)
                key = row["image"]
                if key in rows:
                    raise ValueError(f"Duplicate prediction image: {key}")
                rows[key] = row
        result[language] = rows
    return result


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_delta(
    samples: dict[str, list[tuple[float, float, int]]],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)

    def score(selected: dict[str, list[int]]) -> float:
        language_cers = []
        for language in LANGUAGES:
            rows = samples[language]
            indices = selected[language]
            gt_chars = sum(rows[index][2] for index in indices)
            delta_edits = sum(rows[index][1] - rows[index][0] for index in indices)
            language_cers.append(delta_edits / gt_chars)
        return sum(language_cers) / len(language_cers)

    observed = score({language: list(range(len(samples[language]))) for language in LANGUAGES})
    draws = []
    for _ in range(iterations):
        selected = {
            language: [rng.randrange(len(samples[language])) for _ in samples[language]]
            for language in LANGUAGES
        }
        draws.append(score(selected))
    return {
        "metric": "M3_macro_CER_minus_B1_macro_CER",
        "observed_delta": observed,
        "ci95_percentile": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "probability_m3_lower_cer": sum(value < 0 for value in draws) / len(draws),
        "iterations": iterations,
        "seed": seed,
        "interpretation": "negative_delta_favors_M3",
    }


def paired_samples(
    b1_predictions: dict[str, dict[str, dict[str, Any]]],
    m3_predictions: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, list[tuple[float, float, int]]]:
    result = {}
    for language in LANGUAGES:
        if set(b1_predictions[language]) != set(m3_predictions[language]):
            raise ValueError(f"B1/M3 sample mismatch for {language}")
        rows = []
        for key in sorted(b1_predictions[language]):
            left = b1_predictions[language][key]
            right = m3_predictions[language][key]
            left_gt = left["eval_gt_text"]
            right_gt = right["eval_gt_text"]
            if left_gt != right_gt:
                raise ValueError(f"Ground-truth mismatch: {key}")
            rows.append(
                (
                    float(left["char_edit_distance"]),
                    float(right["char_edit_distance"]),
                    len(left_gt),
                )
            )
        result[language] = rows
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    plan = read_json(args.plan.resolve())
    if args.bootstrap_iterations < 1000:
        raise ValueError("Publication bootstrap requires at least 1000 iterations")
    if (
        plan.get("status") != "passed"
        or plan.get("protocol_id") != "METHOD_TARGET_FINETUNE_MULTISEED_V1"
        or plan.get("test_policy") != "not_evaluated"
    ):
        raise ValueError("Invalid multiseed plan")
    collected = collect_summaries(root, plan)
    rows = []
    method_results = {}
    for method in METHODS:
        records = collected[method]
        macro_cer = [float(item["summary"]["dev"]["macro_cer"]) for item in records]
        macro_line = [float(item["summary"]["dev"]["macro_line_accuracy"]) for item in records]
        method_result = {
            "seeds": [item["seed"] for item in records],
            "training_scope": "fixed_synthetic_pretrain_checkpoint_target_finetune_only",
            "macro_cer": metric_stats(macro_cer),
            "macro_line_accuracy": metric_stats(macro_line),
            "languages": {},
            "runs": [
                {
                    "seed": item["seed"],
                    "summary": item["path"],
                    "best_epoch": item["summary"]["best_epoch"],
                    "macro_cer": item["summary"]["dev"]["macro_cer"],
                }
                for item in records
            ],
        }
        for language in LANGUAGES:
            language_rows = [item["summary"]["dev"]["languages"][language] for item in records]
            method_result["languages"][language] = {
                metric: metric_stats([float(row[metric]) for row in language_rows])
                for metric in ("cer", "wer", "one_minus_ned_macro", "line_accuracy")
            }
        method_results[method] = method_result
        rows.append(
            {
                "method": method,
                "seeds": ",".join(str(value) for value in method_result["seeds"]),
                "macro_cer_mean": method_result["macro_cer"]["mean"],
                "macro_cer_std": method_result["macro_cer"]["std_sample"],
                "macro_line_accuracy_mean": method_result["macro_line_accuracy"]["mean"],
                "macro_line_accuracy_std": method_result["macro_line_accuracy"]["std_sample"],
                **{
                    f"{language}_cer_mean": method_result["languages"][language]["cer"]["mean"]
                    for language in LANGUAGES
                },
                **{
                    f"{language}_cer_std": method_result["languages"][language]["cer"]["std_sample"]
                    for language in LANGUAGES
                },
            }
        )

    b1_by_seed = {item["seed"]: item for item in collected["b1"]}
    m3_by_seed = {item["seed"]: item for item in collected["m3"]}
    common_seeds = sorted(set(b1_by_seed) & set(m3_by_seed))
    if common_seeds != sorted([int(plan["canonical_seed"]), *plan["new_seeds"]]):
        raise ValueError(f"B1/M3 paired seed set mismatch: {common_seeds}")
    per_seed_bootstrap = {}
    all_seed_predictions = {"b1": {}, "m3": {}}
    for seed in common_seeds:
        left = read_predictions(root, b1_by_seed[seed]["summary"])
        right = read_predictions(root, m3_by_seed[seed]["summary"])
        all_seed_predictions["b1"][seed] = left
        all_seed_predictions["m3"][seed] = right
        per_seed_bootstrap[str(seed)] = bootstrap_delta(
            paired_samples(left, right),
            args.bootstrap_iterations,
            args.bootstrap_seed + seed % 1000,
        )

    averaged: dict[str, list[tuple[float, float, int]]] = {}
    for language in LANGUAGES:
        keys = sorted(all_seed_predictions["b1"][common_seeds[0]][language])
        averaged_rows = []
        for key in keys:
            b1_rows = [all_seed_predictions["b1"][seed][language][key] for seed in common_seeds]
            m3_rows = [all_seed_predictions["m3"][seed][language][key] for seed in common_seeds]
            gt = b1_rows[0]["eval_gt_text"]
            if any(row["eval_gt_text"] != gt for row in b1_rows + m3_rows):
                raise ValueError(f"Cross-seed GT mismatch: {key}")
            averaged_rows.append(
                (
                    statistics.mean(float(row["char_edit_distance"]) for row in b1_rows),
                    statistics.mean(float(row["char_edit_distance"]) for row in m3_rows),
                    len(gt),
                )
            )
        averaged[language] = averaged_rows
    bootstrap = {
        "paired_unit": "same_dev_line_stratified_by_language",
        "per_seed": per_seed_bootstrap,
        "seed_averaged_primary": bootstrap_delta(
            averaged,
            args.bootstrap_iterations,
            args.bootstrap_seed,
        ),
    }
    output_root = root / "04_model_training" / "eval_reports" / "multiseed"
    output_root.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "passed",
        "protocol_id": plan["protocol_id"],
        "test_evaluated": False,
        "methods": method_results,
        "paired_bootstrap_b1_vs_m3": bootstrap,
    }
    json_path = output_root / "method_target_multiseed_summary.json"
    csv_path = output_root / "method_target_multiseed_summary.csv"
    bootstrap_path = output_root / "paired_bootstrap_b1_vs_m3.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bootstrap_path.write_text(json.dumps(bootstrap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "METHOD_MULTISEED_SUMMARY_AND_BOOTSTRAP_OK", "summary": str(json_path), "csv": str(csv_path), "bootstrap": str(bootstrap_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
