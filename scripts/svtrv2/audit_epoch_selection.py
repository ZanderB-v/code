#!/usr/bin/env python3
"""Audit Clean Dev checkpoint selection from historical evaluation logs.

The script intentionally uses only completed Clean Dev evaluation summaries. It
does not interpolate missing epochs and never reads corrupted Dev or Test data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple


LANGUAGES = ("zh", "ug", "kk")
METRICS = ("cer", "wer", "one_minus_ned_macro", "line_accuracy")

MODEL_SPECS = {
    "B1": "b1_full_s50_to_target_final_summary.json",
    "M1": "svtrv2_s_m1_dual_order_s50_to_target_final_summary.json",
    "M2": "svtrv2_s_m2_alpha_030_s50_to_target_final_summary.json",
    "M3": "svtrv2_s_m3_dual_order_s50_to_target_final_summary.json",
    "Full": "svtrv2_s_full_dual_order_s50_to_target_final_summary.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=None,
        help="Defaults to ROOT/04_model_training/logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to ROOT/05_evaluation/epoch_selection_audit_v1.",
    )
    parser.add_argument(
        "--require-every-epoch",
        action="store_true",
        help="Fail if the supplied logs do not cover every integer epoch.",
    )
    return parser.parse_args()


def iter_json_objects(text: str) -> Iterable[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            return
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = start + max(consumed, 1)
        if isinstance(value, dict):
            yield value


def select_summary(log_path: Path) -> Dict[str, Any]:
    summaries = []
    for value in iter_json_objects(log_path.read_text(encoding="utf-8", errors="replace")):
        languages = value.get("languages")
        if not isinstance(languages, dict):
            continue
        if not all(language in languages for language in LANGUAGES):
            continue
        if "output_dir" not in value or "dataset_role" not in value:
            continue
        summaries.append(value)
    if not summaries:
        raise ValueError(f"No complete Clean Dev summary found in {log_path}")
    return summaries[-1]


def language_record(summary: Dict[str, Any], language: str) -> Dict[str, float]:
    value = summary["languages"][language]
    result: Dict[str, float] = {}
    for metric in METRICS:
        if metric not in value:
            raise ValueError(f"{language}.{metric} missing from summary")
        number = float(value[metric])
        if not math.isfinite(number):
            raise ValueError(f"{language}.{metric} is not finite")
        result[metric] = number
    return result


def parse_epoch(log_path: Path) -> int:
    match = re.search(r"epoch_(\d+)\.log$", log_path.name)
    if not match:
        raise ValueError(f"Cannot parse epoch from {log_path.name}")
    return int(match.group(1))


def choose_log_files(logs_root: Path, metrics_jsonl: str) -> Tuple[List[Path], Path]:
    metrics_name = Path(metrics_jsonl).name
    metrics_candidates = sorted(logs_root.rglob(metrics_name))
    if len(metrics_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one final-run metrics JSONL {metrics_name!r}, "
            f"found {len(metrics_candidates)}: {[str(path) for path in metrics_candidates]}"
        )
    metrics_path = metrics_candidates[0]
    prefix = metrics_path.stem.replace("_clean_dev_epoch_metrics", "")
    candidates = sorted(metrics_path.parent.glob(f"{prefix}_eval_epoch_*.log"))
    candidates = [
        path
        for path in candidates
        if "multiseed" not in str(path).lower() and "latest" not in str(path).lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No per-epoch evaluation logs found beside final-run metrics JSONL {metrics_path}"
        )
    by_epoch: Dict[int, List[Path]] = {}
    for path in candidates:
        by_epoch.setdefault(parse_epoch(path), []).append(path)
    duplicates = {epoch: paths for epoch, paths in by_epoch.items() if len(paths) != 1}
    if duplicates:
        details = "; ".join(
            f"epoch {epoch}: {[str(path) for path in paths]}"
            for epoch, paths in sorted(duplicates.items())
        )
        raise RuntimeError(f"Ambiguous source logs for {details}")
    return [paths[0] for _, paths in sorted(by_epoch.items())], metrics_path


def make_row(model: str, log_path: Path) -> Dict[str, Any]:
    summary = select_summary(log_path)
    epoch = parse_epoch(log_path)
    row: Dict[str, Any] = {
        "model": model,
        "epoch": epoch,
        "stage": summary.get("stage", ""),
        "checkpoint": summary.get("checkpoint", ""),
        "checkpoint_sha256": summary.get("checkpoint_sha256", ""),
        "report_dir": summary.get("output_dir", ""),
        "source_log": str(log_path),
        "source_log_sha256": sha256_file(log_path),
        "dataset_role": summary.get("dataset_role", ""),
        "selection_metric_in_source": summary.get("selection_metric", ""),
    }
    per_language = {language: language_record(summary, language) for language in LANGUAGES}
    for metric in METRICS:
        macro = mean(per_language[language][metric] for language in LANGUAGES)
        row[f"macro_{metric}"] = macro
        source_value = summary.get(
            {
                "cer": "macro_cer",
                "wer": "macro_wer",
                "one_minus_ned_macro": "macro_one_minus_ned",
                "line_accuracy": "macro_line_accuracy",
            }[metric]
        )
        if source_value is not None and not math.isclose(
            macro, float(source_value), rel_tol=0.0, abs_tol=1e-10
        ):
            raise ValueError(
                f"{model} epoch {epoch}: macro {metric} disagrees with source summary "
                f"({macro} != {source_value})"
            )
        for language in LANGUAGES:
            row[f"{language}_{metric}"] = per_language[language][metric]
    return row


def validate_metrics_jsonl(
    metrics_path: Path,
    rows: Sequence[Dict[str, Any]],
    expected_epochs: Sequence[int],
) -> None:
    records: Dict[int, Dict[str, Any]] = {}
    for line_number, line in enumerate(
        metrics_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        epoch = int(value["epoch"])
        if epoch in records:
            raise ValueError(f"Duplicate epoch {epoch} in {metrics_path} at line {line_number}")
        records[epoch] = value
    expected = sorted(int(epoch) for epoch in expected_epochs)
    observed = sorted(int(row["epoch"]) for row in rows)
    if sorted(records) != expected:
        raise RuntimeError(
            f"{metrics_path.name} epochs do not match final summary: "
            f"expected={expected}, got={sorted(records)}"
        )
    if observed != expected:
        raise RuntimeError(
            f"Evaluation log epochs do not match metrics JSONL: "
            f"expected={expected}, got={observed}"
        )
    for row in rows:
        source_cer = float(records[int(row["epoch"])]["clean_dev_macro_cer"])
        if not math.isclose(row["macro_cer"], source_cer, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(
                f"{row['model']} epoch {row['epoch']}: evaluation Macro CER "
                f"{row['macro_cer']} != metrics JSONL {source_cer}"
            )


def audit_coverage(
    rows: Sequence[Dict[str, Any]],
    require_every_epoch: bool,
    expected_evaluation_epochs: Sequence[int],
) -> Dict[str, Any]:
    epochs = sorted(int(row["epoch"]) for row in rows)
    if len(set(epochs)) != len(epochs):
        raise ValueError("Duplicate epoch rows after log discovery")
    expected = list(range(epochs[0], epochs[-1] + 1))
    missing_in_range = sorted(set(expected) - set(epochs))
    expected_evaluation_epochs = sorted(int(epoch) for epoch in expected_evaluation_epochs)
    missing_expected = sorted(set(expected_evaluation_epochs) - set(epochs))
    unexpected = sorted(set(epochs) - set(expected_evaluation_epochs))
    if missing_expected or unexpected:
        raise RuntimeError(
            "Per-epoch logs do not match final summary evaluated_epochs: "
            f"missing={missing_expected}, unexpected={unexpected}"
        )
    if require_every_epoch and missing_in_range:
        raise RuntimeError(
            f"Missing epochs {missing_in_range}; existing logs are not an every-epoch audit"
        )
    return {
        "first_epoch": epochs[0],
        "last_epoch": epochs[-1],
        "observed_epoch_count": len(epochs),
        "expected_integer_epoch_count_in_range": len(expected),
        "expected_evaluation_epochs": expected_evaluation_epochs,
        "missing_epochs_in_range": missing_in_range,
        "complete_every_integer_epoch": not missing_in_range,
        "matches_final_summary_evaluated_epochs": True,
    }


def selection_row(model: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_cer = min(rows, key=lambda row: (row["macro_cer"], row["epoch"]))
    by_line = max(rows, key=lambda row: (row["macro_line_accuracy"], -row["epoch"]))
    result: Dict[str, Any] = {
        "model": model,
        "observed_epoch_count": len(rows),
        "cer_selected_epoch": by_cer["epoch"],
        "cer_selected_macro_cer": by_cer["macro_cer"],
        "cer_selected_macro_wer": by_cer["macro_wer"],
        "cer_selected_macro_one_minus_ned": by_cer["macro_one_minus_ned_macro"],
        "cer_selected_macro_line_accuracy": by_cer["macro_line_accuracy"],
        "line_accuracy_at_cer_selected_pp": 100.0 * by_cer["macro_line_accuracy"],
        "line_accuracy_max_epoch": by_line["epoch"],
        "line_accuracy_max_macro_line_accuracy": by_line["macro_line_accuracy"],
        "line_accuracy_max_macro_cer": by_line["macro_cer"],
        "line_accuracy_max_macro_wer": by_line["macro_wer"],
        "line_accuracy_max_macro_one_minus_ned": by_line["macro_one_minus_ned_macro"],
        "cer_at_line_accuracy_max_pp": 100.0 * by_line["macro_cer"],
        "line_accuracy_gain_max_over_cer_selected_pp": 100.0
        * (by_line["macro_line_accuracy"] - by_cer["macro_line_accuracy"]),
        "cer_penalty_at_line_accuracy_max_pp": 100.0
        * (by_line["macro_cer"] - by_cer["macro_cer"]),
        "tie_breaking": "earliest_epoch",
        "cer_selected_source_log": by_cer["source_log"],
        "line_accuracy_max_source_log": by_line["source_log"],
    }
    return result


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-specific
        (output_dir / "PLOTS_NOT_CREATED.txt").write_text(
            f"matplotlib unavailable: {exc}\n", encoding="utf-8"
        )
        return []

    plot_paths = []
    colors = {"B1": "#1f77b4", "M1": "#ff7f0e", "M2": "#2ca02c", "M3": "#d62728", "Full": "#9467bd"}
    for metric, filename, title, ylabel in (
        ("macro_cer", "epoch_cer.png", "Clean Dev Macro CER by observed epoch", "Macro CER (%)"),
        (
            "macro_line_accuracy",
            "epoch_line_accuracy.png",
            "Clean Dev Macro Line Accuracy by observed epoch",
            "Macro Line Accuracy (%)",
        ),
    ):
        figure, axis = plt.subplots(figsize=(9, 5.5), dpi=160)
        for model in MODEL_SPECS:
            model_rows = sorted((row for row in rows if row["model"] == model), key=lambda row: row["epoch"])
            axis.plot(
                [row["epoch"] for row in model_rows],
                [100.0 * row[metric] for row in model_rows],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=model,
                color=colors[model],
            )
        axis.set_xlabel("Epoch (observed evaluation checkpoints)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, ncol=3)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path)
        plt.close(figure)
        plot_paths.append(str(path))
    return plot_paths


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    logs_root = (args.logs_root or root / "04_model_training" / "logs").resolve()
    output_dir = (args.output_dir or root / "05_evaluation" / "epoch_selection_audit_v1").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    coverage: Dict[str, Any] = {}
    source_files: Dict[str, Dict[str, Any]] = {}
    for model, pattern in MODEL_SPECS.items():
        final_summary_path = root / "04_model_training" / "eval_reports" / pattern
        if not final_summary_path.is_file():
            raise FileNotFoundError(f"Final summary missing for {model}: {final_summary_path}")
        final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
        training_control = final_summary.get("training_control", {})
        metrics_jsonl = training_control.get("metrics_jsonl")
        expected_evaluation_epochs = training_control.get("evaluated_epochs")
        if not metrics_jsonl or not expected_evaluation_epochs:
            raise ValueError(
                f"Final summary for {model} lacks training_control.metrics_jsonl or evaluated_epochs"
            )
        log_files, metrics_path = choose_log_files(logs_root, metrics_jsonl)
        source_files[model] = {
            "final_summary": str(final_summary_path),
            "final_summary_sha256": sha256_file(final_summary_path),
            "metrics_jsonl": str(metrics_path),
            "metrics_jsonl_sha256": sha256_file(metrics_path),
            "eval_logs": [str(path) for path in log_files],
        }
        model_rows = [make_row(model, path) for path in log_files]
        validate_metrics_jsonl(metrics_path, model_rows, expected_evaluation_epochs)
        coverage[model] = audit_coverage(
            model_rows, args.require_every_epoch, expected_evaluation_epochs
        )
        all_rows.extend(sorted(model_rows, key=lambda row: row["epoch"]))

    all_rows.sort(key=lambda row: (list(MODEL_SPECS).index(row["model"]), row["epoch"]))
    selections = [
        selection_row(model, [row for row in all_rows if row["model"] == model])
        for model in MODEL_SPECS
    ]
    plots = write_plots(output_dir, all_rows)
    script_path = Path(__file__).resolve()
    manifest = {
        "status": "EPOCH_SELECTION_AUDIT_OK",
        "protocol": {
            "checkpoint_selection": "argmin_epoch_macro_CER_clean_dev",
            "macro_definition": "mean_of_zh_ug_kk",
            "primary_metric": "macro_cer",
            "secondary_metrics": ["macro_line_accuracy", "macro_wer", "macro_one_minus_ned"],
            "clean_dev_only": True,
            "corrupted_dev_used": False,
            "test_used": False,
            "tie_breaking": "earliest_epoch",
            "interpolation": False,
        },
        "source": {
            "logs_root": str(logs_root),
            "script_sha256": sha256_file(script_path),
            "models": MODEL_SPECS,
            "files": source_files,
        },
        "coverage": coverage,
        "outputs": {
            "epoch_metrics_csv": str(output_dir / "epoch_metrics.csv"),
            "selection_summary_csv": str(output_dir / "selection_summary.csv"),
            "selection_summary_json": str(output_dir / "selection_summary.json"),
            "epoch_metrics_json": str(output_dir / "epoch_metrics.json"),
            "plots": plots,
        },
        "notes": [
            "Rows are completed Clean Dev evaluations parsed from their raw logs.",
            "Missing integer epochs are reported and are not filled by interpolation.",
            "The current historical logs may be every-2-epoch evaluations rather than every-epoch evaluations.",
        ],
    }
    write_csv(output_dir / "epoch_metrics.csv", all_rows)
    write_csv(output_dir / "selection_summary.csv", selections)
    (output_dir / "epoch_metrics.json").write_text(
        json.dumps(all_rows, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "selection_summary.json").write_text(
        json.dumps(selections, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "AUDIT_COMPLETE").write_text("EPOCH_SELECTION_AUDIT_OK\n", encoding="ascii")

    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    print("\nSelection summary:")
    print(json.dumps(selections, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"EPOCH_SELECTION_AUDIT_FAILED: {exc}", file=sys.stderr)
        raise
