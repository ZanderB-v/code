#!/usr/bin/env python3
"""Rescore frozen per-sample Clean Dev predictions against Clean Dev V2.

This script never runs inference, training, Corrupted Dev, or Test. Predictions
are immutable evidence; only the model-blind verified GT and normalization V2
are used to recompute metrics and checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from metrics_v1 import finalize_metric_bucket, update_metric_bucket


LANGS = ("zh", "ug", "kk")
SPECS = {
    "B1": "04_model_training/eval_reports/b1_full_s50_to_target_dev_epoch_*",
    "M1": "04_model_training/eval_reports/svtrv2_s_m1_dual_order_s50_to_target_dev_epoch_*",
    "M2_alpha030": "04_model_training/eval_reports/svtrv2_s_m2_alpha_030_s50_to_target_dev_epoch_*",
    "M3_alpha015": "04_model_training/eval_reports/svtrv2_s_m3_dual_order_s50_to_target_dev_epoch_*",
    "M3_alpha020": "04_model_training/eval_reports/svtrv2_s_m3_alpha020_dual_order_s50_to_target_dev_epoch_*",
    "M3_alpha025": "04_model_training/eval_reports/svtrv2_s_m3_alpha025_dual_order_s50_to_target_dev_epoch_*",
    "M3_alpha030": "04_model_training/eval_reports/semantic_direction_v2/all_epoch_dev/M3/epoch_*",
    "B1_scale_S25": "04_model_training/eval_reports/b1_scale_step_v1_s25_to_target_dev_epoch_*",
    "B1_scale_S50": "04_model_training/eval_reports/b1_scale_step_v1_s50_to_target_dev_epoch_*",
}
PREDICTION_FILES = {
    "zh": Path("zh/predictions.jsonl"),
    "ug": Path("ug_logical/predictions_logical.jsonl"),
    "kk": Path("kk/predictions.jsonl"),
}
SOURCE_LANGUAGE_COUNTS = {"zh": 346, "ug": 295, "kk": 310}
ALPHA_BY_MODEL = {
    "M3_alpha015": 0.15,
    "M3_alpha020": 0.20,
    "M3_alpha025": 0.25,
    "M3_alpha030": 0.30,
}
SCALE_TIE_THRESHOLD = 0.0002  # 0.02 percentage points in CER.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("05_evaluation/clean_dev_v2_checkpoint_reselection"),
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("01_data_preparation/clean_dev_v2_verified"),
        help="Frozen Clean Dev protocol directory.",
    )
    parser.add_argument(
        "--frozen-manifest",
        default="frozen_clean_dev_v2_manifest.json",
        help="Frozen manifest filename inside --protocol-dir.",
    )
    parser.add_argument(
        "--protocol-label",
        default="Clean Dev V2",
        help="Human-readable protocol label used in generated provenance.",
    )
    parser.add_argument("--models", nargs="+", choices=tuple(SPECS), default=list(SPECS))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)$", path.name)
    if not match:
        raise ValueError(f"Cannot parse epoch from {path}")
    return int(match.group(1))


def load_normalizer(root: Path):
    protocol_scripts = root / "scripts" / "protocol"
    sys.path.insert(0, str(protocol_scripts))
    from text_normalization_v2 import PROTOCOL_ID, normalize_text_v2

    return normalize_text_v2, PROTOCOL_ID


def verify_protocol(root: Path, args: argparse.Namespace) -> tuple[Path, dict, dict[str, int]]:
    protocol = args.protocol_dir
    if not protocol.is_absolute():
        protocol = root / protocol
    frozen_path = protocol / args.frozen_manifest
    frozen = read_json(frozen_path)
    if not str(frozen.get("status", "")).endswith("_FROZEN"):
        raise ValueError(f"{args.protocol_label} is not frozen")
    if frozen.get("test_evaluated") is not False:
        raise ValueError(f"{args.protocol_label} violates the no-Test policy")
    for relative, expected in frozen["artifact_sha256"].items():
        path = protocol / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen {args.protocol_label} artifact mismatch: {relative}")
    implementation = root / "scripts" / "protocol" / "text_normalization_v2.py"
    expected_normalization_hash = (
        frozen.get("source", {}).get("normalization_implementation_sha256")
    )
    if expected_normalization_hash is None:
        normalization_yaml = protocol / "normalization_v2.yaml"
        for line in normalization_yaml.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("implementation_sha256:"):
                expected_normalization_hash = line.split(":", 1)[1].strip()
                break
    if not expected_normalization_hash:
        raise ValueError(f"{args.protocol_label} does not pin normalization_v2")
    if sha256(implementation) != expected_normalization_hash:
        raise ValueError("normalization_v2 implementation differs from the frozen hash")
    language_counts = frozen.get("language_evaluated_counts") or frozen.get("language_counts")
    if language_counts is None:
        raise ValueError(f"{args.protocol_label} has no evaluated language counts")
    language_counts = {lang: int(language_counts[lang]) for lang in LANGS}
    return protocol, frozen, language_counts


def load_labels(protocol: Path, language_counts: dict[str, int]) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for lang in LANGS:
        rows: dict[str, str] = {}
        path = protocol / "labels" / f"{lang}.txt"
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(f"Malformed V2 label {path}:{line_number}")
            image, text = line.split("\t", 1)
            image = image.replace("\\", "/")
            if image in rows:
                raise ValueError(f"Duplicate V2 image key: {image}")
            rows[image] = text
        if len(rows) != language_counts[lang]:
            raise ValueError(f"Unexpected {lang} evaluated row count: {len(rows)}")
        labels[lang] = rows
    return labels


def prediction_text(row: dict, lang: str) -> str:
    key = "pred_logical_text" if lang == "ug" else "pred_text"
    if key not in row:
        raise ValueError(f"Prediction row lacks {key}: {row.get('image')}")
    return str(row[key])


def inspect_report(report: Path, labels: dict[str, dict[str, str]]) -> dict:
    if "test" in report.name.lower() or "corrupt" in str(report).lower():
        raise ValueError(f"Forbidden non-clean-dev report: {report}")
    source_summary = report / "metrics_macro_summary.json"
    if not source_summary.is_file():
        raise FileNotFoundError(source_summary)
    evidence = {}
    for lang in LANGS:
        path = report / PREDICTION_FILES[lang]
        if not path.is_file():
            raise FileNotFoundError(path)
        images = []
        statuses = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                images.append(str(row.get("image", "")).replace("\\", "/"))
                statuses.append(row.get("status", "ok"))
                prediction_text(row, lang)
        if len(images) != SOURCE_LANGUAGE_COUNTS[lang]:
            raise ValueError(
                f"{report} {lang}: expected {SOURCE_LANGUAGE_COUNTS[lang]} source predictions, got {len(images)}"
            )
        if len(set(images)) != len(images):
            raise ValueError(f"{report} {lang}: duplicate image predictions")
        if not set(labels[lang]).issubset(set(images)):
            missing = sorted(set(labels[lang]) - set(images))[:5]
            raise ValueError(f"{report} {lang}: evaluated image set mismatch missing={missing}")
        if any(status != "ok" for status in statuses):
            raise ValueError(f"{report} {lang}: non-ok prediction rows present")
        evidence[lang] = {
            "path": str(path),
            "sha256": sha256(path),
            "source_prediction_rows": len(images),
            "evaluated_rows": len(labels[lang]),
            "excluded_prediction_rows": len(images) - len(labels[lang]),
        }
    return {
        "report": str(report),
        "source_summary": str(source_summary),
        "source_summary_sha256": sha256(source_summary),
        "predictions": evidence,
    }


def discover(root: Path, model: str, labels: dict[str, dict[str, str]]) -> list[dict]:
    reports = sorted(root.glob(SPECS[model]), key=parse_epoch)
    if not reports:
        raise FileNotFoundError(f"No Clean Dev epoch reports for {model}: {SPECS[model]}")
    epochs = [parse_epoch(path) for path in reports]
    if len(epochs) != len(set(epochs)):
        raise ValueError(f"Duplicate epochs for {model}: {epochs}")
    return [{"epoch": parse_epoch(path), **inspect_report(path, labels)} for path in reports]


def score_report(
    report: Path,
    labels: dict[str, dict[str, str]],
    normalize_text,
) -> tuple[dict, dict[str, list[dict]]]:
    per_language = {}
    rescored_rows: dict[str, list[dict]] = {}
    for lang in LANGS:
        metrics = defaultdict(float)
        rows = []
        path = report / PREDICTION_FILES[lang]
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line)
                image = str(source["image"]).replace("\\", "/")
                if image not in labels[lang]:
                    continue
                gt = normalize_text(labels[lang][image])
                pred = normalize_text(prediction_text(source, lang))
                update_metric_bucket(metrics, gt, pred)
                rows.append(
                    {
                        "image": image,
                        "language": lang,
                        "gt_text_v2": gt,
                        "pred_text": pred,
                        "score": source.get("score"),
                    }
                )
        per_language[lang] = finalize_metric_bucket(metrics)
        rescored_rows[lang] = rows
    result = {
        "macro_cer": sum(per_language[lang]["cer"] for lang in LANGS) / 3,
        "macro_wer": sum(per_language[lang]["wer"] for lang in LANGS) / 3,
        "macro_one_minus_ned": sum(
            per_language[lang]["one_minus_ned_macro"] for lang in LANGS
        ) / 3,
        "macro_line_accuracy": sum(
            per_language[lang]["line_accuracy"] for lang in LANGS
        ) / 3,
        "languages": per_language,
    }
    return result, rescored_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_row(model: str, rows: list[dict], protocol_label: str) -> dict:
    selected = min(rows, key=lambda row: (row["macro_cer"], row["epoch"]))
    return {
        "model": model,
        "cer_selected_epoch": selected["epoch"],
        "macro_cer": selected["macro_cer"],
        "macro_wer": selected["macro_wer"],
        "macro_one_minus_ned": selected["macro_one_minus_ned"],
        "macro_line_accuracy": selected["macro_line_accuracy"],
        "source_report": selected["source_report"],
        "source_summary_sha256": selected["source_summary_sha256"],
        "selection_rule": f"argmin_epoch_{protocol_label.replace(' ', '_')}_Macro_CER",
        "tie_breaking": "earliest_epoch",
    }


def write_selected_predictions(
    output: Path,
    model: str,
    selected: dict,
    labels: dict[str, dict[str, str]],
    normalize_text,
) -> None:
    _, rows = score_report(Path(selected["source_report"]), labels, normalize_text)
    selected_dir = output / "selected_predictions" / model
    selected_dir.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        path = selected_dir / f"{lang}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows[lang]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def derive_final_decisions(selections: list[dict], protocol_label: str) -> dict:
    by_model = {row["model"]: row for row in selections}
    alpha_rows = [
        {**by_model[model], "alpha": alpha}
        for model, alpha in ALPHA_BY_MODEL.items()
    ]
    selected_alpha = min(
        alpha_rows,
        key=lambda row: (row["macro_cer"], row["alpha"]),
    )
    s25 = by_model["B1_scale_S25"]
    s50 = by_model["B1_scale_S50"]
    scale_difference = abs(s25["macro_cer"] - s50["macro_cer"])
    if scale_difference < SCALE_TIE_THRESHOLD:
        selected_scale = "S25"
        scale_reason = "absolute CER difference below 0.02 pp; prefer lower cost"
    else:
        selected_scale = min(
            (("S25", s25), ("S50", s50)),
            key=lambda item: item[1]["macro_cer"],
        )[0]
        scale_reason = f"lower {protocol_label} Macro CER"
    baseline = by_model["B1"]
    final_model = selected_alpha
    guardrail = {
        "cer_better_than_B1": final_model["macro_cer"] < baseline["macro_cer"],
        "line_accuracy_better_than_B1": (
            final_model["macro_line_accuracy"]
            > baseline["macro_line_accuracy"]
        ),
    }
    return {
        "status": f"{protocol_label.upper().replace(' ', '_')}_FINAL_DECISIONS_DERIVED",
        "controlled_ablation": [
            by_model[name]
            for name in ("B1", "M1", "M2_alpha030", "M3_alpha030")
        ],
        "alpha_selection": {
            "candidates": alpha_rows,
            "selected_alpha": selected_alpha["alpha"],
            "selected_model": selected_alpha["model"],
            "selection_rule": f"minimum {protocol_label} Macro CER",
            "tie_breaking": "lower alpha",
        },
        "scale_selection": {
            "candidates": {"S25": s25, "S50": s50},
            "absolute_cer_difference": scale_difference,
            "tie_threshold": SCALE_TIE_THRESHOLD,
            "selected_scale": selected_scale,
            "selection_reason": scale_reason,
        },
        "final_candidate_vs_B1_guardrail": guardrail,
        "final_candidate_accepted": all(guardrail.values()),
        "test_evaluated": False,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    protocol, frozen, language_counts = verify_protocol(root, args)
    normalize_text, normalization_protocol = load_normalizer(root)
    labels = load_labels(protocol, language_counts)

    preflight = {}
    preflight_errors = {}
    for model in args.models:
        try:
            preflight[model] = discover(root, model, labels)
        except Exception as error:
            preflight_errors[model] = f"{type(error).__name__}: {error}"
    if preflight_errors:
        failure = {
            "status": f"{args.protocol_label.upper().replace(' ', '_')}_RESELECTION_PREFLIGHT_FAILED",
            "available_models": {
                model: [item["epoch"] for item in items]
                for model, items in preflight.items()
            },
            "errors": preflight_errors,
            "training_started": False,
            "test_evaluated": False,
        }
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        raise RuntimeError(
            f"{args.protocol_label} reselection inputs are incomplete; see preflight errors"
        )
    preflight_summary = {
        "status": f"{args.protocol_label.upper().replace(' ', '_')}_RESELECTION_PREFLIGHT_OK",
        "models": {
            model: {
                "source_glob": SPECS[model],
                "epochs": [item["epoch"] for item in items],
                "report_count": len(items),
            }
            for model, items in preflight.items()
        },
        "frozen_clean_dev_manifest_sha256": sha256(protocol / args.frozen_manifest),
        "evaluated_language_counts": language_counts,
        "test_evaluated": False,
        "training_started": False,
        "errors": [],
    }
    if args.preflight_only:
        print(json.dumps(preflight_summary, ensure_ascii=False, indent=2))
        return

    if output.exists():
        if not args.replace:
            raise FileExistsError(f"Output exists; use --replace: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    all_rows = []
    selections = []
    for model, reports in preflight.items():
        model_rows = []
        model_dir = output / "models" / model
        model_dir.mkdir(parents=True)
        for evidence in reports:
            report = Path(evidence["report"])
            metrics, _ = score_report(report, labels, normalize_text)
            row = {
                "model": model,
                "epoch": evidence["epoch"],
                **{key: metrics[key] for key in (
                    "macro_cer", "macro_wer", "macro_one_minus_ned", "macro_line_accuracy"
                )},
                "source_report": str(report),
                "source_summary_sha256": evidence["source_summary_sha256"],
            }
            for lang in LANGS:
                for metric in ("cer", "wer", "one_minus_ned_macro", "line_accuracy"):
                    row[f"{lang}_{metric}"] = metrics["languages"][lang][metric]
            model_rows.append(row)
            (model_dir / f"epoch_{evidence['epoch']:04d}.json").write_text(
                json.dumps(
                    {
                        **row,
                        "languages": metrics["languages"],
                        "prediction_evidence": evidence["predictions"],
                        "metric_normalization": normalization_protocol,
                        "test_evaluated": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        selection = selected_row(model, model_rows, args.protocol_label)
        selections.append(selection)
        all_rows.extend(model_rows)
        write_selected_predictions(output, model, selection, labels, normalize_text)

    write_csv(output / "epoch_metrics.csv", all_rows)
    write_csv(output / "selection_summary.csv", selections)
    (output / "epoch_metrics.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "selection_summary.json").write_text(
        json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    decisions = derive_final_decisions(selections, args.protocol_label)
    (output / "final_decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": f"{args.protocol_label.upper().replace(' ', '_')}_CHECKPOINT_RESELECTION_COMPLETE",
        "protocol": {
            "clean_dev": frozen["protocol_id"],
            "normalization": normalization_protocol,
            "selection": f"argmin_epoch_{args.protocol_label.replace(' ', '_')}_Macro_CER",
            "tie_breaking": "earliest_epoch",
            "predictions_reused_without_modification": True,
            "inference_rerun": False,
            "training_started": False,
            "corrupted_dev_used": False,
            "test_evaluated": False,
        },
        "preflight": preflight_summary,
        "selection": selections,
        "final_decisions": decisions,
        "alpha_grid_models": [
            "M3_alpha015", "M3_alpha020", "M3_alpha025", "M3_alpha030"
        ],
        "scale_models": ["B1_scale_S25", "B1_scale_S50"],
        "formal_new_module_training_allowed": False,
        "next_action": "review_reselection_before_error_profile_or_new_training",
    }
    (output / "reselection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "RESELECTION_COMPLETE").write_text(
        f"{args.protocol_label.upper().replace(' ', '_')}_CHECKPOINT_RESELECTION_COMPLETE\n",
        encoding="ascii",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
