#!/usr/bin/env python3
"""Paired M3 -> SCDL error migration on frozen Clean Dev V4 only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from metrics_v1 import edit_distance, finalize_metric_bucket, update_metric_bucket
from rescore_clean_dev_v2 import (
    LANGS,
    PREDICTION_FILES,
    inspect_report,
    load_labels,
    load_normalizer,
    prediction_text,
    score_report,
    sha256,
    verify_protocol,
)
from run_stage6_error_migration import levenshtein_alignment, operation_counts


PROTOCOL_ID = "M3_SCDL_CLEAN_DEV_V4_DIAGNOSTIC_V1"
REVIEW_CATEGORIES = (
    "similar_letter_substitution",
    "joining_form",
    "deletion",
    "insertion",
    "order_related",
    "diacritic_or_dot",
    "long_word",
    "ambiguous_image",
    "punctuation_or_normalization",
    "other",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--protocol-dir", type=Path,
        default=Path("01_data_preparation/clean_dev_v4_adjudicated"),
    )
    parser.add_argument(
        "--m3-predictions", type=Path,
        default=Path(
            "05_evaluation/clean_dev_v4_checkpoint_reselection/"
            "selected_predictions/M3_alpha015"
        ),
    )
    parser.add_argument(
        "--scdl-report", type=Path,
        default=Path(
            "04_model_training/eval_reports/"
            "svtrv2_s_m3_scdl_full_v1_dual_order_s50_to_target_best_clean_dev"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("05_evaluation/m3_scdl_diagnostic_v1"),
    )
    return parser.parse_args()


def absolute(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_m3(directory: Path, labels: dict, normalizer) -> tuple[dict, dict]:
    predictions = {}
    hashes = {}
    for lang in LANGS:
        path = directory / f"{lang}.jsonl"
        hashes[lang] = sha256(path)
        rows = {}
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            row = json.loads(line)
            image = str(row["image"]).replace("\\", "/")
            if image in rows:
                raise ValueError(f"Duplicate M3 prediction: {image}")
            if image not in labels[lang]:
                raise ValueError(f"M3 prediction outside V4: {image}")
            if normalizer(row["gt_text_v2"]) != normalizer(labels[lang][image]):
                raise ValueError(f"M3 V4 GT mismatch: {image}")
            rows[image] = normalizer(str(row["pred_text"]))
        if set(rows) != set(labels[lang]):
            raise ValueError(f"M3 prediction image set mismatch: {lang}")
        predictions[lang] = rows
    return predictions, hashes


def load_scdl(directory: Path, labels: dict, normalizer) -> dict:
    predictions = {}
    for lang in LANGS:
        rows = {}
        path = directory / PREDICTION_FILES[lang]
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            row = json.loads(line)
            image = str(row["image"]).replace("\\", "/")
            if image in labels[lang]:
                if row.get("status") != "ok" or image in rows:
                    raise ValueError(f"Invalid SCDL prediction: {image}")
                rows[image] = normalizer(prediction_text(row, lang))
        if set(rows) != set(labels[lang]):
            raise ValueError(f"SCDL prediction image set mismatch: {lang}")
        predictions[lang] = rows
    return predictions


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = absolute(root, args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing diagnostic/review: {output}")
    args.protocol_dir = absolute(root, args.protocol_dir)
    args.frozen_manifest = "frozen_clean_dev_v4_manifest.json"
    args.protocol_label = "Clean Dev V4"
    protocol, frozen, counts = verify_protocol(root, args)
    labels = load_labels(protocol, counts)
    normalizer, normalization_id = load_normalizer(root)
    m3_dir = absolute(root, args.m3_predictions)
    scdl_dir = absolute(root, args.scdl_report)
    evidence = inspect_report(scdl_dir, labels)
    scdl_metrics, _ = score_report(scdl_dir, labels, normalizer)
    m3, m3_hashes = load_m3(m3_dir, labels, normalizer)
    scdl = load_scdl(scdl_dir, labels, normalizer)
    selection_path = root / (
        "05_evaluation/clean_dev_v4_checkpoint_reselection/selection_summary.csv"
    )
    with selection_path.open("r", encoding="utf-8-sig", newline="") as handle:
        selections = [
            row for row in csv.DictReader(handle)
            if row["model"] == "M3_alpha015"
        ]
    if len(selections) != 1 or int(selections[0]["cer_selected_epoch"]) != 34:
        raise ValueError("M3 alpha=.15 Epoch 34 is not the frozen V4 selection")
    m3_metrics = {}
    for lang in LANGS:
        bucket = Counter()
        for image, gt_raw in labels[lang].items():
            update_metric_bucket(
                bucket, normalizer(gt_raw), m3[lang][image]
            )
        m3_metrics[lang] = finalize_metric_bucket(bucket)
    m3_macro_cer = sum(m3_metrics[lang]["cer"] for lang in LANGS) / 3
    if abs(m3_macro_cer - float(selections[0]["macro_cer"])) > 1e-12:
        raise ValueError("M3 predictions do not reproduce V4 selected CER")

    migration = []
    operations = []
    changed = []
    ug_review = []
    for lang in LANGS:
        counts_by_status = Counter()
        m3_ops = Counter()
        scdl_ops = Counter()
        for image, gt_raw in labels[lang].items():
            gt = normalizer(gt_raw)
            base = m3[lang][image]
            candidate = scdl[lang][image]
            base_ed = edit_distance(gt, base)
            candidate_ed = edit_distance(gt, candidate)
            if base_ed == 0 and candidate_ed == 0:
                status = "both_correct"
            elif base_ed > 0 and candidate_ed == 0:
                status = "recovered"
            elif base_ed == 0 and candidate_ed > 0:
                status = "regression"
            elif candidate_ed < base_ed:
                status = "both_wrong_scdl_lower_ed"
            elif candidate_ed > base_ed:
                status = "both_wrong_scdl_higher_ed"
            else:
                status = "both_wrong_equal_ed"
            counts_by_status[status] += 1
            base_counts = operation_counts(levenshtein_alignment(gt, base))
            candidate_counts = operation_counts(levenshtein_alignment(gt, candidate))
            m3_ops.update(base_counts)
            scdl_ops.update(candidate_counts)
            if status not in ("both_correct",):
                row = {
                    "language": lang,
                    "sample_id": Path(image).stem,
                    "image": image,
                    "status": status,
                    "gt": gt,
                    "m3_pred": base,
                    "scdl_pred": candidate,
                    "m3_ed": base_ed,
                    "scdl_ed": candidate_ed,
                    "m3_ed_group": "0" if base_ed == 0 else "1" if base_ed == 1 else "2+",
                    "scdl_ed_group": "0" if candidate_ed == 0 else "1" if candidate_ed == 1 else "2+",
                    "m3_sub": base_counts["sub"],
                    "m3_del": base_counts["del"],
                    "m3_ins": base_counts["ins"],
                    "scdl_sub": candidate_counts["sub"],
                    "scdl_del": candidate_counts["del"],
                    "scdl_ins": candidate_counts["ins"],
                }
                changed.append(row)
                if lang == "ug" and status == "regression":
                    image_file = (
                        root / "01_data_preparation/real_line_dataset_eval_reviewed"
                        / image
                    )
                    ug_review.append({
                        **row,
                        "local_image_path": str(image_file),
                        "local_image_exists": image_file.is_file(),
                        "manual_category": "",
                        "manual_note": "",
                    })
        migration.append({
            "language": lang,
            "samples": len(labels[lang]),
            **{name: counts_by_status[name] for name in (
                "recovered", "regression", "both_correct",
                "both_wrong_scdl_lower_ed", "both_wrong_scdl_higher_ed",
                "both_wrong_equal_ed",
            )},
            "net_recovered_minus_regression": (
                counts_by_status["recovered"] - counts_by_status["regression"]
            ),
        })
        for operation in ("sub", "del", "ins"):
            operations.append({
                "language": lang,
                "operation": operation,
                "m3_count": m3_ops[operation],
                "scdl_count": scdl_ops[operation],
                "scdl_minus_m3": scdl_ops[operation] - m3_ops[operation],
            })

    output.mkdir(parents=True)
    write_csv(output / "migration_by_language.csv", migration, list(migration[0]))
    write_csv(output / "edit_operation_changes.csv", operations, list(operations[0]))
    write_csv(output / "changed_error_rows.csv", changed, list(changed[0]))
    write_csv(
        output / "ug_regressions_review.csv", ug_review,
        list(changed[0]) + [
            "local_image_path", "local_image_exists",
            "manual_category", "manual_note",
        ],
    )
    summary = {
        "status": "M3_SCDL_V4_ERROR_MIGRATION_COMPLETE",
        "protocol_id": PROTOCOL_ID,
        "split": "Clean Dev V4",
        "evaluated_rows": sum(counts.values()),
        "normalization": normalization_id,
        "migration": migration,
        "edit_operation_changes": operations,
        "ug_regression_review_rows": len(ug_review),
        "valid_ug_manual_categories": list(REVIEW_CATEGORIES),
        "scdl_v4_metrics": scdl_metrics,
        "m3_v4_macro_cer_verified": m3_macro_cer,
        "provenance": {
            "frozen_v4_manifest_sha256": sha256(
                protocol / args.frozen_manifest
            ),
            "m3_selection_summary_sha256": sha256(selection_path),
            "m3_prediction_sha256": m3_hashes,
            "scdl_prediction_sha256": evidence["predictions"],
            "scdl_source_summary_sha256": evidence["source_summary_sha256"],
        },
        "training_run": False,
        "corrupted_dev_used": False,
        "test_evaluated": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
