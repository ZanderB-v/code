#!/usr/bin/env python3
"""Evaluate frozen Clean/Corrupted Dev with one model load per checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_SCRIPTS = SCRIPT_DIR.parent / "svtrv2"
sys.path.insert(0, str(PROJECT_SCRIPTS))

from metrics_v1 import finalize_metric_bucket, update_metric_bucket  # noqa: E402
from p1_msr_protocol import EVAL_INFERENCE_BATCH_SIZE  # noqa: E402
from p1_msr_protocol import PROTOCOL_ID as PREPROCESS_PROTOCOL  # noqa: E402
from p1_msr_protocol import normalize_spaces, u2_text  # noqa: E402

from build_corruption_protocol import (  # noqa: E402
    DEFAULT_ROOT,
    LANGUAGES,
    PROTOCOL_ID,
)
from straug12_deterministic import CORRUPTIONS, SEVERITIES, file_sha256  # noqa: E402


MODEL_SPECS = {
    "e0": {
        "display_name": "E0 random initialization + target train",
        "config": "04_model_training/configs/svtrv2_s_e0_random_target_only.yml",
        "checkpoint": (
            "04_model_training/runs/svtrv2_s_e0_random_target_only/"
            "best_clean_dev_macro_cer.pth"
        ),
        "clean_labels": "04_model_training/datasets/e0_random_target_only/labels",
        "clean_reference": (
            "04_model_training/eval_reports/"
            "e0_random_target_final_summary.json"
        ),
        "prediction_branch": "auto",
    },
    "e1": {
        "display_name": "E1 Union14M initialization + target train",
        "config": "04_model_training/configs/svtrv2_s_e1_target_only.yml",
        "checkpoint": (
            "04_model_training/runs/svtrv2_s_e1_target_only/"
            "best_clean_dev_macro_cer.pth"
        ),
        "clean_labels": "04_model_training/datasets/e1_target_only/labels",
        "clean_reference": (
            "04_model_training/eval_reports/"
            "e1_target_only_final_summary.json"
        ),
        "prediction_branch": "auto",
    },
    "b0": {
        "display_name": "B0/E5 S50 pretrain + target finetune, RCTC",
        "config": "04_model_training/configs/svtrv2_s_e5_d2_to_target.yml",
        "checkpoint": (
            "04_model_training/runs/svtrv2_s_e5_d2_to_target/"
            "best_clean_dev_macro_cer.pth"
        ),
        "clean_labels": "04_model_training/datasets/e1_target_only/labels",
        "clean_reference": (
            "04_model_training/eval_reports/"
            "e5_d2_to_target_final_summary.json"
        ),
        "prediction_branch": "auto",
    },
    "b1": {
        "display_name": "B1 full SVTRv2 S50 pretrain + target finetune",
        "config": (
            "04_model_training/configs/"
            "svtrv2_s_b1_full_s50_to_target.yml"
        ),
        "checkpoint": (
            "04_model_training/runs/svtrv2_s_b1_full_s50_to_target/"
            "best_clean_dev_macro_cer.pth"
        ),
        "clean_labels": "04_model_training/datasets/e1_target_only/labels",
        "clean_reference": (
            "04_model_training/eval_reports/"
            "b1_full_s50_to_target_final_summary.json"
        ),
        "prediction_branch": "ctc",
    },
}


# GPU kernels can change one or two decoded characters between otherwise
# identical inference runs. Keep the tolerance far below the batching bug this
# guard is designed to catch, while requiring provenance and corpus cardinality
# to reproduce exactly.
CLEAN_REPRODUCTION_ABS_TOLERANCE = 5e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("e0", "e1", "e5", "b0", "b1"),
        required=True,
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Must remain 1 for frozen P1/MSR evaluation. Mixed-aspect batches "
            "are padded to the batch maximum width by OpenRecognizer and do "
            "not reproduce the per-sample checkpoint-selection protocol."
        ),
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def verify_frozen_artifacts(
    root: Path,
    protocol_root: Path,
    frozen: dict[str, Any],
    frozen_hashes: dict[str, Any],
) -> None:
    """Reject evaluation when frozen data or inference code has drifted."""

    expected = frozen_hashes.get("files", {})
    paths = {
        "corruption_protocol_v1.json": (
            protocol_root / "corruption_protocol_v1.json"
        ),
        "dev_manifest.jsonl": protocol_root / "dev_manifest.jsonl",
        "test_manifest.jsonl": protocol_root / "test_manifest.jsonl",
        "dev_build_summary.json": protocol_root / "dev_build_summary.json",
        "test_build_summary.json": protocol_root / "test_build_summary.json",
        "dev_calibration_review.html": (
            protocol_root / "dev_calibration_review.html"
        ),
        "pilot_review_csv": (
            root
            / "05_evaluation"
            / "corruption_protocol_v1_pilot100_v2"
            / "human_review"
            / "corruption_pilot100_v2_review.csv"
        ),
        "straug12_deterministic.py": (
            root / "scripts" / "corruption" / "straug12_deterministic.py"
        ),
        "build_corruption_protocol.py": (
            root / "scripts" / "corruption" / "build_corruption_protocol.py"
        ),
        "freeze_corruption_protocol.py": (
            root / "scripts" / "corruption" / "freeze_corruption_protocol.py"
        ),
        "evaluate_corruption_dev.py": Path(__file__).resolve(),
        "summarize_corruption_dev.py": (
            root / "scripts" / "corruption" / "summarize_corruption_dev.py"
        ),
        "metrics_v1.py": root / "scripts" / "svtrv2" / "metrics_v1.py",
        "p1_msr_protocol.py": (
            root / "scripts" / "svtrv2" / "p1_msr_protocol.py"
        ),
        "protocol_v2_manifest.json": (
            root
            / "00_docs"
            / "frozen_protocol_v2"
            / "protocol_v2_manifest.json"
        ),
        "target_metadata.csv": (
            root
            / "01_data_preparation"
            / "real_line_dataset_eval_reviewed"
            / "metadata.csv"
        ),
    }
    errors = []
    for name, path in paths.items():
        wanted = expected.get(name)
        if not wanted:
            errors.append(f"missing frozen hash entry: {name}")
        elif not path.is_file():
            errors.append(f"missing frozen artifact: {path}")
        else:
            actual = file_sha256(path)
            if actual != wanted:
                errors.append(
                    f"hash mismatch {name}: expected={wanted}, actual={actual}"
                )
    if errors:
        raise ValueError(
            "Frozen corruption evaluation provenance mismatch:\n"
            + "\n".join(errors)
        )

    revision = frozen.get("evaluation_revision") or {}
    if revision.get("preprocess_protocol") != PREPROCESS_PROTOCOL:
        raise ValueError(
            "Frozen corruption evaluator is not approved for the current "
            f"preprocessing protocol: {revision.get('preprocess_protocol')!r} "
            f"!= {PREPROCESS_PROTOCOL!r}"
        )


def read_labels(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(f"Bad label line at {path}:{line_no}")
            image, text = line.split("\t", 1)
            rows.append({"image": image.replace("\\", "/"), "logical_text": text})
    return rows


def load_openocr(openocr_root: Path):
    openocr_root = openocr_root.resolve()
    sys.path.insert(0, str(openocr_root))
    sys.path.insert(0, str(openocr_root / "tools"))
    from tools.engine.config import Config
    from tools.infer_rec import OpenRecognizer

    return Config, OpenRecognizer


def build_recognizer(
    root: Path,
    config_path: Path,
    checkpoint_path: Path,
    prediction_branch: str,
    device_id: int,
):
    Config, OpenRecognizer = load_openocr(root / "third_party" / "OpenOCR")
    cfg = Config(str(config_path)).cfg
    actual_protocol = cfg["Global"].get("preprocess_protocol")
    if actual_protocol != PREPROCESS_PROTOCOL:
        raise ValueError(
            f"Expected preprocessing {PREPROCESS_PROTOCOL}, got {actual_protocol}"
        )
    cfg["Global"]["checkpoints"] = str(checkpoint_path)
    cfg["Global"]["pretrained_model"] = None
    cfg["Global"]["use_amp"] = False
    cfg["Global"]["infer_img"] = None
    recognizer = OpenRecognizer(
        config=cfg,
        mode="server",
        backend="torch",
        use_gpu="true",
        numId=device_id,
    )
    if prediction_branch == "ctc":
        decoder = getattr(recognizer.model, "decoder", None)
        if decoder is None or not hasattr(decoder, "ctc_decoder"):
            raise ValueError("B1 requires a full GTCDecoder with ctc_decoder")
        decoder.infer_gtc = False
        from openrec.postprocess import build_post_process

        recognizer.post_process_class = build_post_process(
            {
                "name": "CTCLabelDecode",
                "character_dict_path": cfg["Global"]["character_dict_path"],
                "use_space_char": cfg["Global"].get("use_space_char", False),
            },
            cfg["Global"],
        )
    return recognizer


def predictions_for_directory(
    recognizer,
    directory: Path,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    results = recognizer(img_path=str(directory), batch_num=batch_size)
    mapping = {}
    for result in results:
        file_value = result.get("file")
        if not file_value:
            raise ValueError(f"Inference result has no file field: {result}")
        resolved = str(Path(file_value).resolve())
        mapping[resolved] = result
        mapping[Path(file_value).name] = result
    return mapping


def empty_bucket() -> defaultdict[str, float]:
    return defaultdict(float)


def macro_average(
    language_metrics: dict[str, dict[str, Any]],
    field: str,
) -> float | None:
    values = [
        language_metrics[language].get(field)
        for language in LANGUAGES
        if language_metrics[language].get(field) is not None
    ]
    return sum(values) / len(values) if values else None


def evaluate_condition(
    recognizer,
    condition_name: str,
    data_root: Path,
    labels_dir: Path,
    label_prefix: str,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_buckets = {language: empty_bucket() for language in LANGUAGES}
    ug_visual_bucket = empty_bucket()
    prediction_rows = []
    errors = []
    wall_start = time.perf_counter()

    for language in LANGUAGES:
        label_path = labels_dir / f"{label_prefix}_dev_{language}_logical.txt"
        rows = read_labels(label_path)
        parent_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            parent_groups[str(Path(row["image"]).parent)].append(row)

        result_map: dict[str, dict[str, Any]] = {}
        for parent in sorted(parent_groups):
            absolute_parent = data_root / parent
            result_map.update(
                predictions_for_directory(
                    recognizer,
                    absolute_parent,
                    batch_size,
                )
            )

        for row in rows:
            image_path = (data_root / row["image"]).resolve()
            result = result_map.get(str(image_path)) or result_map.get(
                image_path.name
            )
            if result is None:
                errors.append(f"Missing prediction for {image_path}")
                continue
            logical_gt = normalize_spaces(row["logical_text"])
            visual_pred = normalize_spaces(result.get("text") or "")
            if language == "ug":
                visual_gt = u2_text("ug", logical_gt)
                update_metric_bucket(
                    ug_visual_bucket,
                    visual_gt,
                    visual_pred,
                )
                logical_pred = u2_text("ug", visual_pred)
            else:
                visual_gt = logical_gt
                logical_pred = visual_pred
            update_metric_bucket(
                primary_buckets[language],
                logical_gt,
                logical_pred,
            )
            prediction_rows.append(
                {
                    "condition": condition_name,
                    "status": "ok",
                    "image": row["image"],
                    "language": language,
                    "logical_gt": logical_gt,
                    "visual_gt": visual_gt,
                    "visual_prediction": visual_pred,
                    "logical_prediction": logical_pred,
                    "score": result.get("score"),
                    "inference_seconds": result.get("elapse"),
                }
            )
        print(
            json.dumps(
                {
                    "condition": condition_name,
                    "language": language,
                    "rows": len(rows),
                    "errors": len(errors),
                }
            ),
            flush=True,
        )

    if errors:
        raise RuntimeError(
            json.dumps(
                {"condition": condition_name, "errors": errors[:100]},
                ensure_ascii=False,
                indent=2,
            )
        )
    language_metrics = {
        language: finalize_metric_bucket(primary_buckets[language])
        for language in LANGUAGES
    }
    language_metrics["ug"]["u2_visual_order_metrics"] = (
        finalize_metric_bucket(ug_visual_bucket)
    )
    summary = {
        "protocol_id": PROTOCOL_ID,
        "preprocess_protocol": PREPROCESS_PROTOCOL,
        "condition": condition_name,
        "samples": sum(
            language_metrics[language]["samples"] for language in LANGUAGES
        ),
        "macro_cer": macro_average(language_metrics, "cer"),
        "macro_wer": macro_average(language_metrics, "wer"),
        "macro_one_minus_ned": macro_average(
            language_metrics,
            "one_minus_ned_macro",
        ),
        "macro_line_accuracy": macro_average(
            language_metrics,
            "line_accuracy",
        ),
        "languages": language_metrics,
        "wall_seconds": time.perf_counter() - wall_start,
        "uyghur_reporting": {
            "primary": "heuristic logical-order metrics",
            "direct_visual_order": "languages.ug.u2_visual_order_metrics",
        },
    }
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = output_dir / "metrics_macro_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def reuse_completed_condition(
    output_dir: Path,
    condition_name: str,
) -> dict[str, Any] | None:
    summary_path = output_dir / "metrics_macro_summary.json"
    predictions_path = output_dir / "predictions.jsonl"
    if not summary_path.is_file() or not predictions_path.is_file():
        return None
    summary = read_json(summary_path)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "preprocess_protocol": PREPROCESS_PROTOCOL,
        "condition": condition_name,
        "samples": 951,
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if set(summary.get("languages", {})) != set(LANGUAGES):
        mismatches["languages"] = {
            "expected": sorted(LANGUAGES),
            "actual": sorted(summary.get("languages", {})),
        }
    if mismatches:
        print(
            "Ignoring incompatible cached condition: "
            + json.dumps(
                {
                    "summary": str(summary_path),
                    "mismatches": mismatches,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return None
    print(f"Reusing completed condition: {summary_path}", flush=True)
    return summary


def summarize_model(
    root: Path,
    model_key: str,
    model_spec: dict[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    condition_summaries: list[dict[str, Any]],
    output_root: Path,
    batch_size: int,
) -> dict[str, Any]:
    clean = next(
        summary
        for summary in condition_summaries
        if summary["condition"] == "clean"
    )
    clean_reference_path = root / model_spec["clean_reference"]
    clean_reference = read_json(clean_reference_path)
    provenance_mismatches = {}
    expected_config_sha256 = clean_reference.get("config_sha256")
    actual_config_sha256 = file_sha256(config_path)
    if expected_config_sha256 != actual_config_sha256:
        provenance_mismatches["config_sha256"] = {
            "expected": expected_config_sha256,
            "actual": actual_config_sha256,
        }
    expected_checkpoint_sha256 = clean_reference.get(
        "best_checkpoint_sha256"
    )
    actual_checkpoint_sha256 = file_sha256(checkpoint_path)
    if expected_checkpoint_sha256 != actual_checkpoint_sha256:
        provenance_mismatches["checkpoint_sha256"] = {
            "expected": expected_checkpoint_sha256,
            "actual": actual_checkpoint_sha256,
        }
    reference_dev = clean_reference.get("dev", {})
    if reference_dev.get("preprocess_protocol") != PREPROCESS_PROTOCOL:
        provenance_mismatches["preprocess_protocol"] = {
            "expected": reference_dev.get("preprocess_protocol"),
            "actual": PREPROCESS_PROTOCOL,
        }
    for language in LANGUAGES:
        expected_metrics = reference_dev.get("languages", {}).get(
            language,
            {},
        )
        actual_metrics = clean.get("languages", {}).get(language, {})
        for metric in ("samples", "chars"):
            expected_value = expected_metrics.get(metric)
            actual_value = actual_metrics.get(metric)
            if expected_value != actual_value:
                provenance_mismatches[f"{language}_{metric}"] = {
                    "expected": expected_value,
                    "actual": actual_value,
                }
    if provenance_mismatches:
        raise ValueError(
            f"{model_key} clean-dev provenance reproduction failed: "
            + json.dumps(provenance_mismatches, ensure_ascii=False)
        )
    expected_clean_cer = float(clean_reference["best_clean_dev_macro_cer"])
    actual_clean_cer = float(clean["macro_cer"])
    clean_reproduction_error = abs(actual_clean_cer - expected_clean_cer)
    if clean_reproduction_error > CLEAN_REPRODUCTION_ABS_TOLERANCE:
        raise ValueError(
            f"{model_key} clean-dev reproduction failed: "
            f"expected {expected_clean_cer:.15f} from "
            f"{clean_reference_path}, got {actual_clean_cer:.15f}. "
            f"Absolute tolerance is "
            f"{CLEAN_REPRODUCTION_ABS_TOLERANCE:.15f}. "
            "Do not summarize corrupted metrics until preprocessing and "
            "inference batching match checkpoint selection."
        )
    if clean_reproduction_error:
        print(
            json.dumps(
                {
                    "status": "clean_reproduction_within_tolerance",
                    "model": model_key,
                    "expected_macro_cer": expected_clean_cer,
                    "actual_macro_cer": actual_clean_cer,
                    "absolute_error": clean_reproduction_error,
                    "absolute_tolerance": (
                        CLEAN_REPRODUCTION_ABS_TOLERANCE
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    corrupted = [
        summary
        for summary in condition_summaries
        if summary["condition"] != "clean"
    ]
    by_corruption: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_severity: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for summary in corrupted:
        by_corruption[summary["corruption"]].append(summary)
        by_severity[summary["severity"]].append(summary)

    mean_corrupted_cer = sum(item["macro_cer"] for item in corrupted) / len(
        corrupted
    )
    absolute_increase = mean_corrupted_cer - clean["macro_cer"]
    relative_increase = (
        absolute_increase / clean["macro_cer"]
        if clean["macro_cer"]
        else None
    )
    result = {
        "status": "passed",
        "errors": [],
        "protocol_id": PROTOCOL_ID,
        "model_key": model_key,
        "display_name": model_spec["display_name"],
        "config": str(config_path),
        "config_sha256": actual_config_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": actual_checkpoint_sha256,
        "prediction_branch": model_spec["prediction_branch"],
        "inference_batch_size": batch_size,
        "batching_policy": "per_sample_P1_MSR_V3",
        "clean_reference": str(clean_reference_path),
        "clean_reference_sha256": file_sha256(clean_reference_path),
        "clean_reference_macro_cer": expected_clean_cer,
        "clean_reproduction_absolute_error": clean_reproduction_error,
        "clean_reproduction_absolute_tolerance": (
            CLEAN_REPRODUCTION_ABS_TOLERANCE
        ),
        "clean_reproduction_status": (
            "exact"
            if clean_reproduction_error == 0.0
            else "within_tolerance"
        ),
        "clean": clean,
        "corrupted_conditions": len(corrupted),
        "mean_corrupted_macro_cer": mean_corrupted_cer,
        "absolute_macro_cer_increase": absolute_increase,
        "relative_macro_cer_increase": relative_increase,
        "per_corruption": {
            corruption: {
                "mean_macro_cer": sum(
                    item["macro_cer"] for item in items
                )
                / len(items),
                "levels": {
                    str(item["severity"]): item for item in items
                },
            }
            for corruption, items in sorted(by_corruption.items())
        },
        "per_severity": {
            str(severity): {
                "mean_macro_cer": sum(
                    item["macro_cer"] for item in items
                )
                / len(items),
                "conditions": len(items),
            }
            for severity, items in sorted(by_severity.items())
        },
        "test_evaluated": False,
    }
    summary_path = output_root / "model_summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = output_root / "condition_metrics.csv"
    fields = (
        "model",
        "condition",
        "corruption",
        "severity",
        "macro_cer",
        "macro_wer",
        "macro_one_minus_ned",
        "macro_line_accuracy",
        "zh_cer",
        "ug_cer",
        "kk_cer",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in condition_summaries:
            writer.writerow(
                {
                    "model": model_key,
                    "condition": item["condition"],
                    "corruption": item.get("corruption", "clean"),
                    "severity": item.get("severity", 0),
                    "macro_cer": item["macro_cer"],
                    "macro_wer": item["macro_wer"],
                    "macro_one_minus_ned": item["macro_one_minus_ned"],
                    "macro_line_accuracy": item["macro_line_accuracy"],
                    "zh_cer": item["languages"]["zh"]["cer"],
                    "ug_cer": item["languages"]["ug"]["cer"],
                    "kk_cer": item["languages"]["kk"]["cer"],
                }
            )
    return result


def evaluate_model(
    root: Path,
    protocol_root: Path,
    model_key: str,
    device_id: int,
    batch_size: int,
    replace: bool,
) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    config_path = root / spec["config"]
    checkpoint_path = root / spec["checkpoint"]
    clean_labels = root / spec["clean_labels"]
    target_root = (
        root
        / "01_data_preparation"
        / "real_line_dataset_eval_reviewed"
    )
    for path in (config_path, checkpoint_path, clean_labels, target_root):
        if not path.exists():
            raise FileNotFoundError(path)
    output_root = protocol_root / "eval_reports" / model_key
    summary_path = output_root / "model_summary.json"
    if summary_path.exists() and not replace:
        print(f"Reusing completed summary: {summary_path}")
        return read_json(summary_path)
    if output_root.exists() and replace:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    recognizer = build_recognizer(
        root,
        config_path,
        checkpoint_path,
        spec["prediction_branch"],
        device_id,
    )
    condition_summaries = []
    clean_output = output_root / "clean"
    clean = None if replace else reuse_completed_condition(
        clean_output,
        "clean",
    )
    if clean is None:
        clean = evaluate_condition(
            recognizer,
            "clean",
            target_root,
            clean_labels,
            "target",
            clean_output,
            batch_size,
        )
    clean.update({"corruption": "clean", "severity": 0})
    condition_summaries.append(clean)

    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            condition_root = (
                protocol_root
                / "dev"
                / "conditions"
                / corruption
                / f"level_{severity}"
            )
            condition_name = f"{corruption}/level_{severity}"
            output_dir = (
                output_root
                / "corrupted"
                / corruption
                / f"level_{severity}"
            )
            summary = None if replace else reuse_completed_condition(
                output_dir,
                condition_name,
            )
            if summary is None:
                summary = evaluate_condition(
                    recognizer,
                    condition_name,
                    condition_root,
                    condition_root / "labels",
                    "corrupt",
                    output_dir,
                    batch_size,
                )
            summary.update(
                {
                    "corruption": corruption,
                    "severity": severity,
                }
            )
            condition_summaries.append(summary)
    del recognizer
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return summarize_model(
        root,
        model_key,
        spec,
        config_path,
        checkpoint_path,
        condition_summaries,
        output_root,
        batch_size,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size != EVAL_INFERENCE_BATCH_SIZE:
        raise ValueError(
            "P1/MSR corruption evaluation requires --batch-size "
            f"{EVAL_INFERENCE_BATCH_SIZE}. "
            "OpenRecognizer pads mixed-aspect batches to their maximum width, "
            "which changes predictions relative to checkpoint selection."
        )
    root = args.root.resolve()
    protocol_root = (
        args.protocol_root.resolve()
        if args.protocol_root
        else root / "05_evaluation" / "corruption_protocol_v1"
    )
    frozen_path = protocol_root / "corruption_protocol_v1.json"
    freeze_report_path = protocol_root / "freeze_report.json"
    for path in (frozen_path, freeze_report_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Corruption protocol must be frozen before evaluation: {path}"
            )
    frozen = read_json(frozen_path)
    report = read_json(freeze_report_path)
    if frozen.get("status") != "frozen" or report.get("status") != "passed":
        raise ValueError("Corruption protocol freeze did not pass")
    if frozen.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Protocol ID mismatch")
    if report.get("test_inference_run") is not False:
        raise ValueError("Test policy was violated")
    frozen_hashes = read_json(protocol_root / "frozen_hashes.json")
    verify_frozen_artifacts(root, protocol_root, frozen, frozen_hashes)

    normalized_models = []
    for model in args.models:
        model = "b0" if model == "e5" else model
        if model not in normalized_models:
            normalized_models.append(model)
    summaries = []
    for model_key in normalized_models:
        print(f"===== evaluating {model_key} on clean/corrupted dev =====", flush=True)
        summaries.append(
            evaluate_model(
                root,
                protocol_root,
                model_key,
                args.device_id,
                args.batch_size,
                args.replace,
            )
        )
    invocation = {
        "status": "passed",
        "models": normalized_models,
        "device_id": args.device_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "protocol_id": PROTOCOL_ID,
        "test_evaluated": False,
        "summaries": [
            str(protocol_root / "eval_reports" / key / "model_summary.json")
            for key in normalized_models
        ],
    }
    invocation_path = (
        protocol_root
        / "eval_reports"
        / f"invocation_{'_'.join(normalized_models)}.json"
    )
    invocation_path.parent.mkdir(parents=True, exist_ok=True)
    invocation_path.write_text(
        json.dumps(invocation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(invocation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
