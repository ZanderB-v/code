#!/usr/bin/env python3
"""Evaluate frozen B0/B1/B2/M1/M2/M3/Full checkpoints on Corrupted Dev.

The frozen evaluator is imported without modification. This file only supplies
the method registry and verifies it against Method Design Freeze V1.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_corruption_dev as frozen_eval  # noqa: E402


METHOD_KEYS = ("b0", "b1", "b2", "m1", "m2", "m3", "full")
BINDING_ID = "METHOD_CORRUPTION_EVALUATION_BINDING_V1"
OUTPUT_PREFIX = "method_binding_v1_"
IMMUTABLE_CORRUPTION_KEYS = (
    "corruption_protocol_v1.json",
    "dev_manifest.jsonl",
    "test_manifest.jsonl",
    "dev_build_summary.json",
    "test_build_summary.json",
    "dev_calibration_review.html",
    "pilot_review_csv",
    "straug12_deterministic.py",
    "build_corruption_protocol.py",
    "freeze_corruption_protocol.py",
    "metrics_v1.py",
    "protocol_v2_manifest.json",
    "target_metadata.csv",
)
DISPLAY_NAMES = {
    "b0": "B0 SVTRv2-S + RCTC",
    "b1": "B1 Full SVTRv2-S",
    "b2": "B2 logical-order direction-conflict diagnostic",
    "m1": "M1 Dual-Order Semantic Guidance",
    "m2": "M2 Cross-Order Consistency (alpha=0.30)",
    "m3": "M3 Script-Conditioned Adaptation",
    "full": "Full M3 + Local Direction Conditioning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path)
    parser.add_argument("--method-freeze-root", type=Path)
    parser.add_argument("--models", nargs="+", choices=METHOD_KEYS)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--freeze-evaluation-binding", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_recorded_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    normalized = value.replace("\\", "/")
    for marker in (
        "/00_docs/",
        "/01_data_preparation/",
        "/05_evaluation/",
    ):
        if marker in normalized:
            return root / marker.strip("/") / normalized.split(marker, 1)[1]
    return path


def verify_file(root: Path, relative: str, expected_sha256: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = frozen_eval.file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Frozen method artifact hash mismatch: {relative}: "
            f"expected={expected_sha256}, actual={actual}"
        )
    return path


def load_method_specs(root: Path, freeze_root: Path) -> dict[str, dict[str, Any]]:
    verification = read_json(freeze_root / "verification_report.json")
    if verification.get("status") != "passed" or verification.get("errors") != []:
        raise ValueError("Method Design Freeze V1 verification did not pass")
    manifest = read_json(freeze_root / "method_design_v1_manifest.json")
    if manifest.get("status") != "frozen" or not manifest.get("development_closed"):
        raise ValueError("Method Design Freeze V1 is not closed")
    if float(manifest.get("official_m2_consistency_weight")) != 0.30:
        raise ValueError("Official M2 must use consistency_weight=0.30")

    specs: dict[str, dict[str, Any]] = {}
    for method in METHOD_KEYS:
        target = manifest["methods"][method]["target"]
        verify_file(root, target["config"], target["config_sha256"])
        verify_file(root, target["checkpoint"], target["checkpoint_sha256"])
        verify_file(root, target["summary"], target["summary_sha256"])
        config = yaml.safe_load((root / target["config"]).read_text(encoding="utf-8-sig"))
        ctc_order = config.get("Global", {}).get("uyghur_ctc_order", "visual")
        expected_order = "logical" if method == "b2" else "visual"
        if ctc_order != expected_order:
            raise ValueError(
                f"{method} Uyghur CTC order mismatch: "
                f"expected={expected_order}, actual={ctc_order}"
            )
        specs[method] = {
            "display_name": DISPLAY_NAMES[method],
            "config": target["config"],
            "checkpoint": target["checkpoint"],
            "clean_labels": "04_model_training/datasets/e1_target_only/labels",
            "clean_reference": target["summary"],
            "prediction_branch": "ctc" if method != "b0" else "auto",
            "uyghur_ctc_order": ctc_order,
        }
    return specs


def logical_ug_condition_adapter(original):
    """Reuse frozen inference, then score B2's raw UG output as logical order."""

    def evaluate(*args, **kwargs):
        summary = original(*args, **kwargs)
        output_dir = kwargs.get("output_dir")
        if output_dir is None:
            output_dir = args[5]
        output_dir = Path(output_dir)
        prediction_path = output_dir / "predictions.jsonl"
        rows = []
        bucket = defaultdict(float)
        with prediction_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                row = json.loads(line)
                if row["language"] == "ug":
                    logical_prediction = frozen_eval.normalize_spaces(
                        row.get("visual_prediction") or ""
                    )
                    frozen_eval.update_metric_bucket(
                        bucket,
                        row["logical_gt"],
                        logical_prediction,
                    )
                    row["logical_prediction"] = logical_prediction
                    row["visual_gt"] = row["logical_gt"]
                rows.append(row)
        summary["languages"]["ug"] = frozen_eval.finalize_metric_bucket(bucket)
        for field, target in (
            ("cer", "macro_cer"),
            ("wer", "macro_wer"),
            ("one_minus_ned_macro", "macro_one_minus_ned"),
            ("line_accuracy", "macro_line_accuracy"),
        ):
            summary[target] = frozen_eval.macro_average(summary["languages"], field)
        summary["uyghur_ctc_order"] = "logical"
        summary["uyghur_reporting"] = {
            "primary": "direct Unicode logical-order CTC metrics",
            "direct_visual_order": None,
        }
        with prediction_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (output_dir / "metrics_macro_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary

    return evaluate


def logical_ug_reuse_adapter(original):
    """Reject stale B2 conditions produced with the visual-order adapter."""

    def reuse(*args, **kwargs):
        summary = original(*args, **kwargs)
        if summary is not None and summary.get("uyghur_ctc_order") != "logical":
            return None
        return summary

    return reuse


def current_binding_files(root: Path, protocol_root: Path, freeze_root: Path) -> dict[str, str]:
    paths = {
        "corruption_protocol_v1.json": protocol_root / "corruption_protocol_v1.json",
        "dev_manifest.jsonl": protocol_root / "dev_manifest.jsonl",
        "test_manifest.jsonl": protocol_root / "test_manifest.jsonl",
        "method_design_v1_manifest.json": freeze_root / "method_design_v1_manifest.json",
        "evaluate_corruption_dev.py": SCRIPT_DIR / "evaluate_corruption_dev.py",
        "evaluate_frozen_methods_corruption_dev.py": Path(__file__).resolve(),
        "summarize_frozen_methods_corruption.py": SCRIPT_DIR / "summarize_frozen_methods_corruption.py",
        "metrics_v1.py": root / "scripts" / "svtrv2" / "metrics_v1.py",
        "p1_msr_protocol.py": root / "scripts" / "svtrv2" / "p1_msr_protocol.py",
    }
    result = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = frozen_eval.file_sha256(path)
    return result


def verify_or_freeze_binding(
    root: Path,
    protocol_root: Path,
    freeze_root: Path,
    create: bool,
) -> dict[str, Any]:
    path = protocol_root / "method_evaluation_binding_v1.json"
    current = current_binding_files(root, protocol_root, freeze_root)
    if not path.is_file():
        if not create:
            raise FileNotFoundError(
                f"Current P1/MSR method evaluator is not frozen: {path}. "
                "Run once with --freeze-evaluation-binding before inference."
            )
        binding = {
            "status": "frozen",
            "binding_id": BINDING_ID,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "parent_corruption_protocol": "corruption_protocol_v1",
            "approved_preprocess_protocol": frozen_eval.PREPROCESS_PROTOCOL,
            "method_design_freeze": "MULTISCRIPT_METHOD_DESIGN_FREEZE_V1",
            "test_evaluated": False,
            "files": current,
        }
        path.write_text(
            json.dumps(binding, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return binding
    binding = read_json(path)
    if binding.get("status") != "frozen" or binding.get("binding_id") != BINDING_ID:
        raise ValueError("Invalid method corruption evaluation binding")
    if binding.get("test_evaluated") is not False:
        raise ValueError("Method corruption evaluation binding violated test policy")
    if binding.get("approved_preprocess_protocol") != frozen_eval.PREPROCESS_PROTOCOL:
        raise ValueError("Method corruption preprocessing protocol drift")
    if binding.get("files") != current:
        mismatches = {
            key: {"expected": binding.get("files", {}).get(key), "actual": value}
            for key, value in current.items()
            if binding.get("files", {}).get(key) != value
        }
        raise ValueError(
            "Method corruption evaluation binding hash mismatch: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    return binding


def verify_corruption_freeze(
    root: Path,
    protocol_root: Path,
    freeze_root: Path,
    create_binding: bool,
) -> dict[str, Any]:
    frozen_path = protocol_root / "corruption_protocol_v1.json"
    freeze_report_path = protocol_root / "freeze_report.json"
    hashes_path = protocol_root / "frozen_hashes.json"
    for path in (frozen_path, freeze_report_path, hashes_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    frozen = read_json(frozen_path)
    report = read_json(freeze_report_path)
    if frozen.get("status") != "frozen" or report.get("status") != "passed":
        raise ValueError("Corruption Protocol V1 freeze did not pass")
    if report.get("test_inference_run") is not False:
        raise ValueError("Frozen corruption protocol test policy was violated")
    frozen_hashes = read_json(hashes_path).get("files", {})
    # The corruption transformations and manifests remain immutable. The old
    # evaluator/P1 hashes are intentionally excluded because Method Design V1
    # uses a newer P1_MSR_V3 implementation; that exact implementation is bound
    # separately below instead of mutating the parent corruption protocol.
    frozen_paths = {
        "corruption_protocol_v1.json": protocol_root / "corruption_protocol_v1.json",
        "dev_manifest.jsonl": protocol_root / "dev_manifest.jsonl",
        "test_manifest.jsonl": protocol_root / "test_manifest.jsonl",
        "dev_build_summary.json": protocol_root / "dev_build_summary.json",
        "test_build_summary.json": protocol_root / "test_build_summary.json",
        "dev_calibration_review.html": protocol_root / "dev_calibration_review.html",
        "pilot_review_csv": resolve_recorded_path(
            root,
            frozen["calibration"]["pilot_review_csv"],
        ),
        "straug12_deterministic.py": SCRIPT_DIR / "straug12_deterministic.py",
        "build_corruption_protocol.py": SCRIPT_DIR / "build_corruption_protocol.py",
        "freeze_corruption_protocol.py": SCRIPT_DIR / "freeze_corruption_protocol.py",
        "metrics_v1.py": root / "scripts" / "svtrv2" / "metrics_v1.py",
        "protocol_v2_manifest.json": resolve_recorded_path(
            root,
            frozen["data_protocol_dependency"]["manifest"],
        ),
        "target_metadata.csv": resolve_recorded_path(root, frozen["target_metadata"]),
    }
    for key in IMMUTABLE_CORRUPTION_KEYS:
        path = frozen_paths[key]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = frozen_eval.file_sha256(path)
        expected = frozen_hashes.get(key)
        if actual != expected:
            raise ValueError(
                f"Immutable corruption artifact drift: {key}: "
                f"expected={expected}, actual={actual}"
            )
    protocol_verification = read_json(
        resolve_recorded_path(
            root,
            frozen["data_protocol_dependency"]["verification"],
        )
    )
    if (
        protocol_verification.get("status") != "passed"
        or protocol_verification.get("errors") != []
    ):
        raise ValueError("Current Protocol V2 verification did not pass")
    return verify_or_freeze_binding(
        root,
        protocol_root,
        freeze_root,
        create_binding,
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_root = (
        args.protocol_root.resolve()
        if args.protocol_root
        else root / "05_evaluation" / "corruption_protocol_v1"
    )
    freeze_root = (
        args.method_freeze_root.resolve()
        if args.method_freeze_root
        else root / "00_docs" / "frozen_method_design_v1"
    )
    if args.batch_size != frozen_eval.EVAL_INFERENCE_BATCH_SIZE:
        raise ValueError(
            "Frozen P1/MSR evaluation requires batch size "
            f"{frozen_eval.EVAL_INFERENCE_BATCH_SIZE}"
        )

    binding = verify_corruption_freeze(
        root,
        protocol_root,
        freeze_root,
        args.freeze_evaluation_binding,
    )
    specs = load_method_specs(root, freeze_root)
    frozen_eval.MODEL_SPECS.update(specs)
    if args.verify_only:
        print(
            json.dumps(
                {
                    "status": "FROZEN_METHOD_CORRUPTION_PREFLIGHT_OK",
                    "models": list(METHOD_KEYS),
                    "evaluation_binding": binding["binding_id"],
                    "test_evaluated": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not args.models:
        raise ValueError("--models is required unless --verify-only is used")

    summaries = []
    for model in args.models:
        print(f"===== frozen Corrupted Dev: {model} =====", flush=True)
        original_condition = frozen_eval.evaluate_condition
        original_reuse = frozen_eval.reuse_completed_condition
        replace_model = args.replace
        if model == "b2" and not replace_model:
            existing_path = protocol_root / "eval_reports" / f"{OUTPUT_PREFIX}b2" / "model_summary.json"
            if existing_path.is_file():
                existing = read_json(existing_path)
                if existing.get("clean", {}).get("uyghur_ctc_order") != "logical":
                    print("Discarding stale visual-order B2 corruption report", flush=True)
                    replace_model = True
        try:
            if specs[model]["uyghur_ctc_order"] == "logical":
                frozen_eval.evaluate_condition = logical_ug_condition_adapter(
                    original_condition
                )
                frozen_eval.reuse_completed_condition = logical_ug_reuse_adapter(
                    original_reuse
                )
            internal_key = f"{OUTPUT_PREFIX}{model}"
            frozen_eval.MODEL_SPECS[internal_key] = specs[model]
            summaries.append(
                frozen_eval.evaluate_model(
                    root,
                    protocol_root,
                    internal_key,
                    args.device_id,
                    args.batch_size,
                    replace_model,
                )
            )
        finally:
            frozen_eval.evaluate_condition = original_condition
            frozen_eval.reuse_completed_condition = original_reuse
    invocation = {
        "status": "passed",
        "models": args.models,
        "evaluation_binding": binding["binding_id"],
        "device_id": args.device_id,
        "test_evaluated": False,
        "summaries": [
            str(protocol_root / "eval_reports" / f"{OUTPUT_PREFIX}{key}" / "model_summary.json")
            for key in args.models
        ],
    }
    path = protocol_root / "eval_reports" / (
        "frozen_method_invocation_" + "_".join(args.models) + ".json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(invocation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(invocation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
