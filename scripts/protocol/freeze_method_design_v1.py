#!/usr/bin/env python3
"""Freeze or verify the finalized B0/B1/B2/M1/M2/M3/Full method design.

The official freeze must run where every YAML and best checkpoint exists. It
verifies the hashes recorded by each final summary before writing an immutable
manifest. Missing artifacts are errors; summary-only freezes are forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROTOCOL_ID = "MULTISCRIPT_METHOD_DESIGN_FREEZE_V1"
DATA_PROTOCOL_ID = "multilingual_meme_ocr_data_eval_protocol_v2"
PREPROCESS_PROTOCOL = "P1_MSR_V3"
SELECTION_METRIC = "clean_target_dev_macro_CER"
SOURCE_DOCUMENT = "00_docs/METHOD_DESIGN_FREEZE_V1.md"
OUTPUT_RELATIVE = "00_docs/frozen_method_design_v1"


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "b0": {
        "role": "rctc_baseline",
        "method_variant": None,
        "target_model": "svtrv2_s_e5_d2_to_target",
        "target_experiment": "e5",
        "target_summary": "e5_d2_to_target_final_summary.json",
        "synthetic_model": "svtrv2_s_d2_synth50k",
        "synthetic_experiment": "d2",
        "synthetic_summary": "d2_synth50k_final_summary.json",
        "expected_macro_cer": 0.028508178704055528,
        "expected_best_epoch": 8,
    },
    "b1": {
        "role": "full_svtrv2_baseline",
        "method_variant": None,
        "target_model": "svtrv2_s_b1_full_s50_to_target",
        "target_experiment": "b1_full_s50_to_target",
        "target_summary": "b1_full_s50_to_target_final_summary.json",
        "synthetic_model": "svtrv2_s_b1_full_s50",
        "synthetic_experiment": "b1_full_s50",
        "synthetic_summary": "b1_full_s50_final_summary.json",
        "expected_macro_cer": 0.025132151375509035,
        "expected_best_epoch": 20,
    },
    "b2": {
        "role": "direction_conflict_diagnostic",
        "method_variant": "b2",
        "target_model": "svtrv2_s_b2_dual_order_s50_to_target",
        "target_summary": "svtrv2_s_b2_dual_order_s50_to_target_final_summary.json",
        "synthetic_model": "svtrv2_s_b2_dual_order_s50",
        "synthetic_summary": "svtrv2_s_b2_dual_order_s50_final_summary.json",
        "expected_macro_cer": 0.03164987301635231,
        "expected_best_epoch": 42,
        "expected": {
            "ctc_order": "logical",
            "sgm_order": "logical",
            "consistency_weight": 0.0,
            "script_weight": 0.0,
            "direction_weight": 0.0,
            "use_script_adapter": False,
            "use_local_direction": False,
        },
    },
    "m1": {
        "role": "dual_order_semantic_guidance",
        "method_variant": "m1",
        "target_model": "svtrv2_s_m1_dual_order_s50_to_target",
        "target_summary": "svtrv2_s_m1_dual_order_s50_to_target_final_summary.json",
        "synthetic_model": "svtrv2_s_m1_dual_order_s50",
        "synthetic_summary": "svtrv2_s_m1_dual_order_s50_final_summary.json",
        "expected_macro_cer": 0.024245439406437336,
        "expected_best_epoch": 28,
        "expected": {
            "ctc_order": "visual",
            "sgm_order": "logical",
            "consistency_weight": 0.0,
            "script_weight": 0.0,
            "direction_weight": 0.0,
            "use_script_adapter": False,
            "use_local_direction": False,
        },
    },
    "m2": {
        "role": "official_consistency_ablation_alpha_030",
        "method_variant": "m2",
        "target_model": "svtrv2_s_m2_alpha_030_s50_to_target",
        "target_summary": "svtrv2_s_m2_alpha_030_s50_to_target_final_summary.json",
        "synthetic_model": "svtrv2_s_m2_alpha_030_s50",
        "synthetic_summary": "svtrv2_s_m2_alpha_030_s50_final_summary.json",
        "expected_macro_cer": 0.02398565666342996,
        "expected_best_epoch": 32,
        "expected": {
            "ctc_order": "visual",
            "sgm_order": "logical",
            "consistency_weight": 0.30,
            "script_weight": 0.0,
            "direction_weight": 0.0,
            "use_script_adapter": False,
            "use_local_direction": False,
        },
    },
    "m2_control_alpha015": {
        "role": "matched_control_for_script_adapter",
        "method_variant": "m2",
        "target_model": "svtrv2_s_m2_dual_order_s50_to_target",
        "target_summary": "svtrv2_s_m2_dual_order_s50_to_target_final_summary.json",
        "synthetic_model": "svtrv2_s_m2_dual_order_s50",
        "synthetic_summary": "svtrv2_s_m2_dual_order_s50_final_summary.json",
        "expected_macro_cer": 0.024441653015459145,
        "expected_best_epoch": 20,
        "expected": {
            "ctc_order": "visual",
            "sgm_order": "logical",
            "consistency_weight": 0.15,
            "script_weight": 0.0,
            "direction_weight": 0.0,
            "use_script_adapter": False,
            "use_local_direction": False,
        },
    },
    "m3": {
        "role": "primary_proposed_model",
        "method_variant": "m3",
        "target_model": "svtrv2_s_m3_dual_order_s50_to_target",
        "target_summary": "svtrv2_s_m3_dual_order_s50_to_target_final_summary.json",
        "synthetic_model": "svtrv2_s_m3_dual_order_s50",
        "synthetic_summary": "svtrv2_s_m3_dual_order_s50_final_summary.json",
        "expected_macro_cer": 0.022986871482536465,
        "expected_best_epoch": 34,
        "expected": {
            "ctc_order": "visual",
            "sgm_order": "logical",
            "consistency_weight": 0.15,
            "script_weight": 0.10,
            "direction_weight": 0.0,
            "use_script_adapter": True,
            "use_local_direction": False,
        },
    },
    "full": {
        "role": "negative_local_direction_ablation",
        "method_variant": "full",
        "target_model": "svtrv2_s_full_dual_order_s50_to_target",
        "target_summary": "svtrv2_s_full_dual_order_s50_to_target_final_summary.json",
        "synthetic_model": "svtrv2_s_full_dual_order_s50",
        "synthetic_summary": "svtrv2_s_full_dual_order_s50_final_summary.json",
        "expected_macro_cer": 0.023591340040362712,
        "expected_best_epoch": 24,
        "expected": {
            "ctc_order": "visual",
            "sgm_order": "logical",
            "consistency_weight": 0.15,
            "script_weight": 0.10,
            "direction_weight": 0.05,
            "use_script_adapter": True,
            "use_local_direction": True,
        },
    },
}


IMPLEMENTATION_FILES = (
    SOURCE_DOCUMENT,
    "00_docs/MULTISCRIPT_DUAL_ORDER_SGM_V1.md",
    "scripts/protocol/freeze_method_design_v1.py",
    "scripts/svtrv2/dual_order_protocol.py",
    "scripts/svtrv2/evaluate_d2_checkpoint.py",
    "scripts/svtrv2/infer_label_file_metrics.py",
    "scripts/svtrv2/p1_msr_protocol.py",
    "third_party/OpenOCR/openrec/losses/dual_order_gtc_loss.py",
    "third_party/OpenOCR/openrec/modeling/decoders/dual_order_gtc_decoder.py",
    "third_party/OpenOCR/openrec/preprocess/dual_order_gtc_label_encode.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("freeze", "verify"), default="freeze")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def require_close(actual: Any, expected: float, label: str) -> None:
    if actual is None or abs(float(actual) - expected) > 1e-12:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def transform_config(cfg: dict[str, Any], section: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in cfg[section]["dataset"].get("transforms") or []:
        if isinstance(item, dict) and len(item) == 1:
            name, value = next(iter(item.items()))
            result[name] = value or {}
    return result


def validate_dual_config(
    cfg: dict[str, Any],
    experiment_id: str,
    spec: dict[str, Any],
) -> None:
    expected = spec["expected"]
    global_cfg = cfg["Global"]
    decoder = cfg["Architecture"]["Decoder"]
    loss = cfg["Loss"]
    require_equal(
        global_cfg.get("method_variant"),
        spec["method_variant"],
        f"{experiment_id}.Global.method_variant",
    )
    require_equal(
        global_cfg.get("uyghur_ctc_order"),
        expected["ctc_order"],
        f"{experiment_id}.Global.uyghur_ctc_order",
    )
    require_equal(
        decoder.get("name"),
        "DualOrderGTCDecoder",
        f"{experiment_id}.Decoder.name",
    )
    require_equal(
        decoder.get("use_script_adapter"),
        expected["use_script_adapter"],
        f"{experiment_id}.Decoder.use_script_adapter",
    )
    require_equal(
        decoder.get("use_local_direction"),
        expected["use_local_direction"],
        f"{experiment_id}.Decoder.use_local_direction",
    )
    for key in ("consistency_weight", "script_weight", "direction_weight"):
        require_close(loss.get(key), expected[key], f"{experiment_id}.Loss.{key}")
    for section in ("Train", "Eval"):
        transforms = transform_config(cfg, section)
        label = transforms.get("DualOrderGTCLabelEncode")
        if not isinstance(label, dict):
            raise ValueError(f"{experiment_id}.{section} lacks dual-order labels")
        require_equal(
            label.get("ctc_order"),
            expected["ctc_order"],
            f"{experiment_id}.{section}.ctc_order",
        )
        require_equal(
            label.get("sgm_order"),
            expected["sgm_order"],
            f"{experiment_id}.{section}.sgm_order",
        )


def validate_stage(
    root: Path,
    experiment_id: str,
    spec: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    model = spec[f"{stage}_model"]
    summary_rel = f"04_model_training/eval_reports/{spec[f'{stage}_summary']}"
    config_rel = f"04_model_training/configs/{model}.yml"
    checkpoint_rel = (
        f"04_model_training/runs/{model}/best_clean_dev_macro_cer.pth"
    )
    best_json_rel = (
        f"04_model_training/runs/{model}/best_clean_dev_macro_cer.json"
    )
    paths = {
        "summary": root / summary_rel,
        "config": root / config_rel,
        "checkpoint": root / checkpoint_rel,
        "best_json": root / best_json_rel,
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen artifacts:\n" + "\n".join(missing))

    summary = read_json(paths["summary"])
    best = read_json(paths["best_json"])
    cfg = yaml.safe_load(paths["config"].read_text(encoding="utf-8-sig"))
    require_equal(
        summary.get("experiment"),
        spec.get(f"{stage}_experiment", model),
        f"{experiment_id}.{stage}.experiment",
    )
    require_equal(
        summary.get("preprocess_protocol"),
        PREPROCESS_PROTOCOL,
        f"{experiment_id}.{stage}.preprocess_protocol",
    )
    require_equal(
        summary.get("data_protocol_id"),
        DATA_PROTOCOL_ID,
        f"{experiment_id}.{stage}.data_protocol_id",
    )
    require_equal(
        summary.get("checkpoint_selection"),
        SELECTION_METRIC,
        f"{experiment_id}.{stage}.checkpoint_selection",
    )
    if not str(summary.get("test_policy") or "").startswith("not_evaluated"):
        raise ValueError(f"{experiment_id}.{stage} violated the test policy")

    config_hash = sha256_file(paths["config"])
    checkpoint_hash = sha256_file(paths["checkpoint"])
    require_equal(
        summary.get("config_sha256"),
        config_hash,
        f"{experiment_id}.{stage}.config_sha256",
    )
    require_equal(
        summary.get("best_checkpoint_sha256"),
        checkpoint_hash,
        f"{experiment_id}.{stage}.best_checkpoint_sha256",
    )
    require_equal(
        best.get("copied_checkpoint_sha256"),
        checkpoint_hash,
        f"{experiment_id}.{stage}.best_json checkpoint hash",
    )
    require_equal(
        best.get("selection_metric"),
        SELECTION_METRIC,
        f"{experiment_id}.{stage}.best_json selection metric",
    )
    require_equal(
        cfg["Global"].get("preprocess_protocol"),
        PREPROCESS_PROTOCOL,
        f"{experiment_id}.{stage}.config preprocess protocol",
    )
    if spec.get("expected"):
        validate_dual_config(cfg, experiment_id, spec)

    if stage == "target":
        require_equal(
            summary.get("best_epoch"),
            spec["expected_best_epoch"],
            f"{experiment_id}.best_epoch",
        )
        require_close(
            (summary.get("dev") or {}).get("macro_cer"),
            spec["expected_macro_cer"],
            f"{experiment_id}.dev.macro_cer",
        )

    return {
        "model": model,
        "stage": stage,
        "summary": summary_rel,
        "summary_sha256": sha256_file(paths["summary"]),
        "config": config_rel,
        "config_sha256": config_hash,
        "checkpoint": checkpoint_rel,
        "checkpoint_sha256": checkpoint_hash,
        "best_json": best_json_rel,
        "best_json_sha256": sha256_file(paths["best_json"]),
        "best_epoch": summary.get("best_epoch"),
        "clean_dev_macro_cer": (summary.get("dev") or {}).get("macro_cer"),
        "test_policy": summary.get("test_policy"),
    }


def build_manifest(root: Path) -> dict[str, Any]:
    protocol_v2 = root / "00_docs/frozen_protocol_v2/protocol_v2_manifest.json"
    if not protocol_v2.is_file():
        raise FileNotFoundError(protocol_v2)
    methods: dict[str, Any] = {}
    for experiment_id, spec in EXPERIMENTS.items():
        methods[experiment_id] = {
            "role": spec["role"],
            "method_variant": spec["method_variant"],
            "synthetic": validate_stage(root, experiment_id, spec, "synthetic"),
            "target": validate_stage(root, experiment_id, spec, "target"),
        }
        if spec.get("expected"):
            methods[experiment_id]["frozen_method_settings"] = spec["expected"]

    implementation = {}
    for relative in IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        implementation[relative] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    manifest: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "primary_model": "m3",
        "official_m2_consistency_weight": 0.30,
        "primary_m3_consistency_weight": 0.15,
        "negative_ablation": "full",
        "selection_metric": SELECTION_METRIC,
        "test_policy": "not_evaluated_during_model_development",
        "data_protocol": {
            "path": "00_docs/frozen_protocol_v2/protocol_v2_manifest.json",
            "sha256": sha256_file(protocol_v2),
            "verification_fingerprint_sha256": read_json(protocol_v2).get(
                "verification_fingerprint_sha256"
            ),
        },
        "development_closed": True,
        "future_change_policy": (
            "Any architecture, loss-weight, preprocessing, data, or selection "
            "change requires a new protocol ID."
        ),
        "methods": methods,
        "implementation": implementation,
    }
    manifest["manifest_fingerprint_sha256"] = canonical_hash(manifest)
    return manifest


def verify_manifest(root: Path, frozen: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_fingerprint = frozen.get("manifest_fingerprint_sha256")
    unsigned = dict(frozen)
    unsigned.pop("manifest_fingerprint_sha256", None)
    actual_fingerprint = canonical_hash(unsigned)
    if actual_fingerprint != expected_fingerprint:
        errors.append(
            "manifest fingerprint mismatch: "
            f"expected={expected_fingerprint}, actual={actual_fingerprint}"
        )

    for method_id, method in (frozen.get("methods") or {}).items():
        for stage in ("synthetic", "target"):
            artifact = method[stage]
            for key in ("summary", "config", "checkpoint", "best_json"):
                relative = artifact[key]
                path = root / relative
                if not path.is_file():
                    errors.append(f"missing {method_id}/{stage}/{key}: {relative}")
                    continue
                actual = sha256_file(path)
                expected = artifact[f"{key}_sha256"]
                if actual != expected:
                    errors.append(
                        f"hash mismatch {method_id}/{stage}/{key}: "
                        f"expected={expected}, actual={actual}"
                    )

    for relative, record in (frozen.get("implementation") or {}).items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing implementation file: {relative}")
            continue
        actual = sha256_file(path)
        if actual != record.get("sha256"):
            errors.append(
                f"implementation hash mismatch {relative}: "
                f"expected={record.get('sha256')}, actual={actual}"
            )
    data = frozen.get("data_protocol") or {}
    data_path = root / str(data.get("path") or "")
    if not data_path.is_file():
        errors.append(f"missing data protocol: {data_path}")
    elif sha256_file(data_path) != data.get("sha256"):
        errors.append("Protocol V2 manifest hash mismatch")
    return errors


def write_verification(output_dir: Path, errors: list[str]) -> dict[str, Any]:
    report = {
        "protocol_id": PROTOCOL_ID,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "development_closed": not errors,
    }
    write_json(output_dir / "verification_report.json", report)
    return report


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root / OUTPUT_RELATIVE).resolve()
    manifest_path = output_dir / "method_design_v1_manifest.json"

    if args.mode == "freeze":
        if output_dir.exists() and not args.replace:
            raise FileExistsError(
                f"Already frozen: {output_dir}. Use --replace only before the "
                "official freeze is declared."
            )
        manifest = build_manifest(root)
        parent = output_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            staging = Path(temporary) / output_dir.name
            staging.mkdir()
            write_json(staging / "method_design_v1_manifest.json", manifest)
            shutil.copy2(root / SOURCE_DOCUMENT, staging / "METHOD_DESIGN_V1.md")
            report = write_verification(staging, verify_manifest(root, manifest))
            if report["status"] != "passed":
                raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.move(str(staging), str(output_dir))
    else:
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = read_json(manifest_path)
        require_equal(manifest.get("protocol_id"), PROTOCOL_ID, "protocol_id")
        report = write_verification(
            output_dir,
            verify_manifest(root, manifest),
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)
    print("METHOD_DESIGN_FREEZE_V1_OK")


if __name__ == "__main__":
    main()
