#!/usr/bin/env python3
"""Run the exploratory target-only M3+SLDR pilot on frozen Clean Dev V4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


METHOD = "m3_sldr_pilot"
TARGET_NAME = "svtrv2_s_m3_sldr_pilot_dual_order_s50_to_target"
PROTOCOL_ID = "M3_SLDR_TARGET_ONLY_PILOT_CLEAN_DEV_V4_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--m3-s50-checkpoint", type=Path, required=True)
    parser.add_argument("--m3-profile", type=Path, required=True)
    parser.add_argument("--v4-decision-gate", type=Path, required=True)
    parser.add_argument("--v4-reselection-dir", type=Path, required=True)
    parser.add_argument("--v4-protocol-dir", type=Path, required=True)
    parser.add_argument(
        "--v4-frozen-manifest",
        default="frozen_clean_dev_v4_manifest.json",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run(command: list[str], root: Path, log: Path | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    if log is None:
        subprocess.run(command, cwd=str(root), check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f"Command failed ({code}); see {log}")


def prepare(root: Path, source: Path, args: argparse.Namespace) -> Path:
    run(
        [
            sys.executable,
            str(root / "scripts/svtrv2/prepare_dual_order_method.py"),
            "--root", str(root),
            "--method", METHOD,
            "--stage", "target",
            "--source-checkpoint", str(source),
            "--source-role", "frozen_M3_alpha015_S50_pretrained_checkpoint",
            "--max-epoch", "50",
            "--batch-size-per-card", str(args.batch_size),
            "--control-world-size", "2",
            "--control-first-batch-size", "16",
            "--num-workers", str(args.num_workers),
            "--lr", "2.5e-5",
            "--seed", "20260731",
        ],
        root,
        args.log_dir / "prepare.log",
    )
    return root / "04_model_training/configs" / f"{TARGET_NAME}.yml"


def preflight(root: Path, config: Path, args: argparse.Namespace) -> dict:
    run(
        [
            sys.executable,
            str(root / "scripts/svtrv2/validate_dual_order_method_config.py"),
            "--config", str(config),
            "--expected-initialization", "checkpoint",
        ],
        root,
        args.log_dir / "config_validation.log",
    )
    sampler_audit = args.log_dir / "sampler_audit.json"
    run(
        [
            sys.executable,
            str(root / "scripts/svtrv2/audit_p1_ratio_sampler_ddp.py"),
            "--root", str(root),
            "--config", str(config),
            "--world-size", "1",
            "--output", str(sampler_audit),
        ],
        root,
        args.log_dir / "sampler_audit.log",
    )
    smoke = args.log_dir / "single_batch_smoke.json"
    run(
        [
            sys.executable,
            str(root / "scripts/svtrv2/smoke_dual_order_method.py"),
            "--root", str(root),
            "--config", str(config),
            "--device-id", "0",
            "--require-full-initialization",
            "--allow-new-sldr-parameters",
            "--output", str(smoke),
        ],
        root,
        args.log_dir / "single_batch_smoke.log",
    )
    payload = load_json(smoke)
    if payload.get("status") != "DUAL_ORDER_REAL_BATCH_FORWARD_BACKWARD_OK":
        raise RuntimeError(f"Failed SLDR smoke report: {smoke}")
    fresh = payload["initialization_coverage"].get("fresh_parameter_keys", [])
    if not fresh or any(not key.startswith("decoder.sldr.") for key in fresh):
        raise ValueError(f"Invalid fresh-parameter audit: {fresh[:20]}")
    if payload["gradient_groups"].get("sldr", 0) <= 0:
        raise ValueError("No gradient reached SLDR in the real-batch smoke test")
    return {
        "config": str(config),
        "config_sha256": sha256(config),
        "sampler_audit": str(sampler_audit),
        "sampler_audit_sha256": sha256(sampler_audit),
        "smoke": str(smoke),
        "smoke_sha256": sha256(smoke),
        "fresh_sldr_parameter_keys": fresh,
    }


def train(root: Path, config: Path, args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        str(root / "scripts/svtrv2/run_p1_rctc_stage.py"),
        "--root", str(root),
        "--stage", TARGET_NAME,
        "--config", str(config),
        "--run-dir", str(root / "04_model_training/runs" / TARGET_NAME),
        "--labels-dir", str(root / "04_model_training/datasets/e1_target_only/labels"),
        "--label-prefix", "target",
        "--eval-prefix", TARGET_NAME,
        "--expected-initialization", "checkpoint",
        "--config-kind", "dual_order",
        "--prediction-branch", "ctc",
        "--max-epoch", "50",
        "--eval-every", "2",
        "--patience-evals", "5",
        "--min-epoch", "10",
        "--min-delta", "0.0001",
        "--ddp-gpus", str(args.gpu),
        "--eval-gpu", str(args.gpu),
        "--master-port", "29941",
        "--log-dir", str(args.log_dir),
        "--selection-protocol-dir", str(args.v4_protocol_dir),
        "--selection-frozen-manifest", args.v4_frozen_manifest,
        "--selection-protocol-label", "Clean Dev V4",
    ]
    if args.replace:
        command.append("--replace")
    elif args.resume_existing:
        command.append("--resume-existing")
    run(command, root, args.log_dir / "formal.log")
    summary = (
        root / "04_model_training/eval_reports" / f"{TARGET_NAME}_final_summary.json"
    )
    if not summary.is_file():
        raise FileNotFoundError(summary)
    return load_json(summary)


def summarize(root: Path, profile_path: Path, pilot: dict, binding: dict) -> dict:
    profile = load_json(profile_path)
    baseline = {
        **profile["metrics"]["macro"],
        "languages": profile["metrics"]["languages"],
    }
    candidate = pilot["dev"]
    delta_macro = float(candidate["macro_cer"]) - float(baseline["cer"])
    rows = []
    for name, metrics in (("M3_alpha015", baseline), ("M3_plus_SLDR_pilot", candidate)):
        macro_cer = metrics.get("macro_cer", metrics.get("cer"))
        macro_line = metrics.get(
            "macro_line_accuracy", metrics.get("line_accuracy")
        )
        row = {
            "model": name,
            "macro_cer": macro_cer,
            "macro_line_accuracy": macro_line,
        }
        for language in ("zh", "ug", "kk"):
            row[f"{language}_cer"] = metrics["languages"][language]["cer"]
            row[f"{language}_line_accuracy"] = metrics["languages"][language][
                "line_accuracy"
            ]
        rows.append(row)

    deltas = {
        key: float(rows[1][key]) - float(rows[0][key])
        for key in rows[0]
        if key != "model"
    }
    primary_pass = delta_macro < 0
    target_language_pass = deltas["zh_cer"] < 0 or deltas["ug_cer"] < 0
    cer_safety = all(deltas[f"{language}_cer"] <= 0.001 for language in ("zh", "ug", "kk"))
    line_safety = all(
        deltas[f"{language}_line_accuracy"] >= -0.005
        for language in ("zh", "ug", "kk")
    )
    improvement_pp = -delta_macro * 100.0
    if not primary_pass or not cer_safety or not line_safety:
        decision = "PILOT_REJECTED"
    elif improvement_pp < 0.02:
        decision = "WEAK_POSITIVE_REQUIRES_MECHANISM_REVIEW"
    elif target_language_pass:
        decision = "PROMISING_REQUIRES_ERROR_REVIEW_AND_FULL_S50_RETRAIN"
    else:
        decision = "NUMERIC_GAIN_WITHOUT_TARGET_MECHANISM_SUPPORT"

    output_dir = root / "05_evaluation/m3_sldr_target_only_pilot_clean_dev_v4"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "M3_SLDR_TARGET_ONLY_PILOT_COMPLETE",
        "protocol_id": PROTOCOL_ID,
        "experiment_class": "exploratory_pilot_not_formal_main_table",
        "selection_metric": "Clean Dev V4 Macro CER",
        "models": rows,
        "candidate_minus_m3": deltas,
        "macro_cer_improvement_percentage_points": improvement_pp,
        "guardrails": {
            "primary_macro_cer_improved": primary_pass,
            "zh_or_ug_cer_improved": target_language_pass,
            "no_language_cer_worse_by_more_than_0.10pp": cer_safety,
            "no_language_line_accuracy_worse_by_more_than_0.5pp": line_safety,
            "local_glyph_error_reduction": "requires_blind_manual_review_if_numeric_gate_passes",
        },
        "decision": decision,
        "next_action": (
            "build_blind_SLDR_error_review_then_run_full_S50_to_target"
            if decision == "PROMISING_REQUIRES_ERROR_REVIEW_AND_FULL_S50_RETRAIN"
            else "do_not_add_SLDR_to_final_recipe_without_further_authorization"
        ),
        "binding": binding,
        "corrupted_dev_used": False,
        "test_evaluated": False,
    }
    (output_dir / "pilot_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    args = parse_args()
    if args.replace and args.resume_existing:
        raise ValueError("--replace and --resume-existing cannot be combined")
    root = args.root.resolve()
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.m3_s50_checkpoint = args.m3_s50_checkpoint.resolve()
    args.m3_profile = args.m3_profile.resolve()
    args.v4_decision_gate = args.v4_decision_gate.resolve()
    args.v4_reselection_dir = args.v4_reselection_dir.resolve()
    args.v4_protocol_dir = args.v4_protocol_dir.resolve()
    required = (
        args.m3_s50_checkpoint,
        args.m3_profile,
        args.v4_decision_gate,
        args.v4_reselection_dir / "final_decisions.json",
        args.v4_protocol_dir / args.v4_frozen_manifest,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    profile = load_json(args.m3_profile)
    gate = load_json(args.v4_decision_gate)
    decisions_path = args.v4_reselection_dir / "final_decisions.json"
    decisions = load_json(decisions_path)
    if profile.get("status") != "CLEAN_DEV_V4_M3_ERROR_PROFILE_READY":
        raise ValueError("Pilot requires the frozen Clean Dev V4 M3 profile")
    if profile.get("alpha") != 0.15 or profile.get("selected_epoch") != 34:
        raise ValueError("Pilot requires frozen M3 alpha=.15 at epoch 34")
    if profile.get("test_evaluated") is not False:
        raise ValueError("M3 profile violates Test isolation")
    if gate.get("status") != "MANUAL_REVIEW_COMPLETE":
        raise ValueError("Clean Dev V4 manual error review is incomplete")
    if gate.get("test_evaluated") is not False:
        raise ValueError("Clean Dev V4 gate violates Test isolation")
    alpha = decisions.get("alpha_selection", {})
    scale = decisions.get("scale_selection", {})
    if alpha.get("selected_alpha") != 0.15 or alpha.get("selected_model") != "M3_alpha015":
        raise ValueError("Clean Dev V4 does not freeze M3 alpha=.15")
    if scale.get("selected_scale") != "S50":
        raise ValueError("Clean Dev V4 does not freeze S50")

    original_target_prepare_path = (
        root / "04_model_training/datasets/m3_dual_order_s50_to_target/prepare_summary.json"
    )
    if not original_target_prepare_path.is_file():
        raise FileNotFoundError(original_target_prepare_path)
    original_target_prepare = load_json(original_target_prepare_path)
    expected_source_hash = original_target_prepare.get("source_checkpoint_sha256")
    actual_source_hash = sha256(args.m3_s50_checkpoint)
    if actual_source_hash != expected_source_hash:
        raise ValueError(
            "SLDR pilot initialization differs from frozen M3 target initialization: "
            f"expected={expected_source_hash}, actual={actual_source_hash}"
        )

    sldr_gate = gate.get("sldr", {})
    binding = {
        "protocol_id": PROTOCOL_ID,
        "classification": "exploratory_target_only_pilot",
        "reason": (
            "Clean Dev V4 local/joining evidence was below the predeclared "
            "formal threshold; this pilot cannot enter the formal main table"
        ),
        "m3_s50_checkpoint": str(args.m3_s50_checkpoint),
        "m3_s50_checkpoint_sha256": actual_source_hash,
        "frozen_m3_target_prepare": str(original_target_prepare_path),
        "frozen_m3_target_prepare_sha256": sha256(original_target_prepare_path),
        "m3_profile": str(args.m3_profile),
        "m3_profile_sha256": sha256(args.m3_profile),
        "v4_gate": str(args.v4_decision_gate),
        "v4_gate_sha256": sha256(args.v4_decision_gate),
        "v4_manifest": str(args.v4_protocol_dir / args.v4_frozen_manifest),
        "v4_manifest_sha256": sha256(args.v4_protocol_dir / args.v4_frozen_manifest),
        "v4_final_decisions": str(decisions_path),
        "v4_final_decisions_sha256": sha256(decisions_path),
        "observed_local_or_joining_ratio": sldr_gate.get("zh_ug_local_or_joining_ratio"),
        "predeclared_sldr_authorized": sldr_gate.get("sldr_authorized"),
        "synthetic_scale": "S50 checkpoint reused; no synthetic retraining in pilot",
        "alpha": 0.15,
        "reduction": 4,
        "kernels": ["3x3", "1x5", "5x1"],
        "cross_dimensional_axes": ["channel", "height", "width"],
        "script_routing": "reuse frozen M3 visual script posterior",
        "gamma_init": 0.001,
        "new_losses": [],
        "seed": 20260731,
        "global_batch_size": args.batch_size,
        "only_changed_factor": "insert SLDR before M3 script adaptation and RCTC FRM",
        "test_evaluated": False,
    }
    binding_path = args.log_dir / "study_binding.json"
    if binding_path.is_file():
        if load_json(binding_path) != binding:
            raise ValueError("SLDR pilot binding changed during resume")
    else:
        binding_path.write_text(
            json.dumps(binding, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    config = prepare(root, args.m3_s50_checkpoint, args)
    preflight_report = preflight(root, config, args)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "M3_SLDR_TARGET_ONLY_PILOT_PREFLIGHT_OK",
                    "protocol_id": PROTOCOL_ID,
                    "preflight": preflight_report,
                    "binding": binding,
                    "formal_training_started": False,
                    "test_evaluated": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    pilot = train(root, config, args)
    summarize(root, args.m3_profile, pilot, binding)


if __name__ == "__main__":
    main()
