#!/usr/bin/env python3
"""Run one frozen full-S50 M3 refinement track through target fine-tuning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


TRACKS = {
    "sldr": {
        "method": "m3_sldr_full_v1",
        "label": "M3_plus_SLDR_FULL_S50_V1",
        "protocol": "M3_SLDR_FULL_S50_CLEAN_DEV_V4_V1",
        "only_changed_factor": (
            "SLDR V1 is active throughout S50 pretraining and target fine-tuning"
        ),
        "master_ports": (29951, 29952),
    },
    "scdl": {
        "method": "m3_scdl_full_v1",
        "label": "M3_plus_SCDL_FULL_S50_V1",
        "protocol": "M3_SCDL_FULL_S50_CLEAN_DEV_V4_V1",
        "only_changed_factor": (
            "training-only SCDL is active after 20% warmup in both S50 and target"
        ),
        "master_ports": (29953, 29954),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=tuple(TRACKS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--rctc-init", type=Path, required=True)
    parser.add_argument("--m3-profile", type=Path, required=True)
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
    parser.add_argument("--rebuild-dual-order-lmdb", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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


@contextmanager
def preparation_lock(root: Path):
    """Serialize only shared LMDB/config preparation across parallel tracks."""

    lock_path = root / "04_model_training/logs/m3_full_refinement_prepare.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def names(method: str) -> tuple[str, str]:
    base = f"svtrv2_s_{method}_dual_order_s50"
    return base, f"{base}_to_target"


def prepare(
    root: Path,
    method: str,
    stage: str,
    source: Path,
    args: argparse.Namespace,
) -> Path:
    synthetic_name, target_name = names(method)
    command = [
        sys.executable,
        str(root / "scripts/svtrv2/prepare_dual_order_method.py"),
        "--root", str(root),
        "--method", method,
        "--stage", stage,
        "--source-checkpoint", str(source),
        "--source-role", (
            "same_frozen_RCTC_initialization_as_M3"
            if stage == "synthetic"
            else (
                "SCDL_S50_CER_selected_checkpoint_with_target_prototypes_reset"
                if args.track == "scdl"
                else "SLDR_S50_CER_selected_checkpoint"
            )
        ),
        "--max-epoch", "50",
        "--batch-size-per-card", str(args.batch_size),
        "--control-world-size", "2",
        "--control-first-batch-size", "16",
        "--num-workers", str(args.num_workers),
        "--lr", "2.5e-5",
        "--seed", "20260731",
    ]
    if args.rebuild_dual_order_lmdb and stage == "synthetic":
        command.append("--replace-dual-order-lmdb")
    with preparation_lock(root):
        run(command, root, args.log_dir / f"{stage}_prepare.log")
    name = synthetic_name if stage == "synthetic" else target_name
    return root / "04_model_training/configs" / f"{name}.yml"


def preflight(
    root: Path,
    config: Path,
    stage: str,
    args: argparse.Namespace,
) -> dict:
    run(
        [
            sys.executable,
            str(root / "scripts/svtrv2/validate_dual_order_method_config.py"),
            "--config", str(config),
            "--expected-initialization", "checkpoint",
        ],
        root,
        args.log_dir / f"{stage}_config_validation.log",
    )
    audit = args.log_dir / f"{stage}_sampler_audit.json"
    run(
        [
            sys.executable,
            str(root / "scripts/svtrv2/audit_p1_ratio_sampler_ddp.py"),
            "--root", str(root),
            "--config", str(config),
            "--world-size", "1",
            "--output", str(audit),
        ],
        root,
        args.log_dir / f"{stage}_sampler_audit.log",
    )
    smoke = args.log_dir / f"{stage}_smoke.json"
    command = [
        sys.executable,
        str(root / "scripts/svtrv2/smoke_dual_order_method.py"),
        "--root", str(root),
        "--config", str(config),
        "--device-id", "0",
        "--output", str(smoke),
    ]
    if stage == "target":
        command.append("--require-full-initialization")
    run(command, root, args.log_dir / f"{stage}_smoke.log")
    payload = load_json(smoke)
    if payload.get("status") != "DUAL_ORDER_REAL_BATCH_FORWARD_BACKWARD_OK":
        raise RuntimeError(f"Failed smoke report: {smoke}")
    expected_group = args.track
    if payload.get("gradient_groups", {}).get(expected_group, 0) <= 0:
        raise ValueError(f"No gradient reached {expected_group.upper()}")
    coverage = payload.get("initialization_coverage", {})
    if stage == "target" and coverage.get("full_model_complete") is not True:
        raise ValueError("Target initialization is not a complete full-model load")
    return {
        "config": str(config),
        "config_sha256": sha256(config),
        "sampler_audit": str(audit),
        "sampler_audit_sha256": sha256(audit),
        "smoke": str(smoke),
        "smoke_sha256": sha256(smoke),
        "gradient_group": expected_group,
    }


def train(
    root: Path,
    method: str,
    config: Path,
    stage: str,
    args: argparse.Namespace,
) -> dict:
    synthetic_name, target_name = names(method)
    name = synthetic_name if stage == "synthetic" else target_name
    port_index = 0 if stage == "synthetic" else 1
    command = [
        sys.executable,
        str(root / "scripts/svtrv2/run_p1_rctc_stage.py"),
        "--root", str(root),
        "--stage", name,
        "--config", str(config),
        "--run-dir", str(root / "04_model_training/runs" / name),
        "--labels-dir", str(root / "04_model_training/datasets/e1_target_only/labels"),
        "--label-prefix", "target",
        "--eval-prefix", name,
        "--expected-initialization", "checkpoint",
        "--config-kind", "dual_order",
        "--prediction-branch", "ctc",
        "--max-epoch", "50",
        "--eval-every", "5" if stage == "synthetic" else "2",
        "--patience-evals", "4" if stage == "synthetic" else "5",
        "--min-epoch", "15" if stage == "synthetic" else "10",
        "--min-delta", "0.0001",
        "--ddp-gpus", str(args.gpu),
        "--eval-gpu", str(args.gpu),
        "--master-port", str(TRACKS[args.track]["master_ports"][port_index]),
        "--log-dir", str(args.log_dir),
    ]
    if stage == "target":
        command.extend(
            [
                "--selection-protocol-dir", str(args.v4_protocol_dir),
                "--selection-frozen-manifest", args.v4_frozen_manifest,
                "--selection-protocol-label", "Clean Dev V4",
            ]
        )
    if args.replace:
        command.append("--replace")
    elif args.resume_existing:
        command.append("--resume-existing")
    run(command, root, args.log_dir / f"{stage}_formal.log")
    summary = root / "04_model_training/eval_reports" / f"{name}_final_summary.json"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    return load_json(summary)


def reset_scdl_prototypes(
    root: Path,
    source: Path,
    replace: bool,
) -> tuple[Path, Path]:
    import torch

    output_dir = root / "04_model_training/checkpoint_adapters/m3_scdl_full_s50_v1"
    output = output_dir / "target_init_prototypes_reset.pth"
    audit_path = output_dir / "target_init_prototypes_reset_audit.json"
    source_hash = sha256(source)
    if output.is_file() and audit_path.is_file() and not replace:
        audit = load_json(audit_path)
        if (
            audit.get("source_checkpoint_sha256") == source_hash
            and audit.get("output_checkpoint_sha256") == sha256(output)
            and audit.get("prototype_reset_verified") is True
        ):
            return output, audit_path
        raise ValueError("Existing SCDL target reset checkpoint failed provenance audit")

    payload = torch.load(source, map_location=torch.device("cpu"))
    state = payload.get("state_dict", payload)
    prototype_key = "decoder.scdl.prototypes"
    count_key = "decoder.scdl.prototype_counts"
    script_key = "decoder.scdl.prototype_script_ids"
    missing = [key for key in (prototype_key, count_key, script_key) if key not in state]
    if missing:
        raise KeyError(f"SCDL checkpoint lacks prototype state: {missing}")
    before_nonzero = {
        "prototype_values": int(torch.count_nonzero(state[prototype_key]).item()),
        "prototype_counts": int(torch.count_nonzero(state[count_key]).item()),
    }
    state[prototype_key] = torch.zeros_like(state[prototype_key])
    state[count_key] = torch.zeros_like(state[count_key])
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".building.pth")
    torch.save(payload, temporary)
    temporary.replace(output)
    restored = torch.load(output, map_location=torch.device("cpu"))["state_dict"]
    verified = (
        int(torch.count_nonzero(restored[prototype_key]).item()) == 0
        and int(torch.count_nonzero(restored[count_key]).item()) == 0
        and torch.equal(restored[script_key], state[script_key])
    )
    if not verified:
        raise ValueError("SCDL target prototype reset verification failed")
    audit = {
        "status": "SCDL_TARGET_PROTOTYPE_RESET_OK",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_hash,
        "output_checkpoint": str(output),
        "output_checkpoint_sha256": sha256(output),
        "reset_keys": [prototype_key, count_key],
        "preserved_key": script_key,
        "before_nonzero": before_nonzero,
        "after_nonzero": {"prototype_values": 0, "prototype_counts": 0},
        "prototype_reset_verified": verified,
        "optimizer_state_used_for_target": False,
        "test_evaluated": False,
    }
    write_json(audit_path, audit)
    return output, audit_path


def macro_metrics(dev: dict) -> dict:
    languages = dev["languages"]
    return {
        "macro_cer": float(dev["macro_cer"]),
        "macro_wer": sum(float(languages[x]["wer"]) for x in ("zh", "ug", "kk")) / 3,
        "macro_one_minus_ned": sum(
            float(languages[x]["one_minus_ned_macro"]) for x in ("zh", "ug", "kk")
        ) / 3,
        "macro_line_accuracy": float(dev["macro_line_accuracy"]),
    }


def summarize(root: Path, args: argparse.Namespace, target: dict, binding: dict) -> dict:
    profile = load_json(args.m3_profile)
    baseline = {
        "macro_cer": float(profile["metrics"]["macro"]["cer"]),
        "macro_wer": float(profile["metrics"]["macro"]["wer"]),
        "macro_one_minus_ned": float(
            profile["metrics"]["macro"]["one_minus_ned_macro"]
        ),
        "macro_line_accuracy": float(profile["metrics"]["macro"]["line_accuracy"]),
        "languages": profile["metrics"]["languages"],
    }
    candidate = {**macro_metrics(target["dev"]), "languages": target["dev"]["languages"]}
    rows = []
    for model, metrics in (("M3_alpha015", baseline), (TRACKS[args.track]["label"], candidate)):
        row = {"model": model, **{key: metrics[key] for key in (
            "macro_cer", "macro_wer", "macro_one_minus_ned", "macro_line_accuracy"
        )}}
        for language in ("zh", "ug", "kk"):
            for metric in ("cer", "wer", "one_minus_ned_macro", "line_accuracy"):
                row[f"{language}_{metric}"] = float(metrics["languages"][language][metric])
        rows.append(row)
    deltas = {
        key: float(rows[1][key]) - float(rows[0][key])
        for key in rows[0]
        if key != "model"
    }
    cer_improvements = sum(deltas[f"{language}_cer"] < 0 for language in ("zh", "ug", "kk"))
    cer_safety = all(deltas[f"{language}_cer"] <= 0.001 for language in ("zh", "ug", "kk"))
    line_safety = all(
        deltas[f"{language}_line_accuracy"] >= -0.005
        for language in ("zh", "ug", "kk")
    )
    improvement_pp = -deltas["macro_cer"] * 100.0
    if improvement_pp >= 0.02 and cer_improvements >= 2 and cer_safety and line_safety:
        decision = "CLEAR_SINGLE_SEED_POSITIVE_PROCEED_TO_THREE_SEEDS"
    elif deltas["macro_cer"] < 0 and cer_safety and line_safety:
        decision = "WEAK_POSITIVE_REQUIRES_SECOND_SEED"
    else:
        decision = "NO_IMPROVEMENT_STOP_TRACK"

    output_dir = root / f"05_evaluation/m3_{args.track}_full_s50_v1_clean_dev_v4"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": f"M3_{args.track.upper()}_FULL_S50_V1_COMPLETE",
        "protocol_id": TRACKS[args.track]["protocol"],
        "selection_metric": "argmin Clean Dev V4 Macro CER",
        "models": rows,
        "candidate_minus_m3": deltas,
        "macro_cer_improvement_percentage_points": improvement_pp,
        "predeclared_thresholds": {
            "clear_positive": "Macro CER improvement >= 0.02pp",
            "weak_positive": "Macro CER improvement in (0, 0.02pp)",
            "language_requirement": "at least two language CER values improve",
            "cer_safety": "no language CER worsens by more than 0.10pp",
            "line_accuracy_safety": "no language Line Accuracy worsens by more than 0.5pp",
        },
        "guardrails": {
            "languages_with_cer_improvement": cer_improvements,
            "cer_safety_pass": cer_safety,
            "line_accuracy_safety_pass": line_safety,
        },
        "decision": decision,
        "binding": binding,
        "corrupted_dev_used": False,
        "test_evaluated": False,
    }
    write_json(output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    args = parse_args()
    if args.replace and args.resume_existing:
        raise ValueError("--replace and --resume-existing cannot be combined")
    if args.batch_size != 32:
        raise ValueError("Frozen full-S50 comparison requires global batch size 32")
    root = args.root.resolve()
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.rctc_init = args.rctc_init.resolve()
    args.m3_profile = args.m3_profile.resolve()
    args.v4_reselection_dir = args.v4_reselection_dir.resolve()
    args.v4_protocol_dir = args.v4_protocol_dir.resolve()
    decisions_path = args.v4_reselection_dir / "final_decisions.json"
    manifest = args.v4_protocol_dir / args.v4_frozen_manifest
    required = (args.rctc_init, args.m3_profile, decisions_path, manifest)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    profile = load_json(args.m3_profile)
    decisions = load_json(decisions_path)
    if profile.get("status") != "CLEAN_DEV_V4_M3_ERROR_PROFILE_READY":
        raise ValueError("Full refinement requires the frozen Clean Dev V4 M3 profile")
    if profile.get("alpha") != 0.15 or profile.get("selected_epoch") != 34:
        raise ValueError("Full refinement requires frozen M3 alpha=.15 at epoch 34")
    if profile.get("test_evaluated") is not False:
        raise ValueError("M3 profile violates Test isolation")
    alpha = decisions.get("alpha_selection", {})
    scale = decisions.get("scale_selection", {})
    if alpha.get("selected_alpha") != 0.15 or alpha.get("selected_model") != "M3_alpha015":
        raise ValueError("Clean Dev V4 does not freeze M3 alpha=.15")
    if scale.get("selected_scale") != "S50":
        raise ValueError("Clean Dev V4 does not freeze S50")
    if decisions.get("test_evaluated") is not False:
        raise ValueError("Clean Dev V4 decisions violate Test isolation")

    original_prepare = root / "04_model_training/datasets/m3_dual_order_s50/prepare_summary.json"
    if not original_prepare.is_file():
        raise FileNotFoundError(original_prepare)
    expected_hash = load_json(original_prepare).get("source_checkpoint_sha256")
    actual_hash = sha256(args.rctc_init)
    if actual_hash != expected_hash:
        raise ValueError(
            "Refinement initialization differs from frozen M3: "
            f"expected={expected_hash}, actual={actual_hash}"
        )

    spec = TRACKS[args.track]
    method = spec["method"]
    binding = {
        "protocol_id": spec["protocol"],
        "track": args.track,
        "rctc_initialization": str(args.rctc_init),
        "rctc_initialization_sha256": actual_hash,
        "frozen_m3_prepare_summary": str(original_prepare),
        "frozen_m3_prepare_summary_sha256": sha256(original_prepare),
        "m3_clean_dev_v4_profile": str(args.m3_profile),
        "m3_clean_dev_v4_profile_sha256": sha256(args.m3_profile),
        "clean_dev_v4_manifest": str(manifest),
        "clean_dev_v4_manifest_sha256": sha256(manifest),
        "clean_dev_v4_final_decisions": str(decisions_path),
        "clean_dev_v4_final_decisions_sha256": sha256(decisions_path),
        "normalization": "normalization_v2 through frozen Clean Dev V4",
        "synthetic_scale": "S50",
        "preprocess_protocol": "P1_MSR_V3",
        "alpha_coc": 0.15,
        "seed": 20260731,
        "global_batch_size": 32,
        "optimizer_and_schedule": "same frozen M3 recipe",
        "only_changed_factor": spec["only_changed_factor"],
        "track_parameters": (
            {
                "reduction": 4,
                "gamma_init": 0.001,
                "kernels": ["3x3", "1x5", "5x1"],
                "fusion": "sum",
                "script_conditioned_gate": True,
            }
            if args.track == "sldr"
            else {
                "lambda": 0.05,
                "warmup_fraction": 0.2,
                "top_k": 5,
                "temperature": 0.1,
                "same_script_mining": True,
                "script_balanced_loss": True,
                "target_prototype_policy": "reset counts and prototypes before target",
            }
        ),
        "corrupted_dev_used": False,
        "test_evaluated": False,
    }
    binding_path = args.log_dir / "study_binding.json"
    if binding_path.is_file():
        existing_binding = load_json(binding_path)
        changed = {
            key: (existing_binding.get(key), value)
            for key, value in binding.items()
            if existing_binding.get(key) != value
        }
        if changed:
            raise ValueError(f"Full refinement binding changed during resume: {changed}")
        binding = existing_binding
    else:
        write_json(binding_path, binding)

    output_summary = root / f"05_evaluation/m3_{args.track}_full_s50_v1_clean_dev_v4/summary.json"
    if args.resume_existing and output_summary.is_file():
        previous = load_json(output_summary)
        if previous.get("status") == f"M3_{args.track.upper()}_FULL_S50_V1_COMPLETE":
            print(json.dumps({
                "status": "TRACK_ALREADY_COMPLETE",
                "track": args.track,
                "summary": str(output_summary),
                "summary_sha256": sha256(output_summary),
                "test_evaluated": False,
            }, indent=2))
            return

    synthetic_config = prepare(root, method, "synthetic", args.rctc_init, args)
    synthetic_preflight = preflight(root, synthetic_config, "synthetic", args)
    if args.preflight_only:
        preflight_result = {
            "status": f"M3_{args.track.upper()}_FULL_S50_V1_PREFLIGHT_OK",
            "track": args.track,
            "preflight": synthetic_preflight,
            "binding": binding,
            "code_sha256": {
                "runner": sha256(Path(__file__).resolve()),
                "protocol": sha256(
                    root / "scripts/svtrv2/dual_order_protocol.py"
                ),
                "validator": sha256(
                    root / "scripts/svtrv2/validate_dual_order_method_config.py"
                ),
                "smoke": sha256(
                    root / "scripts/svtrv2/smoke_dual_order_method.py"
                ),
            },
            "formal_training_started": False,
            "test_evaluated": False,
        }
        write_json(args.log_dir / "preflight_complete.json", preflight_result)
        print(json.dumps(preflight_result, ensure_ascii=False, indent=2))
        return

    synthetic = train(root, method, synthetic_config, "synthetic", args)
    synthetic_name, _ = names(method)
    synthetic_best = root / "04_model_training/runs" / synthetic_name / "best_clean_dev_macro_cer.pth"
    if not synthetic_best.is_file() or sha256(synthetic_best) != synthetic["best_checkpoint_sha256"]:
        raise ValueError("S50 best checkpoint provenance mismatch")

    target_source = synthetic_best
    if args.track == "scdl":
        target_source, reset_audit = reset_scdl_prototypes(
            root, synthetic_best, args.replace
        )
        binding["target_prototype_reset_audit"] = str(reset_audit)
        binding["target_prototype_reset_audit_sha256"] = sha256(reset_audit)
        write_json(binding_path, binding)

    target_config = prepare(root, method, "target", target_source, args)
    preflight(root, target_config, "target", args)
    target = train(root, method, target_config, "target", args)
    summarize(root, args, target, binding)


if __name__ == "__main__":
    main()
