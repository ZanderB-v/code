#!/usr/bin/env python3
"""Run the controlled S50 -> target M3-noCOC ablation on Clean Dev V4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


METHOD = "m3_nococ"
SYNTHETIC_NAME = "svtrv2_s_m3_nococ_dual_order_s50"
TARGET_NAME = "svtrv2_s_m3_nococ_dual_order_s50_to_target"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--rctc-init", type=Path, required=True)
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
    parser.add_argument("--rebuild-dual-order-lmdb", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def prepare(root: Path, stage: str, source: Path, args: argparse.Namespace) -> Path:
    command = [
        sys.executable,
        str(root / "scripts/svtrv2/prepare_dual_order_method.py"),
        "--root", str(root),
        "--method", METHOD,
        "--stage", stage,
        "--source-checkpoint", str(source),
        "--source-role", (
            "same_frozen_RCTC_initialization_as_M3"
            if stage == "synthetic"
            else "M3_noCOC_S50_CER_selected_checkpoint"
        ),
        "--max-epoch", "50",
        "--batch-size-per-card", str(args.batch_size),
        # Runtime is one physical GPU with a global batch of 32. Preserve the
        # original two-rank 16+16 RatioSampler bucket schedule so optimizer
        # steps and OneCycleLR horizon remain matched to frozen M3.
        "--control-world-size", "2",
        "--control-first-batch-size", "16",
        "--num-workers", str(args.num_workers),
        "--lr", "2.5e-5",
        "--seed", "20260731",
    ]
    if args.rebuild_dual_order_lmdb and stage == "synthetic":
        command.append("--replace-dual-order-lmdb")
    run(command, root)
    name = SYNTHETIC_NAME if stage == "synthetic" else TARGET_NAME
    return root / "04_model_training/configs" / f"{name}.yml"


def preflight(root: Path, config: Path, stage: str, args: argparse.Namespace) -> None:
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
    payload = json.loads(smoke.read_text(encoding="utf-8-sig"))
    if payload.get("status") != "DUAL_ORDER_REAL_BATCH_FORWARD_BACKWARD_OK":
        raise RuntimeError(f"Failed smoke report: {smoke}")


def train(root: Path, config: Path, stage: str, args: argparse.Namespace) -> dict:
    name = SYNTHETIC_NAME if stage == "synthetic" else TARGET_NAME
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
        "--master-port", "29931" if stage == "synthetic" else "29932",
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
    summary_path = root / "04_model_training/eval_reports" / f"{name}_final_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    return json.loads(summary_path.read_text(encoding="utf-8-sig"))


def summarize(root: Path, m3_profile: Path, nococ: dict, binding: dict) -> None:
    m3_summary = json.loads(m3_profile.read_text(encoding="utf-8-sig"))
    m3 = {
        **m3_summary["metrics"]["macro"],
        "languages": m3_summary["metrics"]["languages"],
    }
    nococ_dev = nococ["dev"]
    delta = float(nococ_dev["macro_cer"]) - float(m3["macro_cer"])
    tie_margin = 0.0002  # 0.02 percentage points.
    if delta >= tie_margin:
        interpretation = "COC_SUPPORTED_BY_CANONICAL_SEED"
    elif delta <= -tie_margin:
        interpretation = "NOCOC_FAVORED_REQUIRES_SEED_CONFIRMATION"
    else:
        interpretation = "INCONCLUSIVE_WITHIN_0.02PP"
    rows = []
    for model, dev in (("M3_alpha015", m3), ("M3_noCOC", nococ_dev)):
        languages = dev["languages"]
        rows.append(
            {
                "model": model,
                "macro_cer": dev["macro_cer"],
                "macro_line_accuracy": dev["macro_line_accuracy"],
                "zh_cer": languages["zh"]["cer"],
                "ug_cer": languages["ug"]["cer"],
                "kk_cer": languages["kk"]["cer"],
            }
        )
    output_dir = root / "05_evaluation/m3_nococ_clean_dev_v4"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "m3_nococ_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "M3_NOCOC_ABLATION_COMPLETE",
        "selection_metric": "Clean Dev V4 Macro CER",
        "tie_margin_absolute_cer": tie_margin,
        "m3_noCOC_minus_m3_macro_cer": delta,
        "interpretation": interpretation,
        "models": rows,
        "binding": binding,
        "corrupted_dev_used": False,
        "test_evaluated": False,
    }
    (output_dir / "m3_nococ_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.replace and args.resume_existing:
        raise ValueError("--replace and --resume-existing cannot be combined")
    root = args.root.resolve()
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.rctc_init = args.rctc_init.resolve()
    args.m3_profile = args.m3_profile.resolve()
    args.v4_decision_gate = args.v4_decision_gate.resolve()
    args.v4_reselection_dir = args.v4_reselection_dir.resolve()
    args.v4_protocol_dir = args.v4_protocol_dir.resolve()
    if not args.rctc_init.is_file():
        raise FileNotFoundError(
            f"The exact pre-SGM RCTC initialization used by frozen M3 is required: {args.rctc_init}"
        )
    for path in (args.m3_profile, args.v4_decision_gate):
        if not path.is_file():
            raise FileNotFoundError(path)
    v4_manifest = args.v4_protocol_dir / args.v4_frozen_manifest
    if not v4_manifest.is_file():
        raise FileNotFoundError(v4_manifest)
    gate = json.loads(args.v4_decision_gate.read_text(encoding="utf-8-sig"))
    if gate.get("status") != "MANUAL_REVIEW_COMPLETE":
        raise ValueError("Clean Dev V4 error-profile review is incomplete")
    if gate.get("m3_nococ_allowed_after_profile_review") is not True:
        raise ValueError("Clean Dev V4 does not authorize M3-noCOC")
    if gate.get("test_evaluated") is not False:
        raise ValueError("Clean Dev V4 gate violates Test isolation")
    m3_profile = json.loads(args.m3_profile.read_text(encoding="utf-8-sig"))
    if m3_profile.get("status") != "CLEAN_DEV_V4_M3_ERROR_PROFILE_READY":
        raise ValueError("M3 profile is not bound to Clean Dev V4")
    if m3_profile.get("alpha") != 0.15 or m3_profile.get("selected_epoch") != 34:
        raise ValueError("M3-noCOC requires frozen M3 alpha=.15 at V4 epoch 34")
    if m3_profile.get("test_evaluated") is not False:
        raise ValueError("M3 profile violates Test isolation")
    decisions_path = args.v4_reselection_dir / "final_decisions.json"
    if not decisions_path.is_file():
        raise FileNotFoundError(decisions_path)
    decisions = json.loads(decisions_path.read_text(encoding="utf-8-sig"))
    alpha = decisions.get("alpha_selection", {})
    scale = decisions.get("scale_selection", {})
    if alpha.get("selected_alpha") != 0.15 or alpha.get("selected_model") != "M3_alpha015":
        raise ValueError("Clean Dev V4 alpha selection is not frozen at M3 alpha=.15")
    if scale.get("selected_scale") != "S50":
        raise ValueError("Clean Dev V4 scale selection is not frozen at S50")

    original_prepare_path = (
        root / "04_model_training/datasets/m3_dual_order_s50/prepare_summary.json"
    )
    if not original_prepare_path.is_file():
        raise FileNotFoundError(
            "The frozen M3 S50 prepare summary is required to prove matched "
            f"initialization: {original_prepare_path}"
        )
    original_prepare = json.loads(
        original_prepare_path.read_text(encoding="utf-8-sig")
    )
    expected_init_hash = original_prepare.get("source_checkpoint_sha256")
    actual_init_hash = sha256(args.rctc_init)
    if actual_init_hash != expected_init_hash:
        raise ValueError(
            "M3-noCOC initialization differs from frozen M3: "
            f"expected={expected_init_hash}, actual={actual_init_hash}"
        )

    binding = {
        "protocol": "S50_M3_NOCOC_CLEAN_DEV_V4_ABLATION_V1",
        "rctc_initialization": str(args.rctc_init),
        "rctc_initialization_sha256": actual_init_hash,
        "frozen_m3_prepare_summary": str(original_prepare_path),
        "frozen_m3_prepare_summary_sha256": sha256(original_prepare_path),
        "m3_clean_dev_v4_profile": str(args.m3_profile),
        "m3_clean_dev_v4_profile_sha256": sha256(args.m3_profile),
        "clean_dev_v4_gate": str(args.v4_decision_gate),
        "clean_dev_v4_gate_sha256": sha256(args.v4_decision_gate),
        "clean_dev_v4_manifest": str(v4_manifest),
        "clean_dev_v4_manifest_sha256": sha256(v4_manifest),
        "clean_dev_v4_final_decisions": str(decisions_path),
        "clean_dev_v4_final_decisions_sha256": sha256(decisions_path),
        "synthetic_scale": "S50",
        "preprocess_protocol": "P1_MSR_V3",
        "global_batch_size": args.batch_size,
        "physical_gpu": args.gpu,
        "seed": 20260731,
        "only_changed_factor": "Cross-Order Consistency weight 0.15 -> 0.0",
        "test_evaluated": False,
    }
    binding_path = args.log_dir / "study_binding.json"
    if binding_path.is_file():
        existing = json.loads(binding_path.read_text(encoding="utf-8-sig"))
        if existing != binding:
            raise ValueError("M3-noCOC study binding changed during resume")
    else:
        binding_path.write_text(
            json.dumps(binding, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # The target-data contract is verified by prepare_dual_order_method and
    # validate_dual_order_method_config.  Clean Dev V4 is independently
    # hash-bound above; invoking the historical Protocol V2 freezer here would
    # incorrectly make a new ablation depend on a superseded Dev protocol.
    synthetic_config = prepare(root, "synthetic", args.rctc_init, args)
    preflight(root, synthetic_config, "synthetic", args)
    if args.preflight_only:
        print(json.dumps({"status": "M3_NOCOC_PREFLIGHT_OK", **binding}, indent=2))
        return

    synthetic = train(root, synthetic_config, "synthetic", args)
    synthetic_best = root / "04_model_training/runs" / SYNTHETIC_NAME / "best_clean_dev_macro_cer.pth"
    if sha256(synthetic_best) != synthetic["best_checkpoint_sha256"]:
        raise ValueError("M3-noCOC S50 best checkpoint provenance mismatch")
    target_config = prepare(root, "target", synthetic_best, args)
    preflight(root, target_config, "target", args)
    target = train(root, target_config, "target", args)
    summarize(root, args.m3_profile, target, binding)


if __name__ == "__main__":
    main()
