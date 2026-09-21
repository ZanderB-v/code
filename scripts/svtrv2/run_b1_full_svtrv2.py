#!/usr/bin/env python3
"""Run B1: RCTC Stage 1 -> full SVTRv2 SGM on S50 -> target fine-tune."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from p1_msr_protocol import EVAL_INFERENCE_BATCH_SIZE, PROTOCOL_ID


MODEL_PROTOCOL = "B1_FULL_SVTRV2_S_SGM_U2_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--synthetic-eval-every", type=int, default=5)
    parser.add_argument("--target-eval-every", type=int, default=2)
    parser.add_argument("--synthetic-patience", type=int, default=4)
    parser.add_argument("--target-patience", type=int, default=5)
    parser.add_argument("--synthetic-min-epoch", type=int, default=15)
    parser.add_argument("--target-min-epoch", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size-per-card", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--ddp-gpus", default="0,1")
    parser.add_argument("--eval-gpu", type=int, default=0)
    parser.add_argument("--master-port-base", type=int, default=29740)
    parser.add_argument("--protocol-verify-workers", type=int, default=8)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--rebuild-msr-lmdb", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def run(command: list[str], cwd: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_current_stage_summary(
    root: Path,
    summary_path: Path,
    checkpoint_path: Path,
    config_path: Path,
) -> dict:
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    protocol_manifest = json.loads(
        (
            root
            / "00_docs"
            / "frozen_protocol_v2"
            / "protocol_v2_manifest.json"
        ).read_text(encoding="utf-8-sig")
    )
    expected = {
        "preprocess_protocol": PROTOCOL_ID,
        "external_eval_batch_size": EVAL_INFERENCE_BATCH_SIZE,
        "data_protocol_fingerprint": protocol_manifest.get(
            "verification_fingerprint_sha256"
        ),
        "best_checkpoint_sha256": sha256(checkpoint_path),
        "config_sha256": sha256(config_path),
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "B1 refuses to consume an incompatible D2 artifact: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    return summary


def prepare(
    args: argparse.Namespace,
    *,
    stage: str,
    source_checkpoint: Path,
    source_role: str,
) -> None:
    command = [
        sys.executable,
        str(args.root / "scripts" / "svtrv2" / "prepare_b1_full_svtrv2.py"),
        "--root",
        str(args.root),
        "--stage",
        stage,
        "--source-checkpoint",
        str(source_checkpoint),
        "--source-role",
        source_role,
        "--max-epoch",
        str(args.max_epoch),
        "--batch-size-per-card",
        str(args.batch_size_per_card),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--internal-eval-every",
        "100000",
    ]
    if args.rebuild_msr_lmdb:
        command.append("--replace-msr-lmdb")
    run(command, args.root)


def validate_and_smoke(
    args: argparse.Namespace,
    config: Path,
    b0_config: Path,
    stage: str,
) -> None:
    run(
        [
            sys.executable,
            str(
                args.root
                / "scripts"
                / "svtrv2"
                / "audit_p1_ratio_sampler_ddp.py"
            ),
            "--root",
            str(args.root),
            "--config",
            str(config),
            "--world-size",
            str(len([item for item in args.ddp_gpus.split(",") if item.strip()])),
            "--output",
            str(args.log_dir / f"{config.stem}_ddp_sampler_audit.json"),
        ],
        args.root,
    )
    run(
        [
            sys.executable,
            str(
                args.root
                / "scripts"
                / "svtrv2"
                / "audit_b1_controlled_comparison.py"
            ),
            "--b0-config",
            str(b0_config),
            "--b1-config",
            str(config),
            "--stage",
            stage,
            "--output",
            str(args.log_dir / f"{stage}_controlled_comparison.json"),
        ],
        args.root,
    )
    run(
        [
            sys.executable,
            str(
                args.root
                / "scripts"
                / "svtrv2"
                / "validate_p1_full_svtrv2_config.py"
            ),
            "--config",
            str(config),
            "--expected-initialization",
            "checkpoint",
        ],
        args.root,
    )
    run(
        [
            sys.executable,
            str(
                args.root
                / "scripts"
                / "svtrv2"
                / "smoke_p1_full_svtrv2_config.py"
            ),
            "--root",
            str(args.root),
            "--config",
            str(config),
            "--device-id",
            str(args.eval_gpu),
        ],
        args.root,
    )


def run_stage(
    args: argparse.Namespace,
    *,
    stage: str,
    config: Path,
    run_dir: Path,
    labels_dir: Path,
    label_prefix: str,
    eval_prefix: str,
    eval_every: int,
    patience: int,
    min_epoch: int,
    master_port: int,
) -> None:
    command = [
        sys.executable,
        str(args.root / "scripts" / "svtrv2" / "run_p1_rctc_stage.py"),
        "--root",
        str(args.root),
        "--stage",
        stage,
        "--config",
        str(config),
        "--run-dir",
        str(run_dir),
        "--labels-dir",
        str(labels_dir),
        "--label-prefix",
        label_prefix,
        "--eval-prefix",
        eval_prefix,
        "--expected-initialization",
        "checkpoint",
        "--config-kind",
        "full_svtrv2",
        "--prediction-branch",
        "ctc",
        "--max-epoch",
        str(args.max_epoch),
        "--eval-every",
        str(eval_every),
        "--patience-evals",
        str(patience),
        "--min-epoch",
        str(min_epoch),
        "--min-delta",
        str(args.min_delta),
        "--ddp-gpus",
        args.ddp_gpus,
        "--eval-gpu",
        str(args.eval_gpu),
        "--master-port",
        str(master_port),
        "--log-dir",
        str(args.log_dir),
    ]
    if args.replace:
        command.append("--replace")
    run(command, args.root)


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            str(args.root / "scripts" / "protocol" / "freeze_protocol_v2.py"),
            "--root",
            str(args.root),
            "--mode",
            "verify",
            "--workers",
            str(args.protocol_verify_workers),
        ],
        args.root,
    )

    configs = args.root / "04_model_training" / "configs"
    runs = args.root / "04_model_training" / "runs"
    datasets = args.root / "04_model_training" / "datasets"
    eval_root = args.root / "04_model_training" / "eval_reports"

    d2_run = runs / "svtrv2_s_d2_synth50k"
    d2_best = d2_run / "best_clean_dev_macro_cer.pth"
    d2_meta = d2_run / "best_clean_dev_macro_cer.json"
    d2_config = configs / "svtrv2_s_d2_synth50k.yml"
    d2_summary_path = eval_root / "d2_synth50k_final_summary.json"
    for required in (d2_best, d2_meta, d2_config, d2_summary_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"B1 requires the completed B0/D2 Stage-1 artifact: {required}"
            )
    require_current_stage_summary(
        args.root,
        d2_summary_path,
        d2_best,
        d2_config,
    )

    adapter_dir = (
        args.root / "04_model_training" / "checkpoint_adapters" / "b1_full_s50"
    )
    adapted_d2 = adapter_dir / "d2_rctc_for_full_svtrv2.pth"
    convert = [
        sys.executable,
        str(
            args.root
            / "scripts"
            / "svtrv2"
            / "convert_rctc_checkpoint_to_full_svtrv2.py"
        ),
        "--input",
        str(d2_best),
        "--output",
        str(adapted_d2),
    ]
    if args.replace or not adapted_d2.exists():
        convert.append("--replace")
        run(convert, args.root)

    synthetic_config = configs / "svtrv2_s_b1_full_s50.yml"
    synthetic_run = runs / "svtrv2_s_b1_full_s50"
    prepare(
        args,
        stage="synthetic",
        source_checkpoint=adapted_d2,
        source_role="B0_D2_best_RCTC_remapped_into_nested_CTC_branch",
    )
    validate_and_smoke(args, synthetic_config, d2_config, "synthetic")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "B1_SYNTHETIC_PREFLIGHT_ONLY_OK",
                    "preprocess_protocol": PROTOCOL_ID,
                    "external_eval_batch_size": EVAL_INFERENCE_BATCH_SIZE,
                    "validated_config": str(synthetic_config),
                    "training_started": False,
                    "note": (
                        "The target-stage config depends on the trained B1 S50 "
                        "checkpoint and is validated immediately before target "
                        "fine-tuning."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    run_stage(
        args,
        stage="b1_full_s50",
        config=synthetic_config,
        run_dir=synthetic_run,
        labels_dir=datasets / "d2_synth50k" / "labels",
        label_prefix="real",
        eval_prefix="b1_full_s50",
        eval_every=args.synthetic_eval_every,
        patience=args.synthetic_patience,
        min_epoch=args.synthetic_min_epoch,
        master_port=args.master_port_base,
    )

    synthetic_best = synthetic_run / "best_clean_dev_macro_cer.pth"
    if not synthetic_best.is_file():
        raise FileNotFoundError(synthetic_best)
    target_config = configs / "svtrv2_s_b1_full_s50_to_target.yml"
    target_run = runs / "svtrv2_s_b1_full_s50_to_target"
    prepare(
        args,
        stage="target",
        source_checkpoint=synthetic_best,
        source_role="B1_full_S50_best_clean_target_dev_CTC_branch",
    )
    validate_and_smoke(
        args,
        target_config,
        configs / "svtrv2_s_e5_d2_to_target.yml",
        "target",
    )
    run_stage(
        args,
        stage="b1_full_s50_to_target",
        config=target_config,
        run_dir=target_run,
        labels_dir=datasets / "e1_target_only" / "labels",
        label_prefix="target",
        eval_prefix="b1_full_s50_to_target",
        eval_every=args.target_eval_every,
        patience=args.target_patience,
        min_epoch=args.target_min_epoch,
        master_port=args.master_port_base + 1,
    )

    b1_final_path = (
        eval_root / "b1_full_s50_to_target_final_summary.json"
    )
    b1_final = json.loads(b1_final_path.read_text(encoding="utf-8-sig"))
    b0_final_path = eval_root / "e5_d2_to_target_final_summary.json"
    b0_final = (
        json.loads(b0_final_path.read_text(encoding="utf-8-sig"))
        if b0_final_path.is_file()
        else None
    )
    summary = {
        "comparison": "B0_RCTC_vs_B1_full_SVTRv2",
        "preprocess_protocol": PROTOCOL_ID,
        "external_dev_inference_batch_size": EVAL_INFERENCE_BATCH_SIZE,
        "model_protocol": MODEL_PROTOCOL,
        "controlled_constants": {
            "synthetic_data": "same frozen S50",
            "target_train_dev": "same frozen Protocol V2",
            "preprocessing": PROTOCOL_ID,
            "dictionary": "character_dict_hz_ug_kk_v1",
            "uyghur_label": "U2_visual_order_python_bidi_for_CTC_and_SGM",
            "checkpoint_selection": "clean_target_dev_macro_CER",
            "inference_branch": "RCTC_only",
        },
        "B0_E5": b0_final,
        "B1_full": b1_final,
        "test_policy": "not_evaluated",
    }
    summary_path = eval_root / "b1_full_svtrv2_clean_dev_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
