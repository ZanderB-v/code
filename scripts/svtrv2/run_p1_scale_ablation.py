#!/usr/bin/env python3
"""Run nested S10/S25/S50 synthetic-pretrain to target-finetune ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from p1_msr_protocol import EVAL_INFERENCE_BATCH_SIZE, PROTOCOL_ID


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--scales", nargs="+", choices=("s10", "s25", "s50"), default=["s10", "s25", "s50"])
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--batch-size-per-card", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ddp-gpus", default="0,1")
    parser.add_argument("--eval-gpu", type=int, default=0)
    parser.add_argument("--master-port-base", type=int, default=29710)
    parser.add_argument("--synthetic-root-name", default="synthetic_formal_v2")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--force-rerun-s50",
        action="store_true",
        help="Rerun canonical D2/E5 instead of reusing the formal baseline chain.",
    )
    return parser.parse_args()


def run(command: list[str], cwd: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def validate_and_smoke(
    args: argparse.Namespace,
    config: Path,
    initialization: str,
) -> None:
    run(
        [
            sys.executable,
            str(args.root / "scripts" / "svtrv2" / "audit_p1_ratio_sampler_ddp.py"),
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
            str(args.root / "scripts" / "svtrv2" / "validate_p1_msr_config.py"),
            "--config",
            str(config),
            "--expected-initialization",
            initialization,
        ],
        args.root,
    )
    run(
        [
            sys.executable,
            str(args.root / "scripts" / "svtrv2" / "smoke_p1_rctc_config.py"),
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
    initialization: str,
    eval_every: int,
    patience: int,
    min_epoch: int,
    master_port: int,
) -> None:
    final_summary = (
        args.root
        / "04_model_training"
        / "eval_reports"
        / f"{eval_prefix}_final_summary.json"
    )
    if args.resume and final_summary.is_file():
        summary = json.loads(final_summary.read_text(encoding="utf-8-sig"))
        if summary.get("preprocess_protocol") != PROTOCOL_ID:
            raise ValueError(f"Cannot reuse incompatible summary: {final_summary}")
        print(f"[{stage}] reusing completed stage: {final_summary}", flush=True)
        return
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
        initialization,
        "--max-epoch",
        str(args.max_epoch),
        "--eval-every",
        str(eval_every),
        "--patience-evals",
        str(patience),
        "--min-epoch",
        str(min_epoch),
        "--min-delta",
        "0.0001",
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
    elif args.resume:
        command.append("--resume-existing")
    run(command, args.root)


def canonical_names(scale: str) -> tuple[str, str, str, str]:
    if scale == "s50":
        return (
            "d2_synth50k",
            "svtrv2_s_d2_synth50k",
            "e5_d2_to_target",
            "svtrv2_s_e5_d2_to_target",
        )
    return (
        f"scale_{scale}_pretrain",
        f"svtrv2_s_{scale}_synthetic_pretrain",
        f"scale_{scale}_to_target",
        f"svtrv2_s_{scale}_to_target",
    )


def existing_s50_is_reusable(root: Path) -> bool:
    required = (
        root / "04_model_training" / "runs" / "svtrv2_s_d2_synth50k" / "best_clean_dev_macro_cer.pth",
        root / "04_model_training" / "runs" / "svtrv2_s_e5_d2_to_target" / "best_clean_dev_macro_cer.pth",
        root / "04_model_training" / "eval_reports" / "d2_synth50k_final_summary.json",
        root / "04_model_training" / "eval_reports" / "e5_d2_to_target_final_summary.json",
    )
    if not all(path.is_file() for path in required):
        return False
    protocol_manifest = json.loads(
        (
            root
            / "00_docs"
            / "frozen_protocol_v2"
            / "protocol_v2_manifest.json"
        ).read_text(encoding="utf-8-sig")
    )
    current_fingerprint = protocol_manifest.get(
        "verification_fingerprint_sha256"
    )
    for checkpoint_path, summary_path in zip(required[:2], required[2:]):
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        if summary.get("preprocess_protocol") != PROTOCOL_ID:
            return False
        if summary.get("external_eval_batch_size") != EVAL_INFERENCE_BATCH_SIZE:
            return False
        if summary.get("data_protocol_fingerprint") != current_fingerprint:
            return False
        if summary.get("best_checkpoint_sha256") != sha256(checkpoint_path):
            return False
        config_path = Path(str(summary.get("config") or ""))
        if not config_path.is_file():
            return False
        if summary.get("config_sha256") != sha256(config_path):
            return False
    return True


def main() -> None:
    args = parse_args()
    if args.replace and args.resume:
        raise ValueError("--replace and --resume are mutually exclusive")
    args.root = args.root.resolve()
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    configs = args.root / "04_model_training" / "configs"
    runs = args.root / "04_model_training" / "runs"
    datasets = args.root / "04_model_training" / "datasets"
    eval_root = args.root / "04_model_training" / "eval_reports"

    run(
        [
            sys.executable,
            str(args.root / "scripts" / "protocol" / "freeze_protocol_v2.py"),
            "--root",
            str(args.root),
            "--mode",
            "verify",
            "--workers",
            "8",
        ],
        args.root,
    )

    # Materialize and validate shared target labels/LMDB without training E1.
    run(
        [
            sys.executable,
            str(args.root / "scripts" / "svtrv2" / "prepare_e1_target_only.py"),
            "--root",
            str(args.root),
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
        ],
        args.root,
    )

    summaries = {}
    for scale_index, scale in enumerate(args.scales):
        pretrain_key, pretrain_model, target_key, target_model = canonical_names(scale)
        common_prepare = [
            "--root",
            str(args.root),
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
        if args.preflight_only:
            run(
                [
                    sys.executable,
                    str(args.root / "scripts" / "svtrv2" / "prepare_d2_synth50k.py"),
                    *common_prepare,
                    "--scale",
                    scale,
                    "--synthetic-root-name",
                    args.synthetic_root_name,
                    "--experiment-key",
                    pretrain_key,
                    "--model-name",
                    pretrain_model,
                ],
                args.root,
            )
            validate_and_smoke(
                args, configs / f"{pretrain_model}.yml", "union14m"
            )
            continue
        if (
            scale == "s50"
            and not args.force_rerun_s50
            and existing_s50_is_reusable(args.root)
        ):
            print("[s50] reusing verified D2/E5 baseline results", flush=True)
        else:
            run(
                [
                    sys.executable,
                    str(args.root / "scripts" / "svtrv2" / "prepare_d2_synth50k.py"),
                    *common_prepare,
                    "--scale",
                    scale,
                    "--synthetic-root-name",
                    args.synthetic_root_name,
                    "--experiment-key",
                    pretrain_key,
                    "--model-name",
                    pretrain_model,
                ],
                args.root,
            )
            pretrain_config = configs / f"{pretrain_model}.yml"
            pretrain_run = runs / pretrain_model
            validate_and_smoke(args, pretrain_config, "union14m")
            run_stage(
                args,
                stage=f"{scale}_pretrain",
                config=pretrain_config,
                run_dir=pretrain_run,
                labels_dir=datasets / pretrain_key / "labels",
                label_prefix="real",
                eval_prefix=pretrain_key,
                initialization="union14m",
                eval_every=5,
                patience=4,
                min_epoch=15,
                master_port=args.master_port_base + scale_index * 2,
            )
            pretrain_best = pretrain_run / "best_clean_dev_macro_cer.pth"
            run(
                [
                    sys.executable,
                    str(args.root / "scripts" / "svtrv2" / "prepare_e5_d2_to_target.py"),
                    *common_prepare,
                    "--source-scale",
                    scale,
                    "--source-checkpoint",
                    str(pretrain_best),
                    "--source-config",
                    str(pretrain_config),
                    "--experiment-key",
                    target_key,
                    "--model-name",
                    target_model,
                ],
                args.root,
            )
            target_config = configs / f"{target_model}.yml"
            validate_and_smoke(args, target_config, "checkpoint")
            run_stage(
                args,
                stage=f"{scale}_to_target",
                config=target_config,
                run_dir=runs / target_model,
                labels_dir=datasets / "e1_target_only" / "labels",
                label_prefix="target",
                eval_prefix=target_key,
                initialization="checkpoint",
                eval_every=2,
                patience=5,
                min_epoch=10,
                master_port=args.master_port_base + scale_index * 2 + 1,
            )

        summaries[scale] = {
            "pretrain": json.loads(
                (eval_root / f"{pretrain_key}_final_summary.json").read_text(
                    encoding="utf-8-sig"
                )
            ),
            "target_finetune": json.loads(
                (eval_root / f"{target_key}_final_summary.json").read_text(
                    encoding="utf-8-sig"
                )
            ),
        }

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "RCTC_SCALE_PREFLIGHT_ONLY_OK",
                    "preprocess_protocol": PROTOCOL_ID,
                    "external_eval_batch_size": EVAL_INFERENCE_BATCH_SIZE,
                    "validated_synthetic_scales": args.scales,
                    "training_started": False,
                    "note": (
                        "Target fine-tune configs require each scale's trained "
                        "pretrain checkpoint and are validated immediately before "
                        "their target stage."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    result = {
        "experiment": "nested_synthetic_scale_ablation",
        "preprocess_protocol": PROTOCOL_ID,
        "data_protocol_fingerprint": json.loads(
            (
                args.root
                / "00_docs"
                / "frozen_protocol_v2"
                / "protocol_v2_manifest.json"
            ).read_text(encoding="utf-8-sig")
        ).get("verification_fingerprint_sha256"),
        "nested_scales": args.scales,
        "checkpoint_selection": "clean_target_dev_macro_CER",
        "external_eval_batch_size": EVAL_INFERENCE_BATCH_SIZE,
        "compute_policy": (
            "fixed maximum epochs with dev early stopping; larger scales "
            "therefore receive more optimizer steps per epoch"
        ),
        "test_policy": "not_evaluated",
        "results": summaries,
    }
    output = eval_root / "rctc_scale_ablation_clean_dev_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
