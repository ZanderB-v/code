#!/usr/bin/env python3
"""Run the formal E0 -> E1 -> D2(S50) -> E5 P1/MSR baseline chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from p1_msr_protocol import EVAL_INFERENCE_BATCH_SIZE, PROTOCOL_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--target-eval-every", type=int, default=2)
    parser.add_argument("--synthetic-eval-every", type=int, default=5)
    parser.add_argument("--target-patience", type=int, default=5)
    parser.add_argument("--synthetic-patience", type=int, default=4)
    parser.add_argument("--target-min-epoch", type=int, default=10)
    parser.add_argument("--synthetic-min-epoch", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size-per-card", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ddp-gpus", default="0,1")
    parser.add_argument("--eval-gpu", type=int, default=0)
    parser.add_argument("--master-port-base", type=int, default=29610)
    parser.add_argument("--protocol-verify-workers", type=int, default=8)
    parser.add_argument("--synthetic-root-name", default="synthetic_formal_v2")
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


def prepare_common(args: argparse.Namespace) -> list[str]:
    return [
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


def validate_and_smoke(
    args: argparse.Namespace,
    config: Path,
    initialization: str,
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
                / "validate_p1_msr_config.py"
            ),
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
            str(
                args.root
                / "scripts"
                / "svtrv2"
                / "smoke_p1_rctc_config.py"
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
    initialization: str,
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
    e0_config = configs / "svtrv2_s_e0_random_target_only.yml"
    e1_config = configs / "svtrv2_s_e1_target_only.yml"
    d2_config = configs / "svtrv2_s_d2_synth50k.yml"
    d2_run = runs / "svtrv2_s_d2_synth50k"

    # Materialize and validate every input/config needed before the first
    # expensive training stage. D2 data problems must not appear days later.
    e0_prepare = [
        sys.executable,
        str(args.root / "scripts" / "svtrv2" / "prepare_e0_target_only.py"),
        *prepare_common(args),
    ]
    if args.rebuild_msr_lmdb:
        e0_prepare.append("--replace-msr-lmdb")
    run(e0_prepare, args.root)
    run(
        [
            sys.executable,
            str(args.root / "scripts" / "svtrv2" / "prepare_e1_target_only.py"),
            *prepare_common(args),
        ],
        args.root,
    )
    d2_prepare = [
        sys.executable,
        str(args.root / "scripts" / "svtrv2" / "prepare_d2_synth50k.py"),
        *prepare_common(args),
        "--scale",
        "s50",
        "--synthetic-root-name",
        args.synthetic_root_name,
    ]
    if args.rebuild_msr_lmdb:
        d2_prepare.append("--replace-msr-lmdb")
    run(d2_prepare, args.root)

    validate_and_smoke(args, e0_config, "random")
    validate_and_smoke(args, e1_config, "union14m")
    validate_and_smoke(args, d2_config, "union14m")
    print(
        "[preparation] E0/E1/D2 LMDBs, configs, initialization, forward, "
        "CTC loss, backward, gradients, and full-epoch DDP sampler parity all "
        "passed before formal training.",
        flush=True,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "RCTC_BASELINE_PREFLIGHT_ONLY_OK",
                    "preprocess_protocol": PROTOCOL_ID,
                    "training_started": False,
                    "validated_configs": [
                        str(e0_config),
                        str(e1_config),
                        str(d2_config),
                    ],
                    "note": (
                        "E5 is generated and validated only after D2 selects "
                        "its source checkpoint."
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
        stage="e0",
        config=e0_config,
        run_dir=runs / "svtrv2_s_e0_random_target_only",
        labels_dir=datasets / "e0_random_target_only" / "labels",
        label_prefix="target",
        eval_prefix="e0_random_target",
        initialization="random",
        eval_every=args.target_eval_every,
        patience=args.target_patience,
        min_epoch=args.target_min_epoch,
        master_port=args.master_port_base,
    )
    run_stage(
        args,
        stage="e1",
        config=e1_config,
        run_dir=runs / "svtrv2_s_e1_target_only",
        labels_dir=datasets / "e1_target_only" / "labels",
        label_prefix="target",
        eval_prefix="e1_target_only",
        initialization="union14m",
        eval_every=args.target_eval_every,
        patience=args.target_patience,
        min_epoch=args.target_min_epoch,
        master_port=args.master_port_base + 1,
    )
    run_stage(
        args,
        stage="d2",
        config=d2_config,
        run_dir=d2_run,
        labels_dir=datasets / "d2_synth50k" / "labels",
        label_prefix="real",
        eval_prefix="d2_synth50k",
        initialization="union14m",
        eval_every=args.synthetic_eval_every,
        patience=args.synthetic_patience,
        min_epoch=args.synthetic_min_epoch,
        master_port=args.master_port_base + 2,
    )

    d2_best = d2_run / "best_clean_dev_macro_cer.pth"
    if not d2_best.is_file():
        raise FileNotFoundError(f"D2 best checkpoint is missing: {d2_best}")
    run(
        [
            sys.executable,
            str(args.root / "scripts" / "svtrv2" / "prepare_e5_d2_to_target.py"),
            *prepare_common(args),
            "--source-scale",
            "s50",
            "--source-checkpoint",
            str(d2_best),
            "--source-config",
            str(d2_config),
        ],
        args.root,
    )
    run_stage(
        args,
        stage="e5",
        config=configs / "svtrv2_s_e5_d2_to_target.yml",
        run_dir=runs / "svtrv2_s_e5_d2_to_target",
        labels_dir=datasets / "e1_target_only" / "labels",
        label_prefix="target",
        eval_prefix="e5_d2_to_target",
        initialization="checkpoint",
        eval_every=args.target_eval_every,
        patience=args.target_patience,
        min_epoch=args.target_min_epoch,
        master_port=args.master_port_base + 3,
    )

    eval_root = args.root / "04_model_training" / "eval_reports"
    experiments = {}
    for name, prefix in (
        ("E0", "e0_random_target"),
        ("E1", "e1_target_only"),
        ("D2", "d2_synth50k"),
        ("E5", "e5_d2_to_target"),
    ):
        path = eval_root / f"{prefix}_final_summary.json"
        experiments[name] = json.loads(path.read_text(encoding="utf-8-sig"))
    summary = {
        "chain": "formal_RCTC_baseline_chain",
        "preprocess_protocol": PROTOCOL_ID,
        "external_dev_inference_batch_size": EVAL_INFERENCE_BATCH_SIZE,
        "checkpoint_selection": "clean_target_dev_macro_CER",
        "test_policy": "not_evaluated",
        "preflight": {
            "path": str(args.log_dir / "preflight.json"),
            "sha256": (
                sha256(args.log_dir / "preflight.json")
                if (args.log_dir / "preflight.json").is_file()
                else None
            ),
        },
        "experiments": experiments,
    }
    summary_path = eval_root / "rctc_baseline_chain_clean_dev_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
