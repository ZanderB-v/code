#!/usr/bin/env python3
"""Prepare B1 full SVTRv2-S SGM training on S50 or target-domain data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from p1_msr_protocol import (
    DEFAULT_MAX_RATIO,
    PROTOCOL_ID,
    ensure_synthetic_scale_lmdbs,
    ensure_target_lmdbs,
    make_full_svtrv2_s_config,
    preprocessing_summary,
    read_jsonl,
)
from prepare_e1_target_only import EXPECTED_COUNTS


MODEL_PROTOCOL = "B1_FULL_SVTRV2_S_SGM_U2_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("synthetic", "target"), required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-role", required=True)
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--batch-size-per-card", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--internal-eval-every", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--synthetic-root-name", default="synthetic_formal_v2")
    parser.add_argument("--replace-msr-lmdb", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names(stage: str) -> tuple[str, str]:
    if stage == "synthetic":
        return "b1_full_s50", "svtrv2_s_b1_full_s50"
    return "b1_full_s50_to_target", "svtrv2_s_b1_full_s50_to_target"


def count_lmdb_rows(manifests: dict) -> int:
    return sum(int(item["rows"]) for item in manifests.values())


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_checkpoint = args.source_checkpoint.resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    experiment_key, model_name = names(args.stage)

    target_root = (
        root
        / "01_data_preparation"
        / "real_line_dataset_eval_reviewed"
    )
    target_rows = read_jsonl(target_root / "metadata.jsonl")
    target_train_lmdbs, target_dev_lmdbs, target_manifests = (
        ensure_target_lmdbs(
            root,
            target_root,
            target_rows,
            replace=args.replace_msr_lmdb,
        )
    )
    if args.stage == "synthetic":
        formal_root = (
            root / "03_synthetic_generation" / args.synthetic_root_name
        )
        train_lmdbs, synthetic_manifests = ensure_synthetic_scale_lmdbs(
            root,
            formal_root,
            "s50",
            replace=args.replace_msr_lmdb,
        )
        expected_train_rows = 150000
        actual_train_rows = count_lmdb_rows(synthetic_manifests)
        data_role = "S50_synthetic_train"
    else:
        formal_root = None
        train_lmdbs = target_train_lmdbs
        synthetic_manifests = {}
        expected_train_rows = sum(EXPECTED_COUNTS["train"].values())
        actual_train_rows = int(target_manifests["train"]["rows"])
        data_role = "target_domain_train"
    if actual_train_rows != expected_train_rows:
        raise ValueError(
            f"{args.stage} train rows: expected {expected_train_rows}, "
            f"got {actual_train_rows}"
        )
    expected_dev_rows = sum(EXPECTED_COUNTS["dev"].values())
    if int(target_manifests["dev"]["rows"]) != expected_dev_rows:
        raise ValueError("Target dev LMDB count changed")

    dataset_dir = root / "04_model_training" / "datasets" / experiment_key
    config_dir = root / "04_model_training" / "configs"
    run_dir = root / "04_model_training" / "runs" / model_name
    for path in (dataset_dir, config_dir, run_dir):
        path.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{model_name}.yml"
    config_path.write_text(
        make_full_svtrv2_s_config(
            root=root,
            run_dir=run_dir,
            project_name=model_name,
            train_lmdbs=train_lmdbs,
            eval_lmdbs=target_dev_lmdbs,
            pretrained_model=source_checkpoint,
            max_epoch=args.max_epoch,
            first_batch_size=args.batch_size_per_card,
            num_workers=args.num_workers,
            max_ratio=DEFAULT_MAX_RATIO,
            lr=args.lr,
            internal_eval_every=args.internal_eval_every,
            seed=args.seed,
        ),
        encoding="utf-8",
    )

    summary = {
        "experiment": experiment_key,
        "model_name": model_name,
        "architecture": "full_SVTRv2-S_RCTC_plus_standard_SMTR_SGM",
        "model_protocol": MODEL_PROTOCOL,
        "stage": args.stage,
        "training_data_role": data_role,
        "train_rows": actual_train_rows,
        "dev_rows": expected_dev_rows,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "source_checkpoint_role": args.source_role,
        "training_supervision": {
            "zh": {"ctc": "logical_LTR", "sgm": "logical_LTR"},
            "ug": {
                "ctc": "U2_visual_order_python_bidi",
                "sgm": "U2_visual_order_python_bidi",
            },
            "kk": {"ctc": "logical_LTR", "sgm": "logical_LTR"},
            "same_label_for_both_branches": True,
        },
        "official_stage2_components": {
            "decoder": "GTCDecoder",
            "sgm": "SMTRDecoder",
            "loss": "GTCLoss",
            "ctc_weight": 0.1,
            "gtc_weight": 1.0,
            "sgm_branch_computed": True,
            "reported_inference_branch": "ctc",
        },
        "checkpoint_selection": "clean_target_dev_macro_CER_on_CTC_branch",
        "test_policy": "not_evaluated",
        "preprocess_protocol": PROTOCOL_ID,
        "preprocessing": preprocessing_summary(
            train_lmdbs,
            target_dev_lmdbs,
            args.batch_size_per_card,
            DEFAULT_MAX_RATIO,
        ),
        "target_counts": EXPECTED_COUNTS,
        "target_lmdb_manifests": target_manifests,
        "synthetic_lmdb_manifests": synthetic_manifests,
        "config": str(config_path),
        "run_dir": str(run_dir),
        "max_epoch": args.max_epoch,
        "lr": args.lr,
        "seed": args.seed,
    }
    (dataset_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
