#!/usr/bin/env python3
"""Prepare one controlled B2/M1/M2/M3/Full dual-order training stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dual_order_protocol import (
    DUAL_ORDER_DATA_PROTOCOL,
    METHODS,
    consistency_weight_path_tag,
    ensure_dual_synthetic_lmdbs,
    ensure_dual_target_lmdbs,
    make_method_config,
    method_summary,
)
from p1_msr_protocol import DEFAULT_MAX_RATIO, PROTOCOL_ID, read_jsonl
from prepare_e1_target_only import EXPECTED_COUNTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--stage", choices=("synthetic", "target"), required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-role", required=True)
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--batch-size-per-card", type=int, default=16)
    parser.add_argument("--eval-batch-size-per-card", type=int)
    parser.add_argument("--control-world-size", type=int, default=1)
    parser.add_argument("--control-first-batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--consistency-weight", type=float)
    parser.add_argument("--internal-eval-every", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--synthetic-root-name", default="synthetic_formal_v2")
    parser.add_argument("--replace-dual-order-lmdb", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names(
    method: str,
    stage: str,
    consistency_weight: float | None = None,
) -> tuple[str, str]:
    suffix = "s50" if stage == "synthetic" else "s50_to_target"
    method_name = f"{method}_dual_order"
    if consistency_weight is not None:
        if method != "m2":
            raise ValueError(
                "A consistency-weight override is only valid for method m2"
            )
        method_name = (
            f"m2_alpha_{consistency_weight_path_tag(consistency_weight)}"
        )
    return (
        f"{method_name}_{suffix}",
        f"svtrv2_s_{method_name}_{suffix}",
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_checkpoint = args.source_checkpoint.resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    experiment_key, model_name = names(
        args.method,
        args.stage,
        args.consistency_weight,
    )

    target_root = (
        root / "01_data_preparation" / "real_line_dataset_eval_reviewed"
    )
    target_rows = read_jsonl(target_root / "metadata.jsonl")
    target_train, target_dev, target_manifests = ensure_dual_target_lmdbs(
        root,
        target_root,
        target_rows,
        replace=args.replace_dual_order_lmdb,
    )
    if args.stage == "synthetic":
        formal_root = (
            root / "03_synthetic_generation" / args.synthetic_root_name
        )
        train_lmdbs, synthetic_manifests = ensure_dual_synthetic_lmdbs(
            root,
            formal_root,
            "s50",
            replace=args.replace_dual_order_lmdb,
        )
        expected_train = 150000
        actual_train = sum(
            int(item["rows"]) for item in synthetic_manifests.values()
        )
        data_role = "frozen_S50_synthetic"
    else:
        train_lmdbs = target_train
        synthetic_manifests = {}
        expected_train = sum(EXPECTED_COUNTS["train"].values())
        actual_train = int(target_manifests["train"]["rows"])
        data_role = "frozen_target_train"
    expected_dev = sum(EXPECTED_COUNTS["dev"].values())
    if actual_train != expected_train:
        raise ValueError(
            f"Train rows changed: expected {expected_train}, got {actual_train}"
        )
    if int(target_manifests["dev"]["rows"]) != expected_dev:
        raise ValueError("Target dev rows changed")

    dataset_dir = root / "04_model_training" / "datasets" / experiment_key
    config_dir = root / "04_model_training" / "configs"
    run_dir = root / "04_model_training" / "runs" / model_name
    for path in (dataset_dir, config_dir, run_dir):
        path.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{model_name}.yml"
    config_path.write_text(
        make_method_config(
            method=args.method,
            root=root,
            run_dir=run_dir,
            project_name=model_name,
            train_lmdbs=train_lmdbs,
            eval_lmdbs=target_dev,
            pretrained_model=source_checkpoint,
            max_epoch=args.max_epoch,
            first_batch_size=args.batch_size_per_card,
            num_workers=args.num_workers,
            max_ratio=DEFAULT_MAX_RATIO,
            lr=args.lr,
            internal_eval_every=args.internal_eval_every,
            seed=args.seed,
            consistency_weight=args.consistency_weight,
            eval_batch_size=args.eval_batch_size_per_card,
            control_world_size=args.control_world_size,
            control_first_batch_size=args.control_first_batch_size,
        ),
        encoding="utf-8",
    )

    summary = {
        "experiment": experiment_key,
        "model_name": model_name,
        "stage": args.stage,
        "training_data_role": data_role,
        "train_rows": actual_train,
        "dev_rows": expected_dev,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "source_checkpoint_role": args.source_role,
        "method": method_summary(
            args.method,
            consistency_weight=args.consistency_weight,
        ),
        "controlled_constants": {
            "encoder": "SVTRv2-S",
            "ctc_decoder": "RCTCDecoder",
            "semantic_decoder": "SMTRDecoder",
            "synthetic_data": "frozen S50",
            "target_data": "frozen Protocol V2 train/dev",
            "preprocessing": PROTOCOL_ID,
            "dual_order_data": DUAL_ORDER_DATA_PROTOCOL,
            "dictionary": "character_dict_hz_ug_kk_v1",
            "checkpoint_selection": "clean_target_dev_macro_CER",
            "test_policy": "not_evaluated",
        },
        "target_lmdb_manifests": target_manifests,
        "synthetic_lmdb_manifests": synthetic_manifests,
        "config": str(config_path),
        "run_dir": str(run_dir),
        "max_epoch": args.max_epoch,
        "lr": args.lr,
        "seed": args.seed,
    }
    (dataset_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
