#!/usr/bin/env python3
"""Prepare P1/MSR synthetic-pretrain to target-domain fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from p1_msr_protocol import (
    DEFAULT_FIRST_BATCH_SIZE,
    DEFAULT_MAX_RATIO,
    PROTOCOL_ID,
    SCALE_COUNTS,
    assert_p1_config_text,
    ensure_target_lmdbs,
    make_rctc_config,
    preprocessing_summary,
    read_jsonl,
)
from prepare_e1_target_only import EXPECTED_COUNTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/data_home/wudayu/experiments/multilingual_meme_ocr/"
            "svtrv2_line_recognition"
        ),
    )
    parser.add_argument("--source-scale", choices=sorted(SCALE_COUNTS), default="s50")
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument(
        "--batch-size-per-card",
        type=int,
        default=DEFAULT_FIRST_BATCH_SIZE,
        help="P1/MSR first batch size per GPU at scale 128x32.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Keep equal to E1 for the controlled checkpoint-initialization comparison.",
    )
    parser.add_argument("--internal-eval-every", type=int, default=100000)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260725,
        help="Matched to E1 so the controlled comparison changes only initialization.",
    )
    parser.add_argument(
        "--source-checkpoint",
        "--d2-checkpoint",
        dest="source_checkpoint",
        type=Path,
        default=None,
        help="Synthetic-pretraining checkpoint selected only by clean target dev.",
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=None,
        help="Synthetic-pretraining config; used to enforce P1/MSR compatibility.",
    )
    parser.add_argument("--experiment-key", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--msr-max-ratio",
        type=int,
        default=DEFAULT_MAX_RATIO,
        choices=[DEFAULT_MAX_RATIO],
        help="Frozen P1/MSR maximum width-height ratio.",
    )
    parser.add_argument(
        "--replace-msr-lmdb",
        action="store_true",
        help="Rebuild P1/MSR LMDB caches after a verified data change.",
    )
    return parser.parse_args()


def canonical_names(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.source_scale == "s50":
        return (
            args.experiment_key or "e5_d2_to_target",
            args.model_name or "svtrv2_s_e5_d2_to_target",
            "svtrv2_s_d2_synth50k",
        )
    return (
        args.experiment_key or f"scale_{args.source_scale}_to_target",
        args.model_name or f"svtrv2_s_{args.source_scale}_to_target",
        f"svtrv2_s_{args.source_scale}_synthetic_pretrain",
    )


def count_label_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def validate_target_labels(labels_dir: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for split, language_counts in EXPECTED_COUNTS.items():
        all_path = labels_dir / f"target_{split}_all_u2.txt"
        if not all_path.is_file():
            raise FileNotFoundError(
                f"Missing target labels: {all_path}\n"
                "Run prepare_e1_target_only.py once to materialize shared labels."
            )
        expected_total = sum(language_counts.values())
        actual_total = count_label_rows(all_path)
        if actual_total != expected_total:
            raise ValueError(
                f"Unexpected row count in {all_path}: "
                f"expected {expected_total}, got {actual_total}"
            )
        result[split] = {"path": str(all_path), "rows": actual_total}
        for language, expected in language_counts.items():
            for order in ("logical", "u2"):
                label_path = (
                    labels_dir
                    / f"target_{split}_{language}_{order}.txt"
                )
                if not label_path.is_file():
                    raise FileNotFoundError(f"Missing target label: {label_path}")
                actual = count_label_rows(label_path)
                if actual != expected:
                    raise ValueError(
                        f"Unexpected row count in {label_path}: "
                        f"expected {expected}, got {actual}"
                    )
    return result


def resolve_source(
    root: Path,
    args: argparse.Namespace,
    source_model_name: str,
) -> tuple[Path, Path, dict]:
    run_dir = root / "04_model_training" / "runs" / source_model_name
    config_path = args.source_config or (
        root / "04_model_training" / "configs" / f"{source_model_name}.yml"
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing source config for compatibility check: {config_path}"
        )
    assert_p1_config_text(config_path.read_text(encoding="utf-8-sig"))

    best_json_path = run_dir / "best_clean_dev_macro_cer.json"
    best_meta = {}
    if best_json_path.is_file():
        best_meta = json.loads(best_json_path.read_text(encoding="utf-8-sig"))
        recorded_protocol = best_meta.get("preprocess_protocol")
        if recorded_protocol not in (None, PROTOCOL_ID):
            raise ValueError(
                f"Source checkpoint metadata uses {recorded_protocol}, "
                f"expected {PROTOCOL_ID}"
            )

    checkpoint = args.source_checkpoint or (
        run_dir / "best_clean_dev_macro_cer.pth"
    )
    if not checkpoint.is_file() and args.source_checkpoint is None:
        recorded = best_meta.get("copied_checkpoint") or best_meta.get("checkpoint")
        if recorded and Path(recorded).is_file():
            checkpoint = Path(recorded)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing synthetic-pretraining checkpoint: {checkpoint}"
        )
    return checkpoint.resolve(), config_path.resolve(), best_meta


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    args.root = root
    experiment_key, model_name, source_model_name = canonical_names(args)
    target_root = (
        root
        / "01_data_preparation"
        / "real_line_dataset_eval_reviewed"
    )
    target_metadata_path = target_root / "metadata.jsonl"
    target_rows = read_jsonl(target_metadata_path)
    labels_dir = (
        root / "04_model_training" / "datasets" / "e1_target_only" / "labels"
    )
    required_labels = validate_target_labels(labels_dir)

    out_root = root / "04_model_training" / "datasets" / experiment_key
    config_dir = root / "04_model_training" / "configs"
    run_dir = root / "04_model_training" / "runs" / model_name
    out_root.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    source_checkpoint, source_config, source_best_meta = resolve_source(
        root,
        args,
        source_model_name,
    )
    train_lmdbs, dev_lmdbs, target_lmdb_manifests = ensure_target_lmdbs(
        root,
        target_root,
        target_rows,
        replace=args.replace_msr_lmdb,
    )

    config_path = config_dir / f"{model_name}.yml"
    config_path.write_text(
        make_rctc_config(
            root=root,
            run_dir=run_dir,
            project_name=model_name,
            train_lmdbs=train_lmdbs,
            eval_lmdbs=dev_lmdbs,
            pretrained_model=source_checkpoint,
            max_epoch=args.max_epoch,
            first_batch_size=args.batch_size_per_card,
            num_workers=args.num_workers,
            max_ratio=args.msr_max_ratio,
            lr=args.lr,
            internal_eval_every=args.internal_eval_every,
            seed=args.seed,
        ),
        encoding="utf-8",
    )

    summary = {
        "experiment": "synthetic_pretrain_then_target_finetune",
        "experiment_key": experiment_key,
        "model_name": model_name,
        "source_scale": args.source_scale,
        "samples_per_language": SCALE_COUNTS[args.source_scale],
        "comparison_control": {
            "reference": "E1_target_domain_only",
            "changed_factor": "initial_checkpoint_only",
            "same_target_train_data": True,
            "same_character_dictionary": True,
            "same_preprocessing_protocol": PROTOCOL_ID,
            "same_optimizer_lr_as_e1": args.lr == 5e-5,
            "same_external_selection_metric": "clean_target_dev_macro_CER",
        },
        "target_root": str(target_root),
        "counts": EXPECTED_COUNTS,
        "labels": required_labels,
        "training_label_order": {
            "zh": "logical_ltr",
            "ug": "u2_visual_order_python_bidi",
            "kk": "logical_ltr",
        },
        "initial_checkpoint": str(source_checkpoint),
        "initial_checkpoint_role": (
            f"{args.source_scale}_synthetic_pretraining_best_clean_target_dev"
        ),
        "source_config": str(source_config),
        "source_best_metadata": source_best_meta,
        "config": str(config_path),
        "run_dir": str(run_dir),
        "max_epoch": args.max_epoch,
        "lr": args.lr,
        "seed": args.seed,
        "preprocessing": preprocessing_summary(
            train_lmdbs,
            dev_lmdbs,
            args.batch_size_per_card,
            args.msr_max_ratio,
        ),
        "target_msr_lmdb_manifests": target_lmdb_manifests,
        "preprocess_protocol": PROTOCOL_ID,
        "selection_metric": "clean_target_dev_macro_CER",
        "test_policy": "not_evaluated_during_model_development",
    }
    (out_root / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
