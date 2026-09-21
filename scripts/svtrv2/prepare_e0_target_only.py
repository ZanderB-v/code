#!/usr/bin/env python3
"""Prepare E0 random-initialized target-domain-only training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from p1_msr_protocol import (
    DEFAULT_FIRST_BATCH_SIZE,
    DEFAULT_MAX_RATIO,
    PROTOCOL_ID,
    ensure_target_lmdbs,
    make_rctc_config,
    preprocessing_summary,
)
from prepare_e1_target_only import (
    EXPECTED_COUNTS,
    LANGS,
    logical_text,
    read_jsonl,
    u2_text,
    write_lines,
)


EXPERIMENT = "e0_random_target_only"
MODEL_NAME = "svtrv2_s_e0_random_target_only"


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
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument(
        "--batch-size-per-card",
        type=int,
        default=DEFAULT_FIRST_BATCH_SIZE,
        help="P1/MSR first batch size per GPU at scale 128x32.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--internal-eval-every", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260725)
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


def make_random_init_config(
    args: argparse.Namespace,
    run_dir: Path,
    train_lmdbs: list[Path],
    dev_lmdbs: list[Path],
) -> str:
    return make_rctc_config(
        root=args.root,
        run_dir=run_dir,
        project_name=MODEL_NAME,
        train_lmdbs=train_lmdbs,
        eval_lmdbs=dev_lmdbs,
        pretrained_model=None,
        max_epoch=args.max_epoch,
        first_batch_size=args.batch_size_per_card,
        num_workers=args.num_workers,
        max_ratio=args.msr_max_ratio,
        lr=args.lr,
        internal_eval_every=args.internal_eval_every,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    target_root = (
        root
        / "01_data_preparation"
        / "real_line_dataset_eval_reviewed"
    )
    metadata_path = target_root / "metadata.jsonl"
    rows = read_jsonl(metadata_path)

    out_root = root / "04_model_training" / "datasets" / EXPERIMENT
    labels_dir = out_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    config_dir = root / "04_model_training" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    run_dir = root / "04_model_training" / "runs" / MODEL_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "dev", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        counts[split] = dict(
            Counter(row.get("language") for row in split_rows)
        )
        all_u2_lines: list[str] = []
        for lang in LANGS:
            lang_rows = [
                row
                for row in split_rows
                if row.get("language") == lang
            ]
            logical_lines: list[str] = []
            u2_lines: list[str] = []
            for row in lang_rows:
                image = (row.get("image") or "").replace("\\", "/")
                logical = logical_text(row)
                if not image or not logical:
                    raise ValueError(
                        f"Empty image/text in metadata row: {row.get('id')}"
                    )
                logical_lines.append(f"{image}\t{logical}")
                u2_lines.append(f"{image}\t{u2_text(row)}")

            write_lines(
                labels_dir / f"target_{split}_{lang}_logical.txt",
                logical_lines,
            )
            write_lines(
                labels_dir / f"target_{split}_{lang}_u2.txt",
                u2_lines,
            )
            all_u2_lines.extend(u2_lines)

        write_lines(
            labels_dir / f"target_{split}_all_u2.txt",
            all_u2_lines,
        )

    if counts != EXPECTED_COUNTS:
        raise SystemExit(
            f"Unexpected target-domain counts. Expected {EXPECTED_COUNTS}, got {counts}"
        )
    expected_total = {
        split: sum(language_counts.values())
        for split, language_counts in EXPECTED_COUNTS.items()
    }
    actual_total = {
        split: sum(lang_counts.values())
        for split, lang_counts in counts.items()
    }
    if actual_total != expected_total:
        raise SystemExit(
            "Unexpected target-domain counts. "
            f"Expected {expected_total}, got {actual_total}"
        )

    train_lmdbs, dev_lmdbs, lmdb_manifests = ensure_target_lmdbs(
        root,
        target_root,
        rows,
        replace=args.replace_msr_lmdb,
    )
    config_path = config_dir / f"{MODEL_NAME}.yml"
    config_path.write_text(
        make_random_init_config(
            args,
            run_dir,
            train_lmdbs,
            dev_lmdbs,
        ),
        encoding="utf-8",
    )

    summary = {
        "experiment": "E0_random_initialization_target_domain_only",
        "purpose": (
            "Controlled initialization ablation against E1. Architecture, "
            "target-domain data, labels, dictionary, P1/MSR preprocessing, optimizer, and "
            "evaluation protocol are kept unchanged."
        ),
        "dataset_description": {
            "zh": "natural Chinese meme lines",
            "ug": "target-domain re-rendered Uyghur meme lines",
            "kk": "target-domain re-rendered Kazakh meme lines",
        },
        "root": str(root),
        "target_root": str(target_root),
        "metadata": str(metadata_path),
        "counts": counts,
        "totals": actual_total,
        "training_label_order": {
            "zh": "logical_ltr",
            "ug": "u2_visual_order_python_bidi",
            "kk": "logical_ltr",
        },
        "labels_dir": str(labels_dir),
        "config": str(config_path),
        "run_dir": str(run_dir),
        "initialization": "random",
        "pretrained_model": None,
        "max_epoch": args.max_epoch,
        "lr": args.lr,
        "seed": args.seed,
        "preprocessing": preprocessing_summary(
            train_lmdbs,
            dev_lmdbs,
            args.batch_size_per_card,
            args.msr_max_ratio,
        ),
        "msr_lmdb_manifests": lmdb_manifests,
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
