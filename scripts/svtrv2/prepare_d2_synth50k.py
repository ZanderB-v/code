#!/usr/bin/env python3
"""Prepare P1/MSR synthetic pretraining at S10, S25, or S50 scale."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from p1_msr_protocol import (
    DEFAULT_FIRST_BATCH_SIZE,
    DEFAULT_MAX_RATIO,
    LANGUAGES,
    PROTOCOL_ID,
    SCALE_COUNTS,
    ensure_synthetic_scale_lmdbs,
    ensure_target_lmdbs,
    make_rctc_config,
    normalize_spaces,
    preprocessing_summary,
    read_jsonl,
    synthetic_subset_records,
    u2_text,
)
from prepare_e1_target_only import EXPECTED_COUNTS, write_lines


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
    parser.add_argument("--scale", choices=sorted(SCALE_COUNTS), default="s50")
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
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--synthetic-root-name",
        default="synthetic_formal_v2",
        help="Directory under 03_synthetic_generation containing frozen subsets.",
    )
    parser.add_argument(
        "--experiment-key",
        default=None,
        help="Dataset/evaluation prefix. Defaults to D2 for S50.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Run/config name. Defaults to the canonical D2 name for S50.",
    )
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


def canonical_names(args: argparse.Namespace) -> tuple[str, str]:
    if args.scale == "s50":
        experiment_key = args.experiment_key or "d2_synth50k"
        model_name = args.model_name or "svtrv2_s_d2_synth50k"
    else:
        experiment_key = args.experiment_key or f"scale_{args.scale}_pretrain"
        model_name = args.model_name or f"svtrv2_s_{args.scale}_synthetic_pretrain"
    return experiment_key, model_name


def write_synthetic_labels(
    labels_dir: Path,
    formal_root: Path,
    scale: str,
    records: list[dict],
) -> dict[str, int]:
    all_lines: list[str] = []
    counts = Counter(row["language"] for row in records)
    for language in LANGUAGES:
        language_rows = [row for row in records if row["language"] == language]
        lines = [
            f"{row['image']}\t{row['ctc_text']}"
            for row in language_rows
        ]
        write_lines(labels_dir / f"train_{scale}_{language}.txt", lines)
        all_lines.extend(lines)
    write_lines(labels_dir / f"train_{scale}_all.txt", all_lines)
    (labels_dir / "synthetic_data_root.txt").write_text(
        str(formal_root) + "\n",
        encoding="utf-8",
    )
    return dict(counts)


def write_real_eval_labels(
    labels_dir: Path,
    rows: list[dict],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("dev", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        counts[split] = dict(Counter(row.get("language") for row in split_rows))
        all_u2_lines: list[str] = []
        for language in LANGUAGES:
            language_rows = [
                row for row in split_rows if row.get("language") == language
            ]
            logical_lines: list[str] = []
            u2_lines: list[str] = []
            for row in language_rows:
                image = (row.get("image") or "").replace("\\", "/")
                logical = normalize_spaces(
                    row.get("logical_text") or row.get("text") or ""
                )
                if not image or not logical:
                    raise ValueError(f"Invalid target row: {row.get('id')}")
                logical_lines.append(f"{image}\t{logical}")
                u2_lines.append(
                    f"{image}\t{u2_text(language, logical)}"
                )
            write_lines(
                labels_dir / f"real_{split}_{language}_logical.txt",
                logical_lines,
            )
            write_lines(
                labels_dir / f"real_{split}_{language}_u2eval.txt",
                u2_lines,
            )
            all_u2_lines.extend(u2_lines)
        write_lines(
            labels_dir / f"real_{split}_all_u2eval.txt",
            all_u2_lines,
        )
    expected = {split: EXPECTED_COUNTS[split] for split in ("dev", "test")}
    if counts != expected:
        raise ValueError(
            f"Unexpected target eval counts: expected {expected}, got {counts}"
        )
    return counts


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    args.root = root
    experiment_key, model_name = canonical_names(args)
    formal_root = (
        root / "03_synthetic_generation" / args.synthetic_root_name
    )
    target_root = (
        root
        / "01_data_preparation"
        / "real_line_dataset_eval_reviewed"
    )
    target_metadata = target_root / "metadata.jsonl"

    out_root = root / "04_model_training" / "datasets" / experiment_key
    labels_dir = out_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    config_dir = root / "04_model_training" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    run_dir = root / "04_model_training" / "runs" / model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    synthetic_records = synthetic_subset_records(formal_root, args.scale)
    synthetic_counts = write_synthetic_labels(
        labels_dir,
        formal_root,
        args.scale,
        synthetic_records,
    )
    expected_synthetic = {
        language: SCALE_COUNTS[args.scale] for language in LANGUAGES
    }
    if synthetic_counts != expected_synthetic:
        raise ValueError(
            f"Unexpected synthetic counts: expected {expected_synthetic}, "
            f"got {synthetic_counts}"
        )

    target_rows = read_jsonl(target_metadata)
    target_eval_counts = write_real_eval_labels(labels_dir, target_rows)
    _, dev_lmdbs, target_lmdb_manifests = ensure_target_lmdbs(
        root,
        target_root,
        target_rows,
        replace=args.replace_msr_lmdb,
    )
    train_lmdbs, synthetic_lmdb_manifests = ensure_synthetic_scale_lmdbs(
        root,
        formal_root,
        args.scale,
        replace=args.replace_msr_lmdb,
    )

    pretrained = Path(
        "/home/wudayu/models/openocr_svtrv2/official/"
        "svtrv2_s_union14m/best.pth"
    )
    config_path = config_dir / f"{model_name}.yml"
    config_path.write_text(
        make_rctc_config(
            root=root,
            run_dir=run_dir,
            project_name=model_name,
            train_lmdbs=train_lmdbs,
            eval_lmdbs=dev_lmdbs,
            pretrained_model=pretrained,
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
        "task": "synthetic_pretraining",
        "experiment_key": experiment_key,
        "model_name": model_name,
        "scale": args.scale,
        "samples_per_language": SCALE_COUNTS[args.scale],
        "root": str(root),
        "synthetic_root": str(formal_root),
        "synthetic_counts": synthetic_counts,
        "target_eval_counts": target_eval_counts,
        "bidi_method": "python_bidi",
        "max_epoch": args.max_epoch,
        "lr": args.lr,
        "seed": args.seed,
        "labels_dir": str(labels_dir),
        "config": str(config_path),
        "run_dir": str(run_dir),
        "pretrained_model": str(pretrained),
        "preprocessing": preprocessing_summary(
            train_lmdbs,
            dev_lmdbs,
            args.batch_size_per_card,
            args.msr_max_ratio,
        ),
        "synthetic_msr_lmdb_manifests": synthetic_lmdb_manifests,
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
