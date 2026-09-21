#!/usr/bin/env python3
"""Prepare the P0 fixed-resize versus P1 MSR preprocessing ablation."""

from __future__ import annotations

import argparse
import io
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


LANGUAGES = ("zh", "ug", "kk")
EXPECTED_PILOT_PER_LANGUAGE = 5000


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
    parser.add_argument("--pilot-dir", type=Path, default=None)
    parser.add_argument("--max-epoch", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--p0-batch-size-per-card", type=int, default=16)
    parser.add_argument("--p1-first-batch-size-per-card", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-ratio", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--replace-lmdb", action="store_true")
    return parser.parse_args()


def normalize_spaces(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def load_bidi():
    try:
        from bidi.algorithm import get_display  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Preprocessing ablation requires python-bidi. Install it with: "
            "python -m pip install python-bidi"
        ) from exc
    return get_display


GET_DISPLAY = None


def ctc_text(language: str, logical: str) -> str:
    if language == "ug":
        assert GET_DISPLAY is not None
        return GET_DISPLAY(logical, base_dir="R")
    return logical


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}") from exc
    return rows


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def collect_pilot_rows(pilot_dir: Path) -> list[dict]:
    metadata_path = pilot_dir / "metadata.jsonl"
    rows = []
    for row in read_jsonl(metadata_path):
        language = row.get("language") or ""
        image = (row.get("image") or "").replace("\\", "/")
        logical = normalize_spaces(
            row.get("logical_text") or row.get("text") or ""
        )
        if language not in LANGUAGES or not image or not logical:
            continue
        image_path = pilot_dir / image
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing pilot image: {image_path}")
        rows.append(
            {
                "id": row.get("id") or "",
                "language": language,
                "image": image,
                "image_path": image_path,
                "logical_text": logical,
                "ctc_text": ctc_text(language, logical),
            }
        )
    counts = Counter(row["language"] for row in rows)
    expected = {
        language: EXPECTED_PILOT_PER_LANGUAGE for language in LANGUAGES
    }
    if dict(counts) != expected:
        raise SystemExit(
            f"Pilot counts must be exactly {expected}, got {dict(counts)}"
        )
    return rows


def collect_target_rows(target_dir: Path, split: str) -> list[dict]:
    rows = []
    for row in read_jsonl(target_dir / "metadata.jsonl"):
        if row.get("split") != split:
            continue
        language = row.get("language") or ""
        image = (row.get("image") or "").replace("\\", "/")
        logical = normalize_spaces(
            row.get("logical_text") or row.get("text") or ""
        )
        if language not in LANGUAGES or not image or not logical:
            continue
        image_path = target_dir / image
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing target image: {image_path}")
        rows.append(
            {
                "id": row.get("id") or "",
                "language": language,
                "image": image,
                "image_path": image_path,
                "logical_text": logical,
                "ctc_text": ctc_text(language, logical),
            }
        )
    return rows


def make_lmdb(rows: list[dict], output_dir: Path, replace: bool) -> None:
    try:
        import lmdb  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "LMDB is required for the official MSR pipeline. "
            "Install it with: python -m pip install lmdb"
        ) from exc

    if output_dir.exists():
        if not replace:
            raise SystemExit(
                f"LMDB already exists: {output_dir}. Use --replace-lmdb."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(row["image_path"].stat().st_size for row in rows)
    map_size = max(1 << 30, total_bytes * 4 + (64 << 20))
    env = lmdb.open(str(output_dir), map_size=map_size)
    cache: dict[bytes, bytes] = {}

    def flush() -> None:
        if not cache:
            return
        with env.begin(write=True) as transaction:
            for key, value in cache.items():
                transaction.put(key, value)
        cache.clear()

    for index, row in enumerate(rows, 1):
        image_bytes = row["image_path"].read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
        cache[f"image-{index:09d}".encode()] = image_bytes
        cache[f"label-{index:09d}".encode()] = row["ctc_text"].encode("utf-8")
        cache[f"wh-{index:09d}".encode()] = (
            f"{width}_{height}".encode("ascii")
        )
        if index % 1000 == 0:
            flush()
    cache[b"num-samples"] = str(len(rows)).encode("ascii")
    flush()
    env.sync()
    env.close()


def architecture_yaml() -> str:
    return """Architecture:
  model_type: rec
  algorithm: SVTRv2
  Transform:
  Encoder:
    name: SVTRv2LNConvTwo33
    use_pos_embed: False
    dims: [96, 192, 384]
    depths: [3, 6, 3]
    num_heads: [3, 6, 12]
    mixer: [['Conv','Conv','Conv'],['Conv','Conv','Conv','FGlobal','Global','Global'],['Global','Global','Global']]
    local_k: [[5, 5], [5, 5], [-1, -1]]
    sub_k: [[1, 1], [2, 1], [-1, -1]]
    last_stage: false
    feat2d: True
  Decoder:
    name: RCTCDecoder

Loss:
  name: CTCLoss
  zero_infinity: True

PostProcess:
  name: CTCLabelDecode
  character_dict_path: *character_dict_path
  use_space_char: *use_space_char

Metric:
  name: RecMetric
  main_indicator: acc
  ignore_space: False
  is_filter: False
"""


def global_yaml(
    args: argparse.Namespace,
    name: str,
    run_dir: Path,
    dictionary: Path,
) -> str:
    pretrained = (
        "/home/wudayu/models/openocr_svtrv2/official/"
        "svtrv2_s_union14m/best.pth"
    )
    return f"""Global:
  device: gpu
  epoch_num: {args.max_epoch}
  log_smooth_window: 20
  print_batch_step: 20
  output_dir: {run_dir}
  save_epoch_step: [0, 1]
  eval_epoch_step: [0, 100000000]
  eval_batch_step: [0, 100000000]
  cal_metric_during_train: True
  pretrained_model: {pretrained}
  checkpoints:
  use_tensorboard: false
  infer_img:
  character_dict_path: &character_dict_path {dictionary}
  max_text_length: &max_text_length 60
  use_space_char: &use_space_char True
  save_res_path: {run_dir / "predicts.txt"}
  use_amp: False
  grad_clip_val: 5.0
  project_name: {name}
  seed: {args.seed}

Optimizer:
  name: AdamW
  lr: {args.lr:.8f}
  weight_decay: 0.01
  filter_bias_and_bn: True

LRScheduler:
  name: OneCycleLR
  warmup_epoch: 1
  cycle_momentum: False

"""


def p0_dataset_yaml(
    args: argparse.Namespace,
    pilot_dir: Path,
    target_dir: Path,
    labels_dir: Path,
) -> str:
    return f"""Train:
  dataset:
    name: SimpleDataSet
    data_dir: {pilot_dir}
    label_file_list:
      - {labels_dir / "pilot_train_all_u2.txt"}
    transforms:
      - DecodeImagePIL:
          img_mode: RGB
      - CTCLabelEncode:
          character_dict_path: *character_dict_path
          use_space_char: *use_space_char
          max_text_length: *max_text_length
      - RecTVResize:
          image_shape: [48, 640]
          padding: True
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: True
    batch_size_per_card: {args.p0_batch_size_per_card}
    drop_last: True
    num_workers: {args.num_workers}

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {target_dir}
    label_file_list:
      - {labels_dir / "target_dev_all_u2.txt"}
    transforms:
      - DecodeImagePIL:
          img_mode: RGB
      - CTCLabelEncode:
          character_dict_path: *character_dict_path
          use_space_char: *use_space_char
          max_text_length: *max_text_length
      - RecTVResize:
          image_shape: [48, 640]
          padding: True
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: False
    batch_size_per_card: 32
    drop_last: False
    num_workers: {args.num_workers}
"""


def p1_dataset_yaml(
    args: argparse.Namespace,
    train_lmdb: Path,
    dev_lmdb: Path,
) -> str:
    common = """    transforms:
      - DecodeImagePIL:
          img_mode: RGB
      - CTCLabelEncode:
          character_dict_path: *character_dict_path
          use_space_char: *use_space_char
          max_text_length: *max_text_length
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
"""
    return f"""Train:
  dataset:
    name: RatioDataSetTVResize
    ds_width: True
    padding: False
    base_shape: [[64, 64], [96, 48], [112, 40], [128, 32]]
    base_h: 32
    data_dir_list:
      - {train_lmdb}
{common}  sampler:
    name: RatioSampler
    scales: [[128, 32]]
    first_bs: {args.p1_first_batch_size_per_card}
    fix_bs: False
    divided_factor: [4, 16]
    is_training: True
    max_ratio: {args.max_ratio}
  loader:
    shuffle: True
    batch_size_per_card: {args.p1_first_batch_size_per_card}
    drop_last: True
    max_ratio: {args.max_ratio}
    num_workers: {args.num_workers}

Eval:
  dataset:
    name: RatioDataSetTVResize
    ds_width: True
    padding: False
    base_shape: [[64, 64], [96, 48], [112, 40], [128, 32]]
    base_h: 32
    data_dir_list:
      - {dev_lmdb}
{common}  sampler:
    name: RatioSampler
    scales: [[128, 32]]
    first_bs: {args.p1_first_batch_size_per_card}
    fix_bs: False
    divided_factor: [4, 16]
    is_training: False
    max_ratio: {args.max_ratio}
  loader:
    shuffle: False
    batch_size_per_card: {args.p1_first_batch_size_per_card}
    drop_last: False
    max_ratio: {args.max_ratio}
    num_workers: {args.num_workers}
"""


def main() -> None:
    global GET_DISPLAY
    args = parse_args()
    GET_DISPLAY = load_bidi()
    root = args.root.resolve()
    pilot_dir = (
        args.pilot_dir.resolve()
        if args.pilot_dir
        else root
        / "03_synthetic_generation"
        / "preprocess_pilot_v1"
        / "synthetic_shard_0000_parallel"
    )
    target_dir = (
        root / "01_data_preparation" / "real_line_dataset_eval_reviewed"
    )
    out_dir = (
        root / "04_model_training" / "datasets" / "preprocess_ablation_v1"
    )
    labels_dir = out_dir / "labels"
    lmdb_dir = out_dir / "lmdb"
    config_dir = root / "04_model_training" / "configs"
    runs_dir = root / "04_model_training" / "runs"
    labels_dir.mkdir(parents=True, exist_ok=True)
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    pilot_rows = collect_pilot_rows(pilot_dir)
    target_dev_rows = collect_target_rows(target_dir, "dev")
    for name, rows in {
        "pilot_train": pilot_rows,
        "target_dev": target_dev_rows,
    }.items():
        write_lines(
            labels_dir / f"{name}_all_u2.txt",
            [f"{row['image']}\t{row['ctc_text']}" for row in rows],
        )
        for language in LANGUAGES:
            lang_rows = [
                row for row in rows if row["language"] == language
            ]
            write_lines(
                labels_dir / f"{name}_{language}_logical.txt",
                [
                    f"{row['image']}\t{row['logical_text']}"
                    for row in lang_rows
                ],
            )
            write_lines(
                labels_dir / f"{name}_{language}_u2.txt",
                [
                    f"{row['image']}\t{row['ctc_text']}"
                    for row in lang_rows
                ],
            )

    train_lmdb = lmdb_dir / "pilot_train_u2"
    dev_lmdb = lmdb_dir / "target_dev_u2"
    make_lmdb(pilot_rows, train_lmdb, args.replace_lmdb)
    make_lmdb(target_dev_rows, dev_lmdb, args.replace_lmdb)

    dictionary = (
        root
        / "04_model_training"
        / "character_dict_hz_ug_kk_v1"
        / "character_dict.txt"
    )
    p0_run = runs_dir / "svtrv2_s_preprocess_p0_fixed"
    p1_run = runs_dir / "svtrv2_s_preprocess_p1_msr"
    p0_run.mkdir(parents=True, exist_ok=True)
    p1_run.mkdir(parents=True, exist_ok=True)
    p0_config = config_dir / "svtrv2_s_preprocess_p0_fixed.yml"
    p1_config = config_dir / "svtrv2_s_preprocess_p1_msr.yml"
    p0_config.write_text(
        global_yaml(args, "svtrv2_s_preprocess_p0_fixed", p0_run, dictionary)
        + architecture_yaml()
        + p0_dataset_yaml(args, pilot_dir, target_dir, labels_dir),
        encoding="utf-8",
    )
    p1_config.write_text(
        global_yaml(args, "svtrv2_s_preprocess_p1_msr", p1_run, dictionary)
        + architecture_yaml()
        + p1_dataset_yaml(args, train_lmdb, dev_lmdb),
        encoding="utf-8",
    )

    summary = {
        "experiment": "preprocess_ablation_v1",
        "pilot_dir": str(pilot_dir),
        "target_dir": str(target_dir),
        "pilot_counts": dict(Counter(row["language"] for row in pilot_rows)),
        "target_dev_counts": dict(
            Counter(row["language"] for row in target_dev_rows)
        ),
        "training_data_identical": True,
        "model_architecture": "SVTRv2-S + RCTC",
        "initialization": (
            "/home/wudayu/models/openocr_svtrv2/official/"
            "svtrv2_s_union14m/best.pth"
        ),
        "p0": {
            "preprocessing": "RecTVResize 48x640 aspect-preserving right padding",
            "config": str(p0_config),
            "run_dir": str(p0_run),
        },
        "p1": {
            "preprocessing": (
                "Official RatioDataSetTVResize MSR shapes "
                "[64x64,96x48,112x40,128x32], dynamic long width"
            ),
            "max_ratio": args.max_ratio,
            "config": str(p1_config),
            "run_dir": str(p1_run),
            "train_lmdb": str(train_lmdb),
            "dev_lmdb": str(dev_lmdb),
        },
        "max_epoch": args.max_epoch,
        "eval_every": args.eval_every,
        "selection_split": "target_dev_only",
        "test_used": False,
    }
    (out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
