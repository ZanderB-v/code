#!/usr/bin/env python3
"""Shared P1/MSR data and configuration helpers.

All formal SVTRv2 experiments must use this module instead of embedding resize
blocks in individual preparation scripts. The protocol matches the P1 setting
selected by the preprocessing ablation.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_ID = "P1_MSR_V3"
SAMPLER_PROTOCOL_ID = "RATIO_SAMPLER_DDP_BALANCED_V2"
EVAL_INFERENCE_BATCH_SIZE = 1
BASE_SHAPES = [[64, 64], [96, 48], [112, 40], [128, 32]]
BASE_HEIGHT = 32
SAMPLER_SCALES = [[128, 32]]
DIVIDED_FACTOR = [4, 16]
DEFAULT_MAX_RATIO = 40
DEFAULT_FIRST_BATCH_SIZE = 32
MAX_TEXT_LENGTH = 120
LANGUAGES = ("zh", "ug", "kk")
SCALE_COUNTS = {"s10": 10000, "s25": 25000, "s50": 50000}


def normalize_spaces(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


@lru_cache(maxsize=1)
def get_bidi_display():
    try:
        from bidi.algorithm import get_display  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "P1/MSR formal experiments require python-bidi. Install it in "
            "openocr_svtrv2 with: python -m pip install python-bidi"
        ) from exc
    return get_display


def u2_text(language: str, text: str) -> str:
    logical = normalize_spaces(text)
    if language == "ug":
        return get_bidi_display()(logical, base_dir="R")
    return logical


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_record_fingerprint(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        payload = {
            "id": row["id"],
            "language": row["language"],
            "image": row["image"],
            "ctc_text": row["ctc_text"],
        }
        digest.update(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def target_records(
    target_root: Path,
    metadata_rows: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    records = []
    for row in metadata_rows:
        if row.get("split") != split:
            continue
        language = row.get("language") or ""
        image = (row.get("image") or "").replace("\\", "/")
        logical = normalize_spaces(row.get("logical_text") or row.get("text") or "")
        if language not in LANGUAGES or not image or not logical:
            raise ValueError(f"Invalid target row: {row.get('id')}")
        image_path = target_root / image
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing target image: {image_path}")
        records.append(
            {
                "id": row.get("id") or f"{split}:{len(records)}",
                "language": language,
                "image": image,
                "image_path": image_path,
                "logical_text": logical,
                "ctc_text": u2_text(language, logical),
            }
        )
    return records


def synthetic_subset_records(
    formal_root: Path,
    scale: str,
) -> list[dict[str, Any]]:
    if scale not in SCALE_COUNTS:
        raise ValueError(f"Unsupported synthetic scale: {scale}")
    metadata_path = formal_root / "subsets" / scale / "metadata.jsonl"
    rows = read_jsonl(metadata_path)
    records = []
    for row in rows:
        language = row.get("language") or ""
        image = (row.get("image") or "").replace("\\", "/")
        logical = normalize_spaces(row.get("logical_text") or "")
        ctc = normalize_spaces(row.get("ctc_text") or "")
        if language not in LANGUAGES or not image or not logical or not ctc:
            raise ValueError(f"Invalid synthetic row: {row.get('id')}")
        if language == "ug":
            if row.get("ctc_text_u2_status") != "python_bidi":
                raise ValueError(
                    f"Uyghur sample is not strict python-bidi U2: {row.get('id')}"
                )
            if ctc != normalize_spaces(row.get("ctc_text_u2") or ""):
                raise ValueError(
                    f"Uyghur ctc_text differs from ctc_text_u2: {row.get('id')}"
                )
        image_path = formal_root / image
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing synthetic image: {image_path}")
        records.append(
            {
                "id": row.get("id") or f"{scale}:{len(records)}",
                "language": language,
                "image": image,
                "image_path": image_path,
                "logical_text": logical,
                "ctc_text": ctc,
            }
        )
    expected = {language: SCALE_COUNTS[scale] for language in LANGUAGES}
    counts = dict(Counter(row["language"] for row in records))
    if counts != expected:
        raise ValueError(
            f"Unexpected {scale} counts: expected {expected}, got {counts}"
        )
    return records


def ensure_lmdb(
    records: list[dict[str, Any]],
    output_dir: Path,
    source_name: str,
    replace: bool = False,
) -> dict[str, Any]:
    """Create or validate an atomic LMDB cache for RatioDataSetTVResize."""

    fingerprint = _canonical_record_fingerprint(records)
    counts = dict(Counter(row["language"] for row in records))
    expected_manifest = {
        "preprocess_protocol": PROTOCOL_ID,
        "sampler_protocol": SAMPLER_PROTOCOL_ID,
        "source_name": source_name,
        "record_fingerprint_sha256": fingerprint,
        "rows": len(records),
        "language_counts": counts,
    }
    manifest_path = output_dir / "msr_lmdb_manifest.json"
    if output_dir.exists() and not replace:
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Unverified LMDB exists without manifest: {output_dir}. "
                "Use --replace-msr-lmdb."
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        for key, expected in expected_manifest.items():
            if existing.get(key) != expected:
                raise RuntimeError(
                    f"Stale LMDB at {output_dir}: {key} is "
                    f"{existing.get(key)!r}, expected {expected!r}. "
                    "Use --replace-msr-lmdb."
                )
        return existing

    try:
        import lmdb  # type: ignore
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "P1/MSR requires lmdb and Pillow in openocr_svtrv2. Install with: "
            "python -m pip install lmdb pillow"
        ) from exc

    temporary_dir = output_dir.with_name(output_dir.name + ".building")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    temporary_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = sum(row["image_path"].stat().st_size for row in records)
    map_size = max(1 << 30, total_bytes * 4 + (128 << 20))
    env = lmdb.open(str(temporary_dir), map_size=map_size)
    cache: dict[bytes, bytes] = {}

    def flush() -> None:
        if not cache:
            return
        with env.begin(write=True) as transaction:
            for key, value in cache.items():
                transaction.put(key, value)
        cache.clear()

    for index, row in enumerate(records, 1):
        image_bytes = row["image_path"].read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
        cache[f"image-{index:09d}".encode()] = image_bytes
        cache[f"label-{index:09d}".encode()] = row["ctc_text"].encode("utf-8")
        cache[f"wh-{index:09d}".encode()] = f"{width}_{height}".encode("ascii")
        if index % 1000 == 0:
            flush()
        if index % 10000 == 0:
            print(
                json.dumps(
                    {
                        "lmdb": str(output_dir),
                        "written": index,
                        "rows": len(records),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    cache[b"num-samples"] = str(len(records)).encode("ascii")
    flush()
    env.sync()
    env.close()

    manifest = {
        **expected_manifest,
        "path": str(output_dir),
        "map_size": map_size,
        "image_bytes": total_bytes,
    }
    (temporary_dir / "msr_lmdb_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_dir.rename(output_dir)
    return manifest


def ensure_target_lmdbs(
    root: Path,
    target_root: Path,
    metadata_rows: list[dict[str, Any]],
    replace: bool = False,
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    cache_root = root / "04_model_training" / "msr_lmdb" / PROTOCOL_ID
    train_records = target_records(target_root, metadata_rows, "train")
    dev_records = target_records(target_root, metadata_rows, "dev")
    train_path = cache_root / "target_train_u2"
    dev_path = cache_root / "target_dev_u2"
    manifests = {
        "train": ensure_lmdb(
            train_records, train_path, "target_train_u2", replace=replace
        ),
        "dev": ensure_lmdb(
            dev_records, dev_path, "target_dev_u2", replace=replace
        ),
    }
    return [train_path], [dev_path], manifests


def ensure_synthetic_scale_lmdbs(
    root: Path,
    formal_root: Path,
    scale: str,
    replace: bool = False,
) -> tuple[list[Path], dict[str, Any]]:
    """Build nested incremental LMDBs so larger scales reuse smaller stores."""

    if scale not in SCALE_COUNTS:
        raise ValueError(f"Unsupported scale: {scale}")
    cache_root = (
        root
        / "04_model_training"
        / "msr_lmdb"
        / PROTOCOL_ID
        / formal_root.name
    )
    ordered_scales = ("s10", "s25", "s50")
    selected_paths: list[Path] = []
    manifests: dict[str, Any] = {}
    previous_ids: set[str] = set()

    for current in ordered_scales:
        current_records = synthetic_subset_records(formal_root, current)
        current_ids = {row["id"] for row in current_records}
        if previous_ids and not previous_ids.issubset(current_ids):
            raise ValueError(f"Nested subset invariant failed at {current}")
        increment = [
            row for row in current_records if row["id"] not in previous_ids
        ]
        suffix = current if current == "s10" else f"{current}_increment"
        output_path = cache_root / suffix
        manifests[current] = ensure_lmdb(
            increment,
            output_path,
            f"{formal_root.name}_{suffix}",
            replace=replace,
        )
        selected_paths.append(output_path)
        previous_ids = current_ids
        if current == scale:
            break

    expected_total = SCALE_COUNTS[scale] * len(LANGUAGES)
    actual_total = sum(manifest["rows"] for manifest in manifests.values())
    if actual_total != expected_total:
        raise ValueError(
            f"Incremental LMDB total for {scale} is {actual_total}, "
            f"expected {expected_total}"
        )
    return selected_paths, manifests


def _yaml_paths(paths: list[Path], indent: int = 6) -> str:
    spaces = " " * indent
    return "\n".join(f"{spaces}- {path}" for path in paths)


def msr_dataset_yaml(
    train_lmdbs: list[Path],
    eval_lmdbs: list[Path],
    first_batch_size: int,
    num_workers: int,
    max_ratio: int,
) -> str:
    base_shape = json.dumps(BASE_SHAPES, separators=(",", ":"))
    scales = json.dumps(SAMPLER_SCALES, separators=(",", ":"))
    divided = json.dumps(DIVIDED_FACTOR, separators=(",", ":"))
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
    base_shape: {base_shape}
    base_h: {BASE_HEIGHT}
    data_dir_list:
{_yaml_paths(train_lmdbs)}
{common}  sampler:
    name: RatioSampler
    scales: {scales}
    first_bs: {first_batch_size}
    fix_bs: False
    divided_factor: {divided}
    is_training: True
    max_ratio: {max_ratio}
  loader:
    shuffle: True
    batch_size_per_card: {first_batch_size}
    drop_last: True
    max_ratio: {max_ratio}
    num_workers: {num_workers}

Eval:
  dataset:
    name: RatioDataSetTVResize
    ds_width: True
    padding: False
    base_shape: {base_shape}
    base_h: {BASE_HEIGHT}
    data_dir_list:
{_yaml_paths(eval_lmdbs)}
{common}  sampler:
    name: RatioSampler
    scales: {scales}
    first_bs: {first_batch_size}
    fix_bs: False
    divided_factor: {divided}
    is_training: False
    max_ratio: {max_ratio}
  loader:
    shuffle: False
    batch_size_per_card: {first_batch_size}
    drop_last: False
    max_ratio: {max_ratio}
    num_workers: {num_workers}
"""


def msr_gtc_dataset_yaml(
    train_lmdbs: list[Path],
    eval_lmdbs: list[Path],
    first_batch_size: int,
    num_workers: int,
    max_ratio: int,
    sub_str_len: int,
    eval_batch_size: int | None = None,
    control_world_size: int = 1,
    control_first_batch_size: int | None = None,
) -> str:
    """Build the official SMTR/GTC label pipeline on frozen P1/MSR LMDBs."""

    base_shape = json.dumps(BASE_SHAPES, separators=(",", ":"))
    scales = json.dumps(SAMPLER_SCALES, separators=(",", ":"))
    divided = json.dumps(DIVIDED_FACTOR, separators=(",", ":"))
    eval_batch_size = int(
        first_batch_size if eval_batch_size is None else eval_batch_size
    )
    control_first_batch_size = int(
        first_batch_size
        if control_first_batch_size is None
        else control_first_batch_size
    )
    control_lines = ""
    if control_world_size > 1:
        control_lines = (
            f"    control_world_size: {control_world_size}\n"
            f"    control_first_bs: {control_first_batch_size}\n"
        )
    common = f"""    transforms:
      - DecodeImagePIL:
          img_mode: RGB
      - GTCLabelEncode:
          gtc_label_encode:
            name: SMTRLabelEncode
            sub_str_len: {sub_str_len}
          character_dict_path: *character_dict_path
          use_space_char: *use_space_char
          max_text_length: *max_text_length
      - KeepKeys:
          keep_keys: ['image', 'label', 'label_subs', 'label_next', 'length_subs', 'label_subs_pre', 'label_next_pre', 'length_subs_pre', 'length', 'ctc_label', 'ctc_length']
"""
    return f"""Train:
  dataset:
    name: RatioDataSetTVResize
    ds_width: True
    padding: False
    base_shape: {base_shape}
    base_h: {BASE_HEIGHT}
    data_dir_list:
{_yaml_paths(train_lmdbs)}
{common}  sampler:
    name: RatioSampler
    scales: {scales}
    first_bs: {first_batch_size}
{control_lines.rstrip()}
    fix_bs: False
    divided_factor: {divided}
    is_training: True
    max_ratio: {max_ratio}
  loader:
    shuffle: True
    batch_size_per_card: {first_batch_size}
    drop_last: True
    max_ratio: {max_ratio}
    num_workers: {num_workers}

Eval:
  dataset:
    name: RatioDataSetTVResize
    ds_width: True
    padding: False
    base_shape: {base_shape}
    base_h: {BASE_HEIGHT}
    data_dir_list:
{_yaml_paths(eval_lmdbs)}
{common}  sampler:
    name: RatioSampler
    scales: {scales}
    first_bs: {eval_batch_size}
    fix_bs: False
    divided_factor: {divided}
    is_training: False
    max_ratio: {max_ratio}
  loader:
    shuffle: False
    batch_size_per_card: {eval_batch_size}
    drop_last: False
    max_ratio: {max_ratio}
    num_workers: {num_workers}
"""


def make_rctc_config(
    *,
    root: Path,
    run_dir: Path,
    project_name: str,
    train_lmdbs: list[Path],
    eval_lmdbs: list[Path],
    pretrained_model: Path | None,
    max_epoch: int,
    first_batch_size: int,
    num_workers: int,
    max_ratio: int,
    lr: float,
    internal_eval_every: int,
    seed: int,
) -> str:
    character_dict = (
        root
        / "04_model_training"
        / "character_dict_hz_ug_kk_v1"
        / "character_dict.txt"
    )
    pretrained_text = str(pretrained_model) if pretrained_model else ""
    config = f"""Global:
  device: gpu
  epoch_num: {max_epoch}
  log_smooth_window: 20
  print_batch_step: 20
  output_dir: {run_dir}
  save_epoch_step: [0, 1]
  eval_epoch_step: [0, {internal_eval_every}]
  eval_batch_step: [0, 100000000]
  cal_metric_during_train: True
  pretrained_model: {pretrained_text}
  checkpoints:
  use_tensorboard: false
  infer_img:
  character_dict_path: &character_dict_path {character_dict}
  max_text_length: &max_text_length {MAX_TEXT_LENGTH}
  use_space_char: &use_space_char True
  save_res_path: {run_dir / "predicts.txt"}
  use_amp: False
  grad_clip_val: 5.0
  project_name: {project_name}
  seed: {seed}
  preprocess_protocol: {PROTOCOL_ID}
  sampler_protocol: {SAMPLER_PROTOCOL_ID}

Optimizer:
  name: AdamW
  lr: {lr:.8f}
  weight_decay: 0.01
  filter_bias_and_bn: True

LRScheduler:
  name: OneCycleLR
  warmup_epoch: 2
  cycle_momentum: False

Architecture:
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
    config += msr_dataset_yaml(
        train_lmdbs,
        eval_lmdbs,
        first_batch_size,
        num_workers,
        max_ratio,
    )
    assert_p1_config_text(config)
    return config


def make_full_svtrv2_s_config(
    *,
    root: Path,
    run_dir: Path,
    project_name: str,
    train_lmdbs: list[Path],
    eval_lmdbs: list[Path],
    pretrained_model: Path,
    max_epoch: int,
    first_batch_size: int,
    num_workers: int,
    max_ratio: int,
    lr: float,
    internal_eval_every: int,
    seed: int,
    ctc_weight: float = 0.1,
    sub_str_len: int = 5,
    eval_batch_size: int | None = None,
    control_world_size: int = 1,
    control_first_batch_size: int | None = None,
) -> str:
    """Create the SVTRv2-S Stage-2 config with standard SMTR-based SGM.

    The encoder remains exactly SVTRv2-S. Both the RCTC branch and SMTR
    guidance branch consume the same LMDB label, which is U2 for Uyghur.
    """

    character_dict = (
        root
        / "04_model_training"
        / "character_dict_hz_ug_kk_v1"
        / "character_dict.txt"
    )
    config = f"""Global:
  device: gpu
  epoch_num: {max_epoch}
  log_smooth_window: 20
  print_batch_step: 20
  output_dir: {run_dir}
  save_epoch_step: [0, 1]
  eval_epoch_step: [0, {internal_eval_every}]
  eval_batch_step: [0, 100000000]
  cal_metric_during_train: True
  pretrained_model: {pretrained_model}
  checkpoints:
  use_tensorboard: false
  infer_img:
  character_dict_path: &character_dict_path {character_dict}
  max_text_length: &max_text_length {MAX_TEXT_LENGTH}
  use_space_char: &use_space_char True
  save_res_path: {run_dir / "predicts.txt"}
  use_amp: False
  grad_clip_val: 5.0
  project_name: {project_name}
  seed: {seed}
  preprocess_protocol: {PROTOCOL_ID}
  sampler_protocol: {SAMPLER_PROTOCOL_ID}
  model_protocol: B1_FULL_SVTRV2_S_SGM_U2_V1
  evaluation_branch: ctc

Optimizer:
  name: AdamW
  lr: {lr:.8f}
  weight_decay: 0.01
  filter_bias_and_bn: True

LRScheduler:
  name: OneCycleLR
  warmup_epoch: 2
  cycle_momentum: False

Architecture:
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
    name: GTCDecoder
    infer_gtc: True
    detach: False
    gtc_decoder:
      name: SMTRDecoder
      num_layer: 1
      ds: True
      max_len: *max_text_length
      next_mode: &next_mode True
      sub_str_len: &sub_str_len {sub_str_len}
    ctc_decoder:
      name: RCTCDecoder

Loss:
  name: GTCLoss
  ctc_weight: {ctc_weight}
  gtc_weight: 1.0
  zero_infinity: True
  gtc_loss:
    name: SMTRLoss

PostProcess:
  name: GTCLabelDecode
  gtc_label_decode:
    name: SMTRLabelDecode
    next_mode: *next_mode
  character_dict_path: *character_dict_path
  use_space_char: *use_space_char

Metric:
  name: RecGTCMetric
  main_indicator: acc
  ignore_space: False
  is_filter: False

"""
    config += msr_gtc_dataset_yaml(
        train_lmdbs,
        eval_lmdbs,
        first_batch_size,
        num_workers,
        max_ratio,
        sub_str_len,
        eval_batch_size,
        control_world_size,
        control_first_batch_size,
    )
    assert_p1_config_text(config)
    return config


def assert_p1_config_text(config: str) -> None:
    required = [
        f"preprocess_protocol: {PROTOCOL_ID}",
        f"sampler_protocol: {SAMPLER_PROTOCOL_ID}",
        "name: RatioDataSetTVResize",
        "name: RatioSampler",
        "padding: False",
        f"base_h: {BASE_HEIGHT}",
        f"max_ratio: {DEFAULT_MAX_RATIO}",
        f"max_text_length: &max_text_length {MAX_TEXT_LENGTH}",
    ]
    missing = [token for token in required if token not in config]
    if missing:
        raise ValueError(f"P1/MSR config is missing required tokens: {missing}")
    forbidden = ["RecTVResize:", "image_shape: [48, 640]", "padding: True"]
    present = [token for token in forbidden if token in config]
    if present:
        raise ValueError(f"P0/fixed-resize tokens found in P1 config: {present}")
    if config.count("name: RatioDataSetTVResize") != 2:
        raise ValueError("P1 config must use RatioDataSetTVResize for Train and Eval")
    if config.count("name: RatioSampler") != 2:
        raise ValueError("P1 config must use RatioSampler for Train and Eval")


def preprocessing_summary(
    train_lmdbs: list[Path],
    eval_lmdbs: list[Path],
    first_batch_size: int,
    max_ratio: int,
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "dataset": "RatioDataSetTVResize",
        "sampler": "RatioSampler",
        "sampler_protocol": SAMPLER_PROTOCOL_ID,
        "base_shapes": BASE_SHAPES,
        "base_height": BASE_HEIGHT,
        "sampler_scales": SAMPLER_SCALES,
        "divided_factor": DIVIDED_FACTOR,
        "padding": False,
        "dynamic_width": True,
        "aspect_ratio_bucketing": True,
        "external_eval_batch_size": EVAL_INFERENCE_BATCH_SIZE,
        "max_ratio": max_ratio,
        "max_text_length": MAX_TEXT_LENGTH,
        "first_batch_size_per_card": first_batch_size,
        "train_lmdbs": [str(path) for path in train_lmdbs],
        "eval_lmdbs": [str(path) for path in eval_lmdbs],
    }
