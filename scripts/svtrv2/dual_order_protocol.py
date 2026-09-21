#!/usr/bin/env python3
"""Frozen label and configuration protocol for dual-order SVTRv2 methods."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import unicodedata
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

from p1_msr_protocol import (
    BASE_HEIGHT,
    BASE_SHAPES,
    DEFAULT_MAX_RATIO,
    DIVIDED_FACTOR,
    MAX_TEXT_LENGTH,
    PROTOCOL_ID,
    SAMPLER_PROTOCOL_ID,
    SAMPLER_SCALES,
    SCALE_COUNTS,
    LANGUAGES,
    make_full_svtrv2_s_config,
    synthetic_subset_records,
    target_records,
)


DUAL_ORDER_LABEL_VERSION = "DUAL_ORDER_LABEL_V2"
DUAL_ORDER_DATA_PROTOCOL = f"{PROTOCOL_ID}_DUAL_ORDER_V2"
METHOD_PROTOCOL = "MULTISCRIPT_DUAL_ORDER_SGM_V1"
SCRIPT_IDS = {"zh": 0, "ug": 1, "kk": 2}

# These characters alter shaping or layout but have no visible OCR glyph. The
# frozen target metadata contains four ZWJ/ZWNJ occurrences in three train
# labels. python-bidi removes them during visual reordering, so retaining them
# in logical supervision makes an exact logical/visual permutation impossible.
# Keep the source metadata immutable and normalize only model supervision.
IGNORABLE_OCR_FORMAT_CONTROLS = frozenset(
    {
        "\u200b",  # ZERO WIDTH SPACE
        "\u200c",  # ZERO WIDTH NON-JOINER
        "\u200d",  # ZERO WIDTH JOINER
        "\u2060",  # WORD JOINER
        "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
    }
)

METHODS = {
    "b2": {
        "ctc_order": "logical",
        "sgm_order": "logical",
        "consistency_weight": 0.0,
        "script_weight": 0.0,
        "direction_weight": 0.0,
        "use_script_adapter": False,
        "use_local_direction": False,
        "description": "logical/logical direction-conflict diagnostic",
    },
    "m1": {
        "ctc_order": "visual",
        "sgm_order": "logical",
        "consistency_weight": 0.0,
        "script_weight": 0.0,
        "direction_weight": 0.0,
        "use_script_adapter": False,
        "use_local_direction": False,
        "description": "dual-order semantic guidance",
    },
    "m2": {
        "ctc_order": "visual",
        "sgm_order": "logical",
        "consistency_weight": 0.15,
        "script_weight": 0.0,
        "direction_weight": 0.0,
        "use_script_adapter": False,
        "use_local_direction": False,
        "description": "M1 plus cross-order posterior transport",
    },
    "m3": {
        "ctc_order": "visual",
        "sgm_order": "logical",
        "consistency_weight": 0.15,
        "script_weight": 0.10,
        "direction_weight": 0.0,
        "use_script_adapter": True,
        "use_local_direction": False,
        "description": "M2 plus inferred script-conditioned residual experts",
    },
    "m3_nococ": {
        "ctc_order": "visual",
        "sgm_order": "logical",
        "consistency_weight": 0.0,
        "script_weight": 0.10,
        "direction_weight": 0.0,
        "use_script_adapter": True,
        "use_local_direction": False,
        "description": (
            "M3 architecture with cross-order consistency removed for a "
            "controlled COC contribution audit"
        ),
    },
    "m3_sldr_pilot": {
        "ctc_order": "visual",
        "sgm_order": "logical",
        "consistency_weight": 0.15,
        "script_weight": 0.10,
        "direction_weight": 0.0,
        "use_script_adapter": True,
        "use_local_direction": False,
        "use_sldr": True,
        "description": (
            "Exploratory target-only M3 pilot with script-aware local detail "
            "refinement before RCTC feature rearrangement"
        ),
    },
    "m3_scdl_pilot": {
        "ctc_order": "visual",
        "sgm_order": "logical",
        "consistency_weight": 0.15,
        "script_weight": 0.10,
        "direction_weight": 0.0,
        "use_script_adapter": True,
        "use_local_direction": False,
        "use_scdl": True,
        "scdl_weight": 0.05,
        "scdl_topk": 5,
        "scdl_temperature": 0.1,
        "scdl_warmup_fraction": 0.2,
        "description": (
            "Exploratory target-only M3 pilot with script-aware, exact-CTC-"
            "aligned hard-confusion discriminative learning"
        ),
    },
    "m3_sldr_full_v1": {
        "ctc_order": "visual",
        "sgm_order": "logical",
        "consistency_weight": 0.15,
        "script_weight": 0.10,
        "direction_weight": 0.0,
        "use_script_adapter": True,
        "use_local_direction": False,
        "use_sldr": True,
        "description": (
            "Full S50-to-target M3 training with script-aware local detail "
            "refinement active in both stages"
        ),
    },
    "m3_scdl_full_v1": {
        "ctc_order": "visual",
        "sgm_order": "logical",
        "consistency_weight": 0.15,
        "script_weight": 0.10,
        "direction_weight": 0.0,
        "use_script_adapter": True,
        "use_local_direction": False,
        "use_scdl": True,
        "scdl_weight": 0.05,
        "scdl_topk": 5,
        "scdl_temperature": 0.1,
        "scdl_warmup_fraction": 0.2,
        "description": (
            "Full S50-to-target M3 training with training-only script-aware, "
            "exact-CTC-aligned hard-confusion discriminative learning"
        ),
    },
    "full": {
        "ctc_order": "visual",
        "sgm_order": "logical",
        "consistency_weight": 0.15,
        "script_weight": 0.10,
        "direction_weight": 0.05,
        "use_script_adapter": True,
        "use_local_direction": True,
        "description": "M3 plus predicted local direction field",
    },
}


for _tag, _alpha in (("020", 0.20), ("025", 0.25), ("030", 0.30)):
    METHODS[f"m3_alpha{_tag}"] = {
        **METHODS["m3"],
        "consistency_weight": _alpha,
        "description": f"M3 consistency verification at alpha={_alpha:.2f}",
    }

for _variant, _mode in (("dir_d1", "sgm_only"), ("dir_d2", "script_gated")):
    METHODS[_variant] = {
        **METHODS["m3_alpha030"],
        "semantic_direction": _mode,
        "direction_gate_initial": 0.01,
        "direction_weight": 0.05,
        "description": f"M3 plus training-only semantic direction ({_mode})",
    }


METHODS["dir_d1_v2"] = {
    **METHODS["m3_alpha030"],
    "semantic_direction": "sgm_only",
    "description": "M3 alpha=.30 plus SGM-only direction, shared S50 initialization, no new loss",
}
METHODS["dir_d2_v2"] = {
    **METHODS["m3_alpha030"],
    "semantic_direction": "sample_script_gated",
    "direction_gate_init_logit": -4.0,
    "description": "M3 alpha=.30 plus sample-level script-gated SGM direction, shared S50 initialization, no new loss",
}


def resolve_method_spec(
    method: str,
    *,
    consistency_weight: float | None = None,
) -> dict[str, Any]:
    """Return an isolated method spec with an optional M2 alpha override."""

    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    spec = dict(METHODS[method])
    if consistency_weight is None:
        return spec
    if method != "m2":
        raise ValueError(
            "A consistency-weight override is only valid for method m2"
        )
    weight = float(consistency_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"M2 consistency weight must be in [0, 1], got {weight}")
    spec["consistency_weight"] = weight
    return spec


def consistency_weight_tag(weight: float) -> str:
    """Encode a reproducible two-decimal alpha for paths and protocol IDs."""

    value = float(weight)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Consistency weight must be in [0, 1], got {value}")
    return f"{value:.2f}".replace(".", "_")


def consistency_weight_path_tag(weight: float) -> str:
    """Return the compact alpha tag used by experiment directories."""

    return consistency_weight_tag(weight).replace("_", "")


def _python_bidi():
    try:
        from bidi import algorithm
    except Exception as exc:
        raise RuntimeError(
            "Dual-order preparation requires python-bidi in openocr_svtrv2"
        ) from exc
    return algorithm


def normalize_logical_label(text: str) -> tuple[str, list[str]]:
    """Remove non-glyph format controls while rejecting unknown controls."""

    removed = [char for char in text if char in IGNORABLE_OCR_FORMAT_CONTROLS]
    normalized = "".join(
        char for char in text if char not in IGNORABLE_OCR_FORMAT_CONTROLS
    )
    unsupported = sorted(
        {char for char in normalized if unicodedata.category(char) == "Cf"},
        key=ord,
    )
    if unsupported:
        codepoints = ", ".join(f"U+{ord(char):04X}" for char in unsupported)
        raise ValueError(
            "Logical OCR label contains unsupported Unicode format controls: "
            f"{codepoints}"
        )
    if not normalized:
        raise ValueError("Logical OCR label became empty after normalization")
    return normalized, removed


def bidi_layout(logical_text: str, language: str) -> dict[str, Any]:
    """Return visual text and exact logical/visual index transport."""

    length = len(logical_text)
    if language != "ug":
        return {
            "visual_text": logical_text,
            "logical_to_visual": list(range(length)),
            "visual_to_logical": list(range(length)),
            "visual_directions": [0] * length,
        }

    algorithm = _python_bidi()
    storage = algorithm.get_empty_storage()
    storage["base_level"] = 1
    storage["base_dir"] = "R"
    algorithm.get_embedding_levels(logical_text, storage, False, False)
    for source_index, char_info in enumerate(storage["chars"]):
        char_info["source_index"] = source_index
    algorithm.explicit_embed_and_overrides(storage, False)
    algorithm.resolve_weak_types(storage, False)
    algorithm.resolve_neutral_types(storage, False)
    algorithm.resolve_implicit_levels(storage, False)
    algorithm.reorder_resolved_levels(storage, False)
    algorithm.apply_mirroring(storage, False)

    visual_text = "".join(item["ch"] for item in storage["chars"])
    expected = algorithm.get_display(logical_text, base_dir="R")
    if visual_text != expected:
        raise AssertionError("Stored bidi permutation disagrees with get_display")
    visual_to_logical = [item["source_index"] for item in storage["chars"]]
    if sorted(visual_to_logical) != list(range(length)):
        raise ValueError("python-bidi did not preserve a one-to-one character map")
    logical_to_visual = [-1] * length
    for visual_index, logical_index in enumerate(visual_to_logical):
        logical_to_visual[logical_index] = visual_index
    directions = [int(item["level"] % 2) for item in storage["chars"]]
    return {
        "visual_text": visual_text,
        "logical_to_visual": logical_to_visual,
        "visual_to_logical": visual_to_logical,
        "visual_directions": directions,
    }


def dual_order_payload(record: dict[str, Any]) -> dict[str, Any]:
    logical, removed_controls = normalize_logical_label(record["logical_text"])
    language = record["language"]
    layout = bidi_layout(logical, language)
    expected_visual, _ = normalize_logical_label(record["ctc_text"])
    if layout["visual_text"] != expected_visual:
        raise ValueError(
            f"U2 mismatch for {record['id']}: recomputed visual label differs"
        )
    return {
        "version": DUAL_ORDER_LABEL_VERSION,
        "language": language,
        "logical_text": logical,
        "visual_text": layout["visual_text"],
        "logical_to_visual": layout["logical_to_visual"],
        "visual_directions": layout["visual_directions"],
        "removed_format_controls": [
            f"U+{ord(char):04X}" for char in removed_controls
        ],
    }


def _payload_text(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _records_fingerprint(
    records: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
) -> str:
    if len(records) != len(payloads):
        raise ValueError("Record and dual-order payload counts differ")
    digest = hashlib.sha256()
    for record, payload in zip(records, payloads):
        item = {
            "id": record["id"],
            "language": record["language"],
            "image": record["image"],
            "payload": payload,
        }
        digest.update(
            json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def ensure_dual_order_lmdb(
    records: list[dict[str, Any]],
    output_dir: Path,
    source_name: str,
    replace: bool = False,
) -> dict[str, Any]:
    """Create an atomic P1/MSR LMDB containing dual-order JSON labels."""

    counts = dict(Counter(row["language"] for row in records))
    normalized_records = [dual_order_payload(row) for row in records]
    removed_format_controls = Counter(
        codepoint
        for payload in normalized_records
        for codepoint in payload["removed_format_controls"]
    )
    normalized_record_ids = [
        record["id"]
        for record, payload in zip(records, normalized_records)
        if payload["removed_format_controls"]
    ]
    expected = {
        "preprocess_protocol": PROTOCOL_ID,
        "sampler_protocol": SAMPLER_PROTOCOL_ID,
        "dual_order_data_protocol": DUAL_ORDER_DATA_PROTOCOL,
        "dual_order_label_version": DUAL_ORDER_LABEL_VERSION,
        "python_bidi_version": version("python-bidi"),
        "source_name": source_name,
        "record_fingerprint_sha256": _records_fingerprint(
            records, normalized_records
        ),
        "rows": len(records),
        "language_counts": counts,
        "label_normalization": {
            "policy": "strip_non_glyph_unicode_format_controls_v1",
            "allowed_removed_codepoints": sorted(
                f"U+{ord(char):04X}"
                for char in IGNORABLE_OCR_FORMAT_CONTROLS
            ),
            "removed_occurrences": dict(sorted(removed_format_controls.items())),
            "normalized_record_ids": normalized_record_ids,
        },
    }
    manifest_path = output_dir / "dual_order_lmdb_manifest.json"
    if output_dir.exists() and not replace:
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Unverified dual-order LMDB exists: {output_dir}"
            )
        current = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        for key, value in expected.items():
            if current.get(key) != value:
                raise RuntimeError(
                    f"Stale dual-order LMDB {output_dir}: {key} changed. "
                    "Use --replace-dual-order-lmdb."
                )
        return current

    try:
        import lmdb
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Dual-order P1/MSR requires lmdb and Pillow") from exc

    temporary = output_dir.with_name(output_dir.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    temporary.mkdir(parents=True, exist_ok=True)

    total_bytes = sum(row["image_path"].stat().st_size for row in records)
    map_size = max(1 << 30, total_bytes * 4 + (256 << 20))
    env = lmdb.open(str(temporary), map_size=map_size)
    cache: dict[bytes, bytes] = {}

    def flush() -> None:
        if not cache:
            return
        with env.begin(write=True) as transaction:
            for key, value in cache.items():
                transaction.put(key, value)
        cache.clear()

    for index, (record, payload) in enumerate(
        zip(records, normalized_records), 1
    ):
        image_bytes = record["image_path"].read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
        cache[f"image-{index:09d}".encode()] = image_bytes
        cache[f"label-{index:09d}".encode()] = _payload_text(payload).encode(
            "utf-8"
        )
        cache[f"wh-{index:09d}".encode()] = (
            f"{width}_{height}".encode("ascii")
        )
        if index % 1000 == 0:
            flush()
        if index % 10000 == 0:
            print(
                json.dumps(
                    {
                        "dual_order_lmdb": str(output_dir),
                        "written": index,
                        "rows": len(records),
                    }
                ),
                flush=True,
            )
    cache[b"num-samples"] = str(len(records)).encode("ascii")
    flush()
    env.sync()
    env.close()

    manifest = {
        **expected,
        "path": str(output_dir),
        "map_size": map_size,
        "image_bytes": total_bytes,
    }
    (temporary / manifest_path.name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.rename(output_dir)
    return manifest


def ensure_dual_target_lmdbs(
    root: Path,
    target_root: Path,
    metadata_rows: list[dict[str, Any]],
    replace: bool = False,
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    cache_root = (
        root / "04_model_training" / "msr_lmdb" / DUAL_ORDER_DATA_PROTOCOL
    )
    train = target_records(target_root, metadata_rows, "train")
    dev = target_records(target_root, metadata_rows, "dev")
    train_path = cache_root / "target_train"
    dev_path = cache_root / "target_dev"
    manifests = {
        "train": ensure_dual_order_lmdb(
            train, train_path, "target_train", replace
        ),
        "dev": ensure_dual_order_lmdb(dev, dev_path, "target_dev", replace),
    }
    return [train_path], [dev_path], manifests


def ensure_dual_synthetic_lmdbs(
    root: Path,
    formal_root: Path,
    scale: str,
    replace: bool = False,
) -> tuple[list[Path], dict[str, Any]]:
    if scale not in SCALE_COUNTS:
        raise ValueError(f"Unsupported scale: {scale}")
    cache_root = (
        root
        / "04_model_training"
        / "msr_lmdb"
        / DUAL_ORDER_DATA_PROTOCOL
        / formal_root.name
    )
    paths = []
    manifests = {}
    previous_ids: set[str] = set()
    for current in ("s10", "s25", "s50"):
        records = synthetic_subset_records(formal_root, current)
        current_ids = {row["id"] for row in records}
        if previous_ids and not previous_ids.issubset(current_ids):
            raise ValueError(f"Nested subset invariant failed at {current}")
        increment = [row for row in records if row["id"] not in previous_ids]
        suffix = current if current == "s10" else f"{current}_increment"
        output = cache_root / suffix
        manifests[current] = ensure_dual_order_lmdb(
            increment,
            output,
            f"{formal_root.name}_{suffix}",
            replace,
        )
        paths.append(output)
        previous_ids = current_ids
        if current == scale:
            break
    expected = SCALE_COUNTS[scale] * len(LANGUAGES)
    actual = sum(int(item["rows"]) for item in manifests.values())
    if actual != expected:
        raise ValueError(
            f"Dual-order {scale} LMDB total {actual}, expected {expected}"
        )
    return paths, manifests


def _dual_transform_block(ctc_order: str, sgm_order: str) -> str:
    return f"""      - DualOrderGTCLabelEncode:
          ctc_order: {ctc_order}
          sgm_order: {sgm_order}
          gtc_label_encode:
            name: SMTRLabelEncode
            sub_str_len: 5
          character_dict_path: *character_dict_path
          use_space_char: *use_space_char
          max_text_length: *max_text_length
      - KeepKeys:
          keep_keys: ['image', 'label', 'label_subs', 'label_next', 'length_subs', 'label_subs_pre', 'label_next_pre', 'length_subs_pre', 'length', 'visual_token_ids', 'logical_token_ids', 'logical_to_visual', 'visual_direction_ids', 'visual_script_ids', 'consistency_mask', 'ctc_label', 'ctc_length']
"""


def make_method_config(
    *,
    method: str,
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
    consistency_weight: float | None = None,
    eval_batch_size: int | None = None,
    control_world_size: int = 1,
    control_first_batch_size: int | None = None,
) -> str:
    spec = resolve_method_spec(
        method,
        consistency_weight=consistency_weight,
    )
    protocol_suffix = method.upper()
    if consistency_weight is not None:
        protocol_suffix = (
            f"M2_ALPHA_{consistency_weight_tag(consistency_weight)}"
        )
    config = make_full_svtrv2_s_config(
        root=root,
        run_dir=run_dir,
        project_name=project_name,
        train_lmdbs=train_lmdbs,
        eval_lmdbs=eval_lmdbs,
        pretrained_model=pretrained_model,
        max_epoch=max_epoch,
        first_batch_size=first_batch_size,
        num_workers=num_workers,
        max_ratio=max_ratio,
        lr=lr,
        internal_eval_every=internal_eval_every,
        seed=seed,
        eval_batch_size=eval_batch_size,
        control_world_size=control_world_size,
        control_first_batch_size=control_first_batch_size,
    )

    old_transform = """      - GTCLabelEncode:
          gtc_label_encode:
            name: SMTRLabelEncode
            sub_str_len: 5
          character_dict_path: *character_dict_path
          use_space_char: *use_space_char
          max_text_length: *max_text_length
      - KeepKeys:
          keep_keys: ['image', 'label', 'label_subs', 'label_next', 'length_subs', 'label_subs_pre', 'label_next_pre', 'length_subs_pre', 'length', 'ctc_label', 'ctc_length']
"""
    expected_occurrences = 2
    if config.count(old_transform) != expected_occurrences:
        raise AssertionError("B1 transform template changed upstream")
    config = config.replace(
        old_transform,
        _dual_transform_block(spec["ctc_order"], spec["sgm_order"]),
    )
    config = config.replace(
        "  model_protocol: B1_FULL_SVTRV2_S_SGM_U2_V1\n",
        (
            f"  model_protocol: {METHOD_PROTOCOL}_{protocol_suffix}\n"
            f"  dual_order_data_protocol: {DUAL_ORDER_DATA_PROTOCOL}\n"
            f"  method_variant: {method}\n"
            f"  uyghur_ctc_order: {spec['ctc_order']}\n"
        ),
        1,
    )
    config = config.replace(
        "    name: GTCDecoder\n"
        "    infer_gtc: True\n"
        "    detach: False\n",
        (
            "    name: DualOrderGTCDecoder\n"
            "    infer_gtc: True\n"
            "    detach: False\n"
            f"    use_script_adapter: {str(spec['use_script_adapter'])}\n"
            f"    use_local_direction: {str(spec['use_local_direction'])}\n"
            "    script_bottleneck_ratio: 4\n"
        ),
        1,
    )
    if spec.get("use_sldr", False):
        config = config.replace(
            "    script_bottleneck_ratio: 4\n",
            "    script_bottleneck_ratio: 4\n"
            "    use_sldr: True\n"
            "    sldr_reduction: 4\n"
            "    sldr_gamma_init: 0.001\n",
            1,
        )
    if spec.get("use_scdl", False):
        dictionary = (
            root
            / "04_model_training"
            / "character_dict_hz_ug_kk_v1"
            / "character_dict.txt"
        )
        config = config.replace(
            "    script_bottleneck_ratio: 4\n",
            "    script_bottleneck_ratio: 4\n"
            "    use_scdl: True\n"
            f"    scdl_character_dict_path: {dictionary}\n"
            "    scdl_use_space_char: True\n",
            1,
        )
    config = config.replace(
        "  name: GTCLoss\n"
        "  ctc_weight: 0.1\n"
        "  gtc_weight: 1.0\n",
        (
            "  name: DualOrderGTCLoss\n"
            "  ctc_weight: 0.1\n"
            "  gtc_weight: 1.0\n"
            f"  consistency_weight: {spec['consistency_weight']}\n"
            f"  script_weight: {spec['script_weight']}\n"
            f"  direction_weight: {spec['direction_weight']}\n"
            "  consistency_temperature: 1.0\n"
            "  alignment_prior_scale: 0.75\n"
        ),
        1,
    )
    if spec.get("use_scdl", False):
        config = config.replace(
            "  alignment_prior_scale: 0.75\n",
            "  alignment_prior_scale: 0.75\n"
            f"  scdl_weight: {spec['scdl_weight']}\n"
            f"  scdl_topk: {spec['scdl_topk']}\n"
            f"  scdl_temperature: {spec['scdl_temperature']}\n"
            f"  scdl_warmup_fraction: {spec['scdl_warmup_fraction']}\n",
            1,
        )
    if method in ("dir_d1", "dir_d2"):
        config = config.replace(
            "    script_bottleneck_ratio: 4\n",
            "    script_bottleneck_ratio: 4\n"
            f"    semantic_direction: {spec['semantic_direction']}\n"
            f"    direction_gate_initial: {spec['direction_gate_initial']}\n",
            1,
        )
    if method in ("dir_d1_v2", "dir_d2_v2"):
        gate_config = ""
        if method == "dir_d2_v2":
            gate_config = "    direction_gate_init_logit: -4.0\n"
        config = config.replace(
            "    script_bottleneck_ratio: 4\n",
            "    script_bottleneck_ratio: 4\n"
            f"    semantic_direction: {spec['semantic_direction']}\n"
            f"{gate_config}",
            1,
        )
    required = (
        "- DualOrderGTCLabelEncode:",
        "name: DualOrderGTCDecoder",
        "name: DualOrderGTCLoss",
        f"method_variant: {method}",
        f"uyghur_ctc_order: {spec['ctc_order']}",
    )
    missing = [token for token in required if token not in config]
    if missing:
        raise AssertionError(f"Method config is incomplete: {missing}")
    return config


def method_summary(
    method: str,
    *,
    consistency_weight: float | None = None,
) -> dict[str, Any]:
    spec = resolve_method_spec(
        method,
        consistency_weight=consistency_weight,
    )
    protocol_variant = f"{METHOD_PROTOCOL}_{method.upper()}"
    if consistency_weight is not None:
        protocol_variant = (
            f"{METHOD_PROTOCOL}_M2_ALPHA_"
            f"{consistency_weight_tag(consistency_weight)}"
        )
    return {
        "method": method,
        "method_protocol": METHOD_PROTOCOL,
        "method_protocol_variant": protocol_variant,
        "dual_order_data_protocol": DUAL_ORDER_DATA_PROTOCOL,
        "preprocess_protocol": PROTOCOL_ID,
        "sampler_protocol": SAMPLER_PROTOCOL_ID,
        "ctc_supervision": spec["ctc_order"],
        "sgm_supervision": spec["sgm_order"],
        "cross_order_consistency": spec["consistency_weight"] > 0,
        "script_conditioned_adapter": spec["use_script_adapter"],
        "script_aware_local_detail_refinement": spec.get("use_sldr", False),
        "script_aware_confusion_discrimination": spec.get("use_scdl", False),
        "local_direction_conditioning": spec["use_local_direction"],
        "semantic_direction": spec.get("semantic_direction", "none"),
        "loss_weights": {
            "ctc": 0.1,
            "sgm": 1.0,
            "consistency": spec["consistency_weight"],
            "script": spec["script_weight"],
            "direction": spec["direction_weight"],
            "scdl": spec.get("scdl_weight", 0.0),
        },
        "description": spec["description"],
        "p1_msr": {
            "base_shapes": BASE_SHAPES,
            "base_height": BASE_HEIGHT,
            "sampler_scales": SAMPLER_SCALES,
            "divided_factor": DIVIDED_FACTOR,
            "max_ratio": DEFAULT_MAX_RATIO,
            "max_text_length": MAX_TEXT_LENGTH,
        },
    }
