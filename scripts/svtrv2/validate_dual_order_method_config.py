#!/usr/bin/env python3
"""Fail-fast validation for B2/M1/M2/M3/Full P1/MSR configs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from dual_order_protocol import (
    DUAL_ORDER_DATA_PROTOCOL,
    DUAL_ORDER_LABEL_VERSION,
    METHODS,
    METHOD_PROTOCOL,
)
from p1_msr_protocol import (
    BASE_HEIGHT,
    BASE_SHAPES,
    DEFAULT_MAX_RATIO,
    DIVIDED_FACTOR,
    MAX_TEXT_LENGTH,
    PROTOCOL_ID,
    SAMPLER_PROTOCOL_ID,
    SAMPLER_SCALES,
    assert_p1_config_text,
)
from validate_p1_msr_config import require_equal, sha256, validate_initialization


EXPECTED_KEEP_KEYS = [
    "image",
    "label",
    "label_subs",
    "label_next",
    "length_subs",
    "label_subs_pre",
    "label_next_pre",
    "length_subs_pre",
    "length",
    "visual_token_ids",
    "logical_token_ids",
    "logical_to_visual",
    "visual_direction_ids",
    "visual_script_ids",
    "consistency_mask",
    "ctc_label",
    "ctc_length",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--expected-initialization", choices=("checkpoint",), default="checkpoint"
    )
    parser.add_argument("--allow-missing-lmdb", action="store_true")
    return parser.parse_args()


def transform_map(cfg, section):
    result = {}
    for item in cfg[section]["dataset"].get("transforms") or []:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"Malformed transform in {section}: {item!r}")
        name, value = next(iter(item.items()))
        result[name] = value or {}
    return result


def validate_dataset(cfg, section, allow_missing):
    dataset = cfg[section]["dataset"]
    sampler = cfg[section]["sampler"]
    loader = cfg[section]["loader"]
    require_equal(dataset.get("name"), "RatioDataSetTVResize", "dataset.name")
    require_equal(dataset.get("base_shape"), BASE_SHAPES, "dataset.base_shape")
    require_equal(dataset.get("base_h"), BASE_HEIGHT, "dataset.base_h")
    require_equal(dataset.get("padding"), False, "dataset.padding")
    require_equal(sampler.get("name"), "RatioSampler", "sampler.name")
    require_equal(sampler.get("scales"), SAMPLER_SCALES, "sampler.scales")
    require_equal(
        sampler.get("divided_factor"), DIVIDED_FACTOR, "sampler.divided_factor"
    )
    require_equal(sampler.get("max_ratio"), DEFAULT_MAX_RATIO, "max_ratio")
    require_equal(loader.get("max_ratio"), DEFAULT_MAX_RATIO, "loader.max_ratio")

    manifests = []
    for raw_path in dataset.get("data_dir_list") or []:
        lmdb_path = Path(raw_path)
        manifest_path = lmdb_path / "dual_order_lmdb_manifest.json"
        if not manifest_path.is_file():
            if allow_missing:
                manifests.append(
                    {"path": str(lmdb_path), "status": "missing_allowed"}
                )
                continue
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        require_equal(
            manifest.get("preprocess_protocol"),
            PROTOCOL_ID,
            "manifest.preprocess_protocol",
        )
        require_equal(
            manifest.get("sampler_protocol"),
            SAMPLER_PROTOCOL_ID,
            "manifest.sampler_protocol",
        )
        require_equal(
            manifest.get("dual_order_data_protocol"),
            DUAL_ORDER_DATA_PROTOCOL,
            "manifest.dual_order_data_protocol",
        )
        require_equal(
            manifest.get("dual_order_label_version"),
            DUAL_ORDER_LABEL_VERSION,
            "manifest.dual_order_label_version",
        )
        if not (lmdb_path / "data.mdb").is_file():
            raise FileNotFoundError(lmdb_path / "data.mdb")
        try:
            import lmdb
        except Exception as exc:
            raise RuntimeError("Install lmdb in openocr_svtrv2") from exc
        environment = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        try:
            with environment.begin(write=False) as transaction:
                payload = json.loads(
                    transaction.get(b"label-000000001").decode("utf-8")
                )
                require_equal(
                    payload.get("version"),
                    DUAL_ORDER_LABEL_VERSION,
                    "first_label.version",
                )
                rows = int(transaction.get(b"num-samples").decode("ascii"))
                require_equal(rows, manifest.get("rows"), "LMDB row count")
        finally:
            environment.close()
        manifests.append(manifest)
    return manifests


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    text = config_path.read_text(encoding="utf-8-sig")
    assert_p1_config_text(text)
    cfg = yaml.safe_load(text)
    global_cfg = cfg["Global"]
    method = global_cfg.get("method_variant")
    if method not in METHODS:
        raise ValueError(f"Unknown method_variant: {method}")
    spec = METHODS[method]
    expected_spec = dict(spec)
    model_protocol = str(global_cfg.get("model_protocol", ""))
    alpha_match = re.fullmatch(
        rf"{re.escape(METHOD_PROTOCOL)}_M2_ALPHA_(\d+)_(\d+)",
        model_protocol,
    )
    if method == "m2" and alpha_match:
        integer, fractional = alpha_match.groups()
        consistency_weight = float(f"{integer}.{fractional}")
        if not 0.0 <= consistency_weight <= 1.0:
            raise ValueError(
                f"M2 alpha must be in [0, 1], got {consistency_weight}"
            )
        expected_spec["consistency_weight"] = consistency_weight
    require_equal(
        global_cfg.get("preprocess_protocol"),
        PROTOCOL_ID,
        "Global.preprocess_protocol",
    )
    require_equal(
        global_cfg.get("sampler_protocol"),
        SAMPLER_PROTOCOL_ID,
        "Global.sampler_protocol",
    )
    require_equal(
        global_cfg.get("dual_order_data_protocol"),
        DUAL_ORDER_DATA_PROTOCOL,
        "Global.dual_order_data_protocol",
    )
    require_equal(
        global_cfg.get("uyghur_ctc_order"),
        spec["ctc_order"],
        "Global.uyghur_ctc_order",
    )
    if not str(global_cfg.get("model_protocol", "")).startswith(
        METHOD_PROTOCOL
    ):
        raise ValueError("Global.model_protocol is not a dual-order method")

    decoder = cfg["Architecture"]["Decoder"]
    require_equal(decoder.get("name"), "DualOrderGTCDecoder", "Decoder.name")
    require_equal(decoder.get("infer_gtc"), True, "Decoder.infer_gtc")
    require_equal(
        decoder.get("use_script_adapter"),
        spec["use_script_adapter"],
        "Decoder.use_script_adapter",
    )
    require_equal(
        decoder.get("use_local_direction"),
        spec["use_local_direction"],
        "Decoder.use_local_direction",
    )
    require_equal(
        decoder.get("use_sldr", False),
        spec.get("use_sldr", False),
        "Decoder.use_sldr",
    )
    require_equal(
        decoder.get("use_scdl", False),
        spec.get("use_scdl", False),
        "Decoder.use_scdl",
    )
    loss = cfg["Loss"]
    if method in ("m3_sldr_pilot", "m3_sldr_full_v1"):
        require_equal(decoder.get("sldr_reduction"), 4, "sldr_reduction")
        require_equal(decoder.get("sldr_gamma_init"), 0.001, "sldr_gamma_init")
        require_equal(decoder.get("detach"), False, "Decoder.detach")
        loss = cfg["Loss"]
        require_equal(loss.get("consistency_weight"), 0.15, "locked alpha")
        require_equal(loss.get("direction_weight"), 0.0, "no new loss")
    if method in ("m3_scdl_pilot", "m3_scdl_full_v1"):
        require_equal(decoder.get("detach"), False, "Decoder.detach")
        require_equal(
            decoder.get("scdl_use_space_char"), True, "scdl_use_space_char"
        )
        dictionary = Path(decoder.get("scdl_character_dict_path", ""))
        if not dictionary.is_file():
            raise FileNotFoundError(dictionary)
        require_equal(loss.get("consistency_weight"), 0.15, "locked alpha")
        require_equal(loss.get("direction_weight"), 0.0, "no direction loss")
        require_equal(loss.get("scdl_weight"), 0.05, "scdl_weight")
        require_equal(loss.get("scdl_topk"), 5, "scdl_topk")
        require_equal(loss.get("scdl_temperature"), 0.1, "scdl_temperature")
        require_equal(
            loss.get("scdl_warmup_fraction"), 0.2, "scdl_warmup_fraction"
        )
    require_equal(
        (decoder.get("gtc_decoder") or {}).get("name"),
        "SMTRDecoder",
        "Decoder.gtc_decoder.name",
    )
    require_equal(
        (decoder.get("ctc_decoder") or {}).get("name"),
        "RCTCDecoder",
        "Decoder.ctc_decoder.name",
    )

    require_equal(
        decoder.get("semantic_direction", "none"),
        spec.get("semantic_direction", "none"),
        "Decoder.semantic_direction",
    )
    if method in ("dir_d1", "dir_d2"):
        require_equal(decoder.get("detach"), False, "Decoder.detach")
        require_equal(decoder.get("direction_gate_initial"), 0.01, "direction_gate_initial")
    if method in ("dir_d1_v2", "dir_d2_v2"):
        require_equal(decoder.get("detach"), False, "Decoder.detach")
        require_equal(cfg["Loss"].get("consistency_weight"), 0.30, "locked alpha")
        require_equal(cfg["Loss"].get("direction_weight"), 0.0, "no new direction loss")
    if method == "dir_d1_v2" and "direction_gate_init_logit" in decoder:
        raise ValueError("D1 must not define a direction gate")
    if method == "dir_d2_v2":
        require_equal(decoder.get("direction_gate_init_logit"), -4.0, "direction_gate_init_logit")
    require_equal(loss.get("name"), "DualOrderGTCLoss", "Loss.name")
    for key in ("consistency_weight", "script_weight", "direction_weight"):
        require_equal(loss.get(key), expected_spec[key], f"Loss.{key}")

    for section in ("Train", "Eval"):
        transforms = transform_map(cfg, section)
        dual = transforms.get("DualOrderGTCLabelEncode")
        if not isinstance(dual, dict):
            raise ValueError(f"{section} lacks DualOrderGTCLabelEncode")
        require_equal(dual.get("ctc_order"), spec["ctc_order"], "ctc_order")
        require_equal(dual.get("sgm_order"), spec["sgm_order"], "sgm_order")
        require_equal(
            dual.get("max_text_length"), MAX_TEXT_LENGTH, "max_text_length"
        )
        require_equal(
            (transforms.get("KeepKeys") or {}).get("keep_keys"),
            EXPECTED_KEEP_KEYS,
            "KeepKeys",
        )
    initialization = validate_initialization(
        cfg, args.expected_initialization
    )
    train_manifests = validate_dataset(
        cfg, "Train", args.allow_missing_lmdb
    )
    eval_manifests = validate_dataset(cfg, "Eval", args.allow_missing_lmdb)
    result = {
        "status": "DUAL_ORDER_METHOD_CONFIG_OK",
        "method": method,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "preprocess_protocol": PROTOCOL_ID,
        "sampler_protocol": SAMPLER_PROTOCOL_ID,
        "initialization_checkpoint": initialization,
        "train_rows": sum(item.get("rows", 0) for item in train_manifests),
        "eval_rows": sum(item.get("rows", 0) for item in eval_manifests),
        "test_policy": "not_evaluated",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
