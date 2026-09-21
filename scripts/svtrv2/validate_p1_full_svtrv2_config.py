#!/usr/bin/env python3
"""Fail-fast validation for the B1 full SVTRv2-S + SGM P1/MSR config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from p1_msr_protocol import (
    MAX_TEXT_LENGTH,
    PROTOCOL_ID,
    SAMPLER_PROTOCOL_ID,
    assert_p1_config_text,
)
from validate_p1_msr_config import (
    require_equal,
    sha256,
    validate_dataset,
    validate_initialization,
)


MODEL_PROTOCOL = "B1_FULL_SVTRV2_S_SGM_U2_V1"
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
    "ctc_label",
    "ctc_length",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--expected-initialization",
        choices=("checkpoint",),
        default="checkpoint",
    )
    parser.add_argument("--allow-missing-lmdb", action="store_true")
    return parser.parse_args()


def transform_map(cfg: dict[str, Any], section: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    transforms = cfg[section]["dataset"].get("transforms") or []
    for item in transforms:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"Malformed transform in {section}: {item!r}")
        name, value = next(iter(item.items()))
        result[name] = value or {}
    return result


def validate_gtc_labels(cfg: dict[str, Any], section: str) -> None:
    transforms = transform_map(cfg, section)
    if "CTCLabelEncode" in transforms:
        raise ValueError(
            f"{section} uses CTCLabelEncode; B1 requires GTCLabelEncode so "
            "the same U2 label supervises both SGM and RCTC."
        )
    gtc = transforms.get("GTCLabelEncode")
    if not isinstance(gtc, dict):
        raise ValueError(f"{section} is missing GTCLabelEncode")
    require_equal(
        gtc.get("max_text_length"),
        MAX_TEXT_LENGTH,
        f"{section}.GTCLabelEncode.max_text_length",
    )
    inner = gtc.get("gtc_label_encode") or {}
    require_equal(
        inner.get("name"),
        "SMTRLabelEncode",
        f"{section}.GTCLabelEncode.gtc_label_encode.name",
    )
    require_equal(
        inner.get("sub_str_len"),
        5,
        f"{section}.GTCLabelEncode.sub_str_len",
    )
    keep = transforms.get("KeepKeys") or {}
    require_equal(
        keep.get("keep_keys"),
        EXPECTED_KEEP_KEYS,
        f"{section}.KeepKeys.keep_keys",
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    text = config_path.read_text(encoding="utf-8-sig")
    assert_p1_config_text(text)

    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Install PyYAML in openocr_svtrv2: python -m pip install pyyaml"
        ) from exc
    cfg = yaml.safe_load(text)

    global_cfg = cfg["Global"]
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
        global_cfg.get("model_protocol"),
        MODEL_PROTOCOL,
        "Global.model_protocol",
    )
    require_equal(
        global_cfg.get("evaluation_branch"),
        "ctc",
        "Global.evaluation_branch",
    )
    require_equal(
        global_cfg.get("max_text_length"),
        MAX_TEXT_LENGTH,
        "Global.max_text_length",
    )

    architecture = cfg["Architecture"]
    require_equal(architecture.get("algorithm"), "SVTRv2", "algorithm")
    encoder = architecture["Encoder"]
    require_equal(
        encoder.get("name"),
        "SVTRv2LNConvTwo33",
        "Architecture.Encoder.name",
    )
    require_equal(encoder.get("dims"), [96, 192, 384], "Encoder.dims")
    require_equal(encoder.get("depths"), [3, 6, 3], "Encoder.depths")

    decoder = architecture["Decoder"]
    require_equal(decoder.get("name"), "GTCDecoder", "Decoder.name")
    require_equal(decoder.get("infer_gtc"), True, "Decoder.infer_gtc")
    require_equal(decoder.get("detach"), False, "Decoder.detach")
    require_equal(
        (decoder.get("gtc_decoder") or {}).get("name"),
        "SMTRDecoder",
        "Decoder.gtc_decoder.name",
    )
    require_equal(
        (decoder.get("gtc_decoder") or {}).get("num_layer"),
        1,
        "Decoder.gtc_decoder.num_layer",
    )
    require_equal(
        (decoder.get("ctc_decoder") or {}).get("name"),
        "RCTCDecoder",
        "Decoder.ctc_decoder.name",
    )

    loss = cfg["Loss"]
    require_equal(loss.get("name"), "GTCLoss", "Loss.name")
    require_equal(loss.get("ctc_weight"), 0.1, "Loss.ctc_weight")
    require_equal(loss.get("gtc_weight"), 1.0, "Loss.gtc_weight")
    require_equal(
        (loss.get("gtc_loss") or {}).get("name"),
        "SMTRLoss",
        "Loss.gtc_loss.name",
    )
    require_equal(
        cfg["PostProcess"].get("name"),
        "GTCLabelDecode",
        "PostProcess.name",
    )
    require_equal(cfg["Metric"].get("name"), "RecGTCMetric", "Metric.name")

    if not isinstance(cfg["Optimizer"].get("lr"), (float, int)):
        raise TypeError("Optimizer.lr must be numeric")
    if "test" in text.lower():
        raise ValueError("B1 development config unexpectedly contains test")

    for section in ("Train", "Eval"):
        validate_gtc_labels(cfg, section)
    initialization = validate_initialization(
        cfg,
        args.expected_initialization,
    )
    train_manifests = validate_dataset(
        cfg,
        "Train",
        args.allow_missing_lmdb,
    )
    eval_manifests = validate_dataset(
        cfg,
        "Eval",
        args.allow_missing_lmdb,
    )

    result = {
        "status": "P1_FULL_SVTRV2_S_CONFIG_OK",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "preprocess_protocol": PROTOCOL_ID,
        "sampler_protocol": SAMPLER_PROTOCOL_ID,
        "model_protocol": MODEL_PROTOCOL,
        "initialization_checkpoint": initialization,
        "training_supervision": {
            "ctc": "same_LMDB_U2_label",
            "sgm": "same_LMDB_U2_label",
        },
        "checkpoint_selection": "clean_target_dev_macro_CER_on_CTC_branch",
        "train_rows": sum(item.get("rows", 0) for item in train_manifests),
        "eval_rows": sum(item.get("rows", 0) for item in eval_manifests),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
