#!/usr/bin/env python3
"""Verify that B0 and B1 differ only in the intended full-SVTRv2 stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-config", type=Path, required=True)
    parser.add_argument("--b1-config", type=Path, required=True)
    parser.add_argument("--stage", choices=("synthetic", "target"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def get(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for key in path.split("."):
        value = value[key]
    return value


def canonicalize(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\\", "/").replace(
            "/home/data_home/",
            "/data_home/",
        )
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    if isinstance(value, dict):
        return {key: canonicalize(item) for key, item in value.items()}
    return value


def main() -> None:
    args = parse_args()
    import yaml  # type: ignore

    b0 = yaml.safe_load(args.b0_config.read_text(encoding="utf-8-sig"))
    b1 = yaml.safe_load(args.b1_config.read_text(encoding="utf-8-sig"))
    equal_fields = [
        "Global.character_dict_path",
        "Global.max_text_length",
        "Global.use_space_char",
        "Global.preprocess_protocol",
        "Global.sampler_protocol",
        "Architecture.model_type",
        "Architecture.algorithm",
        "Architecture.Encoder",
        "Train.dataset.name",
        "Train.dataset.ds_width",
        "Train.dataset.padding",
        "Train.dataset.base_shape",
        "Train.dataset.base_h",
        "Train.dataset.data_dir_list",
        "Train.sampler.name",
        "Train.sampler.scales",
        "Train.sampler.divided_factor",
        "Train.sampler.max_ratio",
        "Train.loader.max_ratio",
        "Eval.dataset.name",
        "Eval.dataset.ds_width",
        "Eval.dataset.padding",
        "Eval.dataset.base_shape",
        "Eval.dataset.base_h",
        "Eval.dataset.data_dir_list",
        "Eval.sampler.name",
        "Eval.sampler.scales",
        "Eval.sampler.divided_factor",
        "Eval.sampler.max_ratio",
        "Eval.loader.max_ratio",
    ]
    mismatches = []
    for field in equal_fields:
        b0_value = get(b0, field)
        b1_value = get(b1, field)
        if canonicalize(b0_value) != canonicalize(b1_value):
            mismatches.append(
                {"field": field, "b0": b0_value, "b1": b1_value}
            )
    if mismatches:
        raise ValueError(
            "B1 controlled-comparison mismatch:\n"
            + json.dumps(mismatches, ensure_ascii=False, indent=2)
        )

    decoder = b1["Architecture"]["Decoder"]
    required = {
        "decoder": decoder.get("name") == "GTCDecoder",
        "sgm": (decoder.get("gtc_decoder") or {}).get("name")
        == "SMTRDecoder",
        "rctc": (decoder.get("ctc_decoder") or {}).get("name")
        == "RCTCDecoder",
        "gtc_loss": b1["Loss"].get("name") == "GTCLoss",
        "ctc_weight": b1["Loss"].get("ctc_weight") == 0.1,
        "same_u2_label_encoder": all(
            any("GTCLabelEncode" in item for item in b1[section]["dataset"]["transforms"])
            for section in ("Train", "Eval")
        ),
        "ctc_only_selection": b1["Global"].get("evaluation_branch") == "ctc",
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise ValueError(f"B1 method checks failed: {failed}")
    if "test" in args.b1_config.read_text(encoding="utf-8-sig").lower():
        raise ValueError("B1 config contains a test reference")

    result = {
        "status": "B1_CONTROLLED_COMPARISON_OK",
        "stage": args.stage,
        "b0_config": str(args.b0_config.resolve()),
        "b1_config": str(args.b1_config.resolve()),
        "equal_fields": equal_fields,
        "intended_changes": {
            "decoder": "RCTCDecoder -> GTCDecoder(SMTRDecoder + RCTCDecoder)",
            "loss": "CTCLoss -> GTCLoss(SMTR + 0.1*CTC)",
            "label_encoder": (
                "CTCLabelEncode -> GTCLabelEncode; both branches receive the "
                "same frozen U2 LMDB text"
            ),
            "stage2_resources": (
                "batch size and learning rate reduced for the auxiliary SGM"
            ),
        },
        "method_checks": required,
        "test_policy": "not_referenced_or_evaluated",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
