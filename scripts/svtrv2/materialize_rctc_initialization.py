#!/usr/bin/env python3
"""Materialize a complete 4891-class RCTC initialization for shape preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config_path = args.config.resolve()
    source_path = args.source.resolve()
    output_path = args.output.resolve()
    if not config_path.is_file() or not source_path.is_file():
        raise FileNotFoundError(f"Missing config/source: {config_path}, {source_path}")
    if output_path.exists() and not args.replace:
        raise FileExistsError(f"Output exists; use --replace: {output_path}")

    openocr_root = root / "third_party/OpenOCR"
    sys.path.insert(0, str(openocr_root))
    sys.path.insert(0, str(openocr_root / "tools"))
    import numpy as np
    import torch

    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from tools.engine.config import Config

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = Config(str(config_path)).cfg
    postprocess = build_post_process(cfg["PostProcess"], cfg["Global"])
    cfg["Architecture"]["Decoder"]["out_channels"] = (
        postprocess.get_character_num()
    )
    model = build_model(cfg["Architecture"])
    model_state = model.state_dict()
    payload = torch.load(source_path, map_location=torch.device("cpu"))
    source_state = payload.get("state_dict", payload)
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    encoder_keys = [key for key in model_state if key.startswith("encoder.")]
    missing_encoder = [key for key in encoder_keys if key not in compatible]
    if missing_encoder:
        raise ValueError(
            "Union14M does not fully initialize the visual encoder: "
            f"{missing_encoder[:20]}"
        )
    model.load_state_dict(compatible, strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, output_path)
    report = {
        "status": "RCTC_4891_INITIALIZATION_MATERIALIZED",
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "encoder_loaded_ratio": 1.0,
        "loaded_tensors": len(compatible),
        "materialized_tensors": len(model_state),
        "dictionary_dependent_head": "randomly_initialized_for_shape_preflight_only",
        "formal_training_checkpoint": False,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
