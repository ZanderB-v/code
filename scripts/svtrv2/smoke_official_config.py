#!/usr/bin/env python3
"""Build an official SVTRv2 config and run one GPU forward pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENOCR_ROOT = PROJECT_ROOT / "third_party" / "OpenOCR"
sys.path.insert(0, str(OPENOCR_ROOT))

import torch

from openrec.modeling import build_model
from openrec.postprocess import build_post_process
from tools.engine.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rec/svtrv2/svtrv2_small_rctc.yml",
        help="Config path relative to the OpenOCR repository.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = OPENOCR_ROOT / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    cfg = Config(str(config_path)).cfg
    postprocess = build_post_process(cfg["PostProcess"], cfg["Global"])
    cfg["Architecture"]["Decoder"]["out_channels"] = (
        postprocess.get_character_num()
    )

    device = torch.device(args.device)
    torch.manual_seed(20260717)
    model = build_model(cfg["Architecture"]).to(device).eval()
    image = torch.randn(
        args.batch_size,
        3,
        args.height,
        args.width,
        device=device,
    )

    with torch.inference_mode():
        output = model(image)

    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor output, got {type(output).__name__}")
    if not torch.isfinite(output).all():
        raise RuntimeError("Model output contains NaN or Inf")

    result = {
        "status": "MODEL_FORWARD_SMOKE_OK",
        "config": str(config_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input_shape": list(image.shape),
        "output_shape": list(output.shape),
        "character_num": postprocess.get_character_num(),
        "note": "Random weights and input; this validates configuration and execution only.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
