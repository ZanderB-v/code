#!/usr/bin/env python3
"""Run one P1/MSR batch through full SVTRv2-S, GTCLoss, and backward."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--smoke-batch-size",
        type=int,
        default=2,
        help="First MSR batch size used by the forward/backward smoke test.",
    )
    return parser.parse_args()


def tensor_shapes(value: Any) -> Any:
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: tensor_shapes(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_shapes(item) for item in value]
    return type(value).__name__


def audit_ratio_sampler(dataloader) -> dict:
    batch_list = getattr(getattr(dataloader, "batch_sampler", None), "batch_list", None)
    if batch_list is None:
        raise RuntimeError("RatioSampler batch_list is unavailable for audit")
    sampled_indices = [
        int(sample[2])
        for sampler_batch in batch_list
        for sample in sampler_batch
    ]
    dataset_samples = len(dataloader.dataset)
    unique_sampled_indices = len(set(sampled_indices))
    if unique_sampled_indices != dataset_samples:
        raise RuntimeError(
            "RatioSampler does not cover the complete train dataset: "
            f"dataset_samples={dataset_samples}, "
            f"unique_sampled_indices={unique_sampled_indices}, "
            f"assigned_with_padding={len(sampled_indices)}"
        )
    return {
        "dataset_samples": dataset_samples,
        "sampler_batches": len(batch_list),
        "assigned_with_padding": len(sampled_indices),
        "unique_sampled_indices": unique_sampled_indices,
        "complete_dataset_coverage": True,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config_path = args.config.resolve()
    openocr_root = root / "third_party" / "OpenOCR"
    sys.path.insert(0, str(openocr_root))
    sys.path.insert(0, str(openocr_root / "tools"))

    import torch
    from openrec.losses import build_loss
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from tools.data import build_dataloader
    from tools.engine.config import Config
    from tools.utils.ckpt import load_pretrained_params
    from tools.utils.logging import get_logger

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    cfg = Config(str(config_path)).cfg
    cfg["Global"]["distributed"] = False
    cfg["Global"]["use_amp"] = False
    cfg["Train"]["loader"]["num_workers"] = 0
    cfg["Train"]["sampler"]["first_bs"] = min(
        args.smoke_batch_size,
        int(cfg["Train"]["sampler"]["first_bs"]),
    )
    cfg["Train"]["loader"]["batch_size_per_card"] = min(
        args.smoke_batch_size,
        int(cfg["Train"]["loader"]["batch_size_per_card"]),
    )
    logger = get_logger("p1_full_svtrv2_smoke")

    postprocess = build_post_process(cfg["PostProcess"], cfg["Global"])
    out_channels = postprocess.get_character_num()
    if not isinstance(out_channels, list) or len(out_channels) != 2:
        raise ValueError(
            f"Full SVTRv2 must expose [SGM, CTC] class counts: {out_channels}"
        )
    cfg["Architecture"]["Decoder"]["out_channels"] = out_channels
    dataloader = build_dataloader(cfg, "Train", logger, task="rec")
    sampler_audit = audit_ratio_sampler(dataloader)
    batch = next(iter(dataloader))

    device = torch.device(f"cuda:{args.device_id}")
    model = build_model(cfg["Architecture"]).to(device).train()
    pretrained = cfg["Global"].get("pretrained_model")
    if pretrained:
        payload = torch.load(pretrained, map_location=torch.device("cpu"))
        source_state = payload.get("state_dict", payload)
        model_state = model.state_dict()
        required_prefixes = ("encoder.", "decoder.ctc_decoder.")
        missing_required = [
            name
            for name in model_state
            if name.startswith(required_prefixes)
            and (
                name not in source_state
                or tuple(source_state[name].shape) != tuple(model_state[name].shape)
            )
        ]
        if missing_required:
            raise ValueError(
                "Stage-1/full checkpoint does not cover the encoder and RCTC "
                f"branch: {missing_required[:20]}"
            )
        load_pretrained_params(model, str(pretrained), logger)
    loss_fn = build_loss(cfg["Loss"])

    batch_tensor = [item.to(device) for item in batch]
    model.zero_grad(set_to_none=True)
    predictions = model(batch_tensor[0], data=batch_tensor[1:])
    if set(predictions) != {"gtc_pred", "ctc_pred"}:
        raise ValueError(f"Unexpected full-model outputs: {predictions.keys()}")
    loss_map = loss_fn(predictions, batch_tensor)
    for name in ("loss", "ctc_loss", "gtc_loss"):
        if name not in loss_map or not torch.isfinite(loss_map[name]).all():
            raise FloatingPointError(f"Invalid {name}: {loss_map.get(name)}")
    loss_map["loss"].backward()

    groups = {"encoder": 0, "sgm": 0, "rctc": 0}
    squared_norm = 0.0
    nonfinite = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
        squared_norm += float(parameter.grad.detach().float().norm()) ** 2
        if name.startswith("encoder."):
            groups["encoder"] += 1
        elif "decoder.gtc_decoder." in name:
            groups["sgm"] += 1
        elif "decoder.ctc_decoder." in name:
            groups["rctc"] += 1
    if nonfinite:
        raise FloatingPointError(f"Non-finite gradients: {nonfinite[:10]}")
    missing_groups = [name for name, count in groups.items() if count == 0]
    if missing_groups:
        raise RuntimeError(f"No gradients reached: {missing_groups}")

    result = {
        "status": "P1_FULL_SVTRV2_ONE_BATCH_SMOKE_OK",
        "config": str(config_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(args.device_id),
        "input_shape": list(batch_tensor[0].shape),
        "batch_shapes": tensor_shapes(batch_tensor),
        "prediction_shapes": tensor_shapes(predictions),
        "loss": float(loss_map["loss"].detach().cpu()),
        "ctc_loss": float(loss_map["ctc_loss"].detach().cpu()),
        "gtc_loss": float(loss_map["gtc_loss"].detach().cpu()),
        "gradient_groups": groups,
        "gradient_l2_norm": math.sqrt(squared_norm),
        "ratio_sampler_audit": sampler_audit,
        "class_counts": {"sgm": out_channels[0], "ctc": out_channels[1]},
        "initialization_coverage": {
            "encoder_and_rctc_complete": True,
            "sgm": (
                "loaded_from_full_checkpoint"
                if any(
                    name.startswith("decoder.gtc_decoder.")
                    for name in source_state
                )
                else "random_initialization"
            ),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
