#!/usr/bin/env python3
"""Run one real P1/MSR batch through model, CTC loss, and backward."""

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
    if args.device_id >= torch.cuda.device_count():
        raise ValueError(
            f"Requested cuda:{args.device_id}, "
            f"but only {torch.cuda.device_count()} GPUs are visible"
        )

    cfg = Config(str(config_path)).cfg
    cfg["Global"]["distributed"] = False
    cfg["Global"]["use_amp"] = False
    cfg["Train"]["loader"]["num_workers"] = 0
    logger = get_logger("p1_rctc_smoke")

    postprocess = build_post_process(cfg["PostProcess"], cfg["Global"])
    cfg["Architecture"]["Decoder"]["out_channels"] = (
        postprocess.get_character_num()
    )
    dataloader = build_dataloader(cfg, "Train", logger, task="rec")
    sampler_audit = audit_ratio_sampler(dataloader)
    batch = next(iter(dataloader))

    device = torch.device(f"cuda:{args.device_id}")
    model = build_model(cfg["Architecture"]).to(device).train()
    pretrained = cfg["Global"].get("pretrained_model")
    if pretrained:
        load_pretrained_params(model, str(pretrained), logger)
    loss_fn = build_loss(cfg["Loss"])

    batch_tensor = [item.to(device) for item in batch]
    model.zero_grad(set_to_none=True)
    predictions = model(batch_tensor[0], data=batch_tensor[1:])
    loss_map = loss_fn(predictions, batch_tensor)
    loss_value = loss_map["loss"]
    if not torch.isfinite(loss_value).all():
        raise FloatingPointError(f"Non-finite smoke loss: {loss_value}")
    loss_value.backward()

    gradient_tensors = 0
    finite_gradient_tensors = 0
    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient_tensors += 1
        if torch.isfinite(parameter.grad).all():
            finite_gradient_tensors += 1
        squared_norm += float(parameter.grad.detach().float().norm().item()) ** 2
    if gradient_tensors == 0:
        raise RuntimeError("Backward produced no gradients")
    if finite_gradient_tensors != gradient_tensors:
        raise FloatingPointError(
            f"Non-finite gradients: "
            f"{gradient_tensors - finite_gradient_tensors}/{gradient_tensors}"
        )

    result = {
        "status": "P1_RCTC_ONE_BATCH_SMOKE_OK",
        "config": str(config_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(args.device_id),
        "initialization": str(pretrained) if pretrained else "random",
        "input_shape": list(batch_tensor[0].shape),
        "label_shape": list(batch_tensor[1].shape),
        "length_shape": list(batch_tensor[2].shape),
        "prediction_shape": tensor_shapes(predictions),
        "loss": float(loss_value.detach().cpu()),
        "gradient_tensors": gradient_tensors,
        "gradient_l2_norm": math.sqrt(squared_norm),
        "train_batches": len(dataloader),
        "ratio_sampler_audit": sampler_audit,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
