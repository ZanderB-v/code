#!/usr/bin/env python3
"""Run a real dual-order P1/MSR batch through forward and backward."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from p1_msr_protocol import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-full-initialization", action="store_true")
    parser.add_argument("--allow-new-semantic-parameters", action="store_true")
    parser.add_argument("--allow-new-sldr-parameters", action="store_true")
    parser.add_argument("--allow-new-scdl-parameters", action="store_true")
    return parser.parse_args()


def tensor_shapes(value: Any) -> Any:
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: tensor_shapes(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_shapes(item) for item in value]
    return type(value).__name__


def audit_initialization(
    model_state,
    source_state,
    require_full,
    allow_new_semantic,
    allow_new_sldr,
    allow_new_scdl,
):
    fresh = []
    if allow_new_semantic:
        if not require_full:
            raise ValueError("New semantic parameters require full shared-M3 initialization")
        fresh = [name for name in model_state if name.startswith(
            ("decoder.semantic_direction_conditioner.", "decoder.direction_gate_logits"))]
        if any(name in source_state for name in fresh):
            raise ValueError("Shared M3 initialization already contains direction parameters")
        if set(source_state) - set(model_state):
            raise ValueError("Unexpected source parameters in shared M3 checkpoint")
    if allow_new_sldr:
        if not require_full:
            raise ValueError("New SLDR parameters require full shared-M3 initialization")
        fresh.extend(
            name for name in model_state if name.startswith("decoder.sldr.")
        )
        if not fresh:
            raise ValueError("SLDR was requested but the model has no SLDR parameters")
        if any(name in source_state for name in fresh):
            raise ValueError("Shared M3 initialization already contains SLDR parameters")
        unexpected = set(source_state) - set(model_state)
        if unexpected:
            raise ValueError(
                "Unexpected source parameters in shared M3 checkpoint: "
                f"{sorted(unexpected)[:20]}"
            )
    if allow_new_scdl:
        if not require_full:
            raise ValueError("New SCDL parameters require full shared-M3 initialization")
        fresh.extend(
            name for name in model_state if name.startswith("decoder.scdl.")
        )
        if not fresh:
            raise ValueError("SCDL was requested but the model has no SCDL state")
        if any(name in source_state for name in fresh):
            raise ValueError("Shared M3 initialization already contains SCDL state")
        unexpected = set(source_state) - set(model_state)
        if unexpected:
            raise ValueError(
                "Unexpected source parameters in shared M3 checkpoint: "
                f"{sorted(unexpected)[:20]}"
            )
    fresh = sorted(set(fresh))
    required = list(model_state) if require_full else [name for name in model_state
                if name.startswith(("encoder.", "decoder.ctc_decoder."))]
    missing = [name for name in required if name not in fresh and (
        name not in source_state or tuple(source_state[name].shape) != tuple(model_state[name].shape))]
    if missing:
        raise ValueError(f"Initialization does not cover the required parameters: {missing[:20]}")
    return fresh


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config_path = args.config.resolve()
    openocr_root = root / "third_party" / "OpenOCR"
    sys.path.insert(0, str(openocr_root))
    sys.path.insert(0, str(openocr_root / "tools"))

    import lmdb
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
    if args.smoke_batch_size < 1:
        raise ValueError("--smoke-batch-size must be positive")
    cfg = Config(str(config_path)).cfg
    if args.allow_new_semantic_parameters and (
        not args.require_full_initialization
        or cfg["Global"].get("method_variant") not in ("dir_d1_v2", "dir_d2_v2")
    ):
        raise ValueError("Fresh semantic parameters are allowed only for full M3 initialization in D1/D2 V2")
    if args.allow_new_sldr_parameters and (
        not args.require_full_initialization
        or cfg["Global"].get("method_variant")
        not in ("m3_sldr_pilot", "m3_sldr_full_v1")
        or not cfg["Architecture"]["Decoder"].get("use_sldr")
    ):
        raise ValueError(
            "Fresh SLDR parameters are allowed only for a controlled M3+SLDR run"
        )
    if args.allow_new_scdl_parameters and (
        not args.require_full_initialization
        or cfg["Global"].get("method_variant")
        not in ("m3_scdl_pilot", "m3_scdl_full_v1")
        or not cfg["Architecture"]["Decoder"].get("use_scdl")
    ):
        raise ValueError(
            "Fresh SCDL parameters are allowed only for a controlled M3+SCDL run"
        )
    cfg["Global"]["distributed"] = False
    cfg["Global"]["use_amp"] = False
    cfg["Train"]["loader"]["num_workers"] = 0
    control_world_size = int(cfg["Train"]["sampler"].get("control_world_size", 1))
    # A smoke forward/backward intentionally uses a tiny local batch. The
    # formal single-GPU control mode requires its original global batch and
    # therefore cannot coexist with that tiny batch. The formal sampler audit
    # still validates control emulation; this local smoke only validates data,
    # initialization, forward, loss, and gradients.
    if control_world_size > 1:
        cfg["Train"]["sampler"].pop("control_world_size", None)
        cfg["Train"]["sampler"].pop("control_first_bs", None)
    cfg["Train"]["sampler"]["first_bs"] = min(
        args.smoke_batch_size, int(cfg["Train"]["sampler"]["first_bs"])
    )
    cfg["Train"]["loader"]["batch_size_per_card"] = min(
        args.smoke_batch_size,
        int(cfg["Train"]["loader"]["batch_size_per_card"]),
    )
    logger = get_logger("dual_order_method_smoke")

    lmdb_sources = []
    for source in cfg["Train"]["dataset"].get("data_dir_list", []):
        source_path = Path(source)
        if not source_path.is_dir():
            raise FileNotFoundError(f"Missing train LMDB directory: {source_path}")
        env = lmdb.open(
            str(source_path),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=4,
        )
        with env.begin(write=False) as transaction:
            raw_count = transaction.get(b"num-samples")
            first_image = transaction.get(b"image-000000001")
            first_label = transaction.get(b"label-000000001")
        env.close()
        if raw_count is None:
            raise ValueError(f"LMDB has no num-samples key: {source_path}")
        row_count = int(raw_count)
        if row_count <= 0 or first_image is None or first_label is None:
            raise ValueError(
                f"LMDB is empty or incomplete: {source_path}, rows={row_count}"
            )
        lmdb_sources.append({"path": str(source_path), "rows": row_count})
    if not lmdb_sources:
        raise ValueError("Train config has no LMDB data_dir_list entries")

    postprocess = build_post_process(cfg["PostProcess"], cfg["Global"])
    out_channels = postprocess.get_character_num()
    if not isinstance(out_channels, list) or len(out_channels) != 2:
        raise ValueError(f"Expected [SGM, CTC] class counts: {out_channels}")
    cfg["Architecture"]["Decoder"]["out_channels"] = out_channels
    dataloader = build_dataloader(cfg, "Train", logger, task="rec")
    if len(dataloader) <= 0:
        raise RuntimeError("Dual-order smoke train dataloader is empty")
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    batch_list = getattr(batch_sampler, "batch_list", None)
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
    batch = next(iter(dataloader))

    device = torch.device(f"cuda:{args.device_id}")
    model = build_model(cfg["Architecture"]).to(device).train()
    pretrained = cfg["Global"].get("pretrained_model")
    source_state = {}
    fresh_semantic_names = []
    if pretrained:
        payload = torch.load(pretrained, map_location=torch.device("cpu"))
        source_state = payload.get("state_dict", payload)
        model_state = model.state_dict()
        fresh_semantic_names = audit_initialization(
            model_state, source_state, args.require_full_initialization,
            args.allow_new_semantic_parameters,
            args.allow_new_sldr_parameters,
            args.allow_new_scdl_parameters)
        load_pretrained_params(model, str(pretrained), logger)
    loss_fn = build_loss(cfg["Loss"])

    batch_tensor = [item.to(device) for item in batch]
    if cfg["Architecture"]["Decoder"].get("use_scdl"):
        # Seed the prototype bank only after the configured warmup boundary;
        # formal warmup intentionally leaves the bank untouched.
        loss_fn.set_training_progress(100, 100)
        with torch.no_grad():
            warmup_predictions = model(batch_tensor[0], data=batch_tensor[1:])
            loss_fn(warmup_predictions, batch_tensor)
        loss_fn.set_training_progress(100, 100)
    model.zero_grad(set_to_none=True)
    predictions = model(batch_tensor[0], data=batch_tensor[1:])
    required_outputs = {"gtc_pred", "ctc_pred"}
    if not required_outputs.issubset(predictions):
        raise ValueError(f"Missing model outputs: {predictions.keys()}")
    loss_map = loss_fn(predictions, batch_tensor)
    expected_losses = {
        "loss",
        "ctc_loss",
        "gtc_loss",
        "cross_order_loss",
        "script_loss",
        "direction_loss",
        "scdl_loss",
        "scdl_active_weight",
        "scdl_valid_tokens",
        "scdl_aligned_tokens",
        "scdl_valid_samples",
        "scdl_excluded_alignment_samples",
    }
    if not expected_losses.issubset(loss_map):
        raise ValueError(f"Missing loss terms: {loss_map.keys()}")
    for name, value in loss_map.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Invalid {name}: {value}")
    weighted_terms = {
        "cross_order_loss": float(cfg["Loss"].get("consistency_weight", 0)),
        "script_loss": float(cfg["Loss"].get("script_weight", 0)),
        "direction_loss": float(cfg["Loss"].get("direction_weight", 0)),
        "scdl_loss": float(cfg["Loss"].get("scdl_weight", 0)),
    }
    for name, weight in weighted_terms.items():
        value = float(loss_map[name].detach().cpu())
        if weight == 0.0 and abs(value) > 1e-10:
            raise ValueError(f"Disabled {name} is nonzero: {value}")
        if weight > 0.0 and value <= 0.0:
            raise ValueError(f"Enabled {name} is not positive: {value}")
    loss_map["loss"].backward()

    groups = {
        "encoder": 0,
        "sgm": 0,
        "rctc": 0,
        "script_adapter": 0,
        "local_direction": 0,
        "semantic_direction": 0,
        "direction_gate": 0,
        "sldr": 0,
        "scdl": 0,
    }
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
        elif "decoder.script_adapter." in name:
            groups["script_adapter"] += 1
        elif "decoder.direction_conditioner." in name:
            groups["local_direction"] += 1
        elif "decoder.semantic_direction_conditioner." in name:
            groups["semantic_direction"] += 1
        elif "decoder.direction_gate_logits" in name:
            groups["direction_gate"] += 1
        elif name.startswith("decoder.sldr."):
            groups["sldr"] += 1
        elif name.startswith("decoder.scdl."):
            groups["scdl"] += 1
    if nonfinite:
        raise FloatingPointError(f"Non-finite gradients: {nonfinite[:10]}")
    required_groups = ["encoder", "sgm", "rctc"]
    if cfg["Architecture"]["Decoder"].get("use_script_adapter"):
        required_groups.append("script_adapter")
    if cfg["Architecture"]["Decoder"].get("use_local_direction"):
        required_groups.append("local_direction")
    semantic_mode = cfg["Architecture"]["Decoder"].get("semantic_direction", "none")
    if semantic_mode != "none":
        required_groups.append("semantic_direction")
    if semantic_mode in ("script_gated", "sample_script_gated"):
        required_groups.append("direction_gate")
    if cfg["Architecture"]["Decoder"].get("use_sldr"):
        required_groups.append("sldr")
    if cfg["Architecture"]["Decoder"].get("use_scdl"):
        required_groups.append("scdl")
    missing_groups = [name for name in required_groups if groups[name] == 0]
    if missing_groups:
        raise RuntimeError(f"No gradients reached: {missing_groups}")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    sldr_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("decoder.sldr.")
    )
    scdl_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("decoder.scdl.")
    )

    result = {
        "status": "DUAL_ORDER_REAL_BATCH_FORWARD_BACKWARD_OK",
        "config": str(config_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(args.device_id),
        "method_variant": cfg["Global"].get("method_variant"),
        "uyghur_ctc_order": cfg["Global"].get("uyghur_ctc_order"),
        "input_shape": list(batch_tensor[0].shape),
        "train_lmdb_sources": lmdb_sources,
        "train_dataloader_batches": len(dataloader),
        "ratio_sampler_audit": {
            "dataset_samples": dataset_samples,
            "sampler_batches": len(batch_list),
            "assigned_with_padding": len(sampled_indices),
            "unique_sampled_indices": unique_sampled_indices,
            "complete_dataset_coverage": True,
            "control_world_emulation_disabled_for_tiny_smoke": (
                control_world_size > 1
            ),
        },
        "batch_shapes": tensor_shapes(batch_tensor),
        "prediction_shapes": tensor_shapes(predictions),
        "losses": {
            key: float(value.detach().cpu()) for key, value in loss_map.items()
        },
        "gradient_groups": groups,
        "gradient_l2_norm": math.sqrt(squared_norm),
        "parameter_audit": {
            "total_parameters": total_parameters,
            "sldr_parameters": sldr_parameters,
            "sldr_parameter_ratio": (
                sldr_parameters / total_parameters if total_parameters else 0.0
            ),
            "scdl_parameters": scdl_parameters,
            "scdl_parameter_ratio": (
                scdl_parameters / total_parameters if total_parameters else 0.0
            ),
        },
        "class_counts": {"sgm": out_channels[0], "ctc": out_channels[1]},
        "initialization_coverage": {
            "checkpoint": str(pretrained) if pretrained else None,
            "checkpoint_sha256": sha256_file(Path(pretrained)) if pretrained else None,
            "loaded_keys": len(source_state),
            "encoder_and_rctc_complete": True,
            "full_model_required": args.require_full_initialization,
            "full_model_complete": (
                not fresh_semantic_names if args.require_full_initialization else None
            ),
            "shared_m3_complete": True if (
                args.allow_new_semantic_parameters
                or args.allow_new_sldr_parameters
                or args.allow_new_scdl_parameters
            ) else None,
            "fresh_semantic_parameter_keys": fresh_semantic_names,
            "fresh_parameter_keys": fresh_semantic_names,
            "sgm": (
                "loaded"
                if any(
                    name.startswith("decoder.gtc_decoder.")
                    for name in source_state
                )
                else "random_initialization"
            ),
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
