#!/usr/bin/env python3
"""Run a short single-GPU DDP HEM training smoke without saving a checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("b1", "m3"), required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    openocr_root = root / "third_party/OpenOCR"
    sys.path.insert(0, str(openocr_root))
    sys.path.insert(0, str(openocr_root / "tools"))

    import torch
    import torch.distributed as dist
    from openrec.losses import build_loss
    from openrec.modeling import build_model
    from openrec.optimizer import build_optimizer
    from openrec.postprocess import build_post_process
    from tools.data import build_dataloader
    from tools.engine.config import Config
    from tools.utils.ckpt import load_pretrained_params
    from tools.utils.logging import get_logger

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("HEM smoke is frozen to physical GPU1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device (physical GPU1)")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 1 or args.local_rank != 0:
        raise RuntimeError("HEM V1 smoke requires one process on physical GPU1")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats(device)

    cfg = Config(str(args.config.resolve())).cfg
    if cfg["Global"].get("hem_protocol") != "near_miss_hem_training_v1":
        raise ValueError("Config does not declare frozen HEM V1")
    if cfg["Global"].get("physical_gpu_ids") != [1]:
        raise ValueError("Config GPU policy changed")
    if int(cfg["Train"]["loader"]["batch_size_per_card"]) != 32:
        raise ValueError("HEM V1 must preserve global batch 32 on one GPU")
    if int(cfg["Train"]["sampler"].get("control_world_size", 0)) != 2:
        raise ValueError("HEM V1 must emulate the original two-rank schedule")
    if int(cfg["Train"]["sampler"].get("control_first_bs", 0)) != 16:
        raise ValueError("HEM V1 must preserve control batch 16 per rank")
    cfg["Global"]["distributed"] = True
    cfg["Global"]["use_amp"] = False
    cfg["Train"]["loader"]["num_workers"] = min(
        4, int(cfg["Train"]["loader"]["num_workers"])
    )
    logger = get_logger(f"{args.model}_hem_smoke")

    postprocess = build_post_process(cfg["PostProcess"], cfg["Global"])
    cfg["Architecture"]["Decoder"]["out_channels"] = postprocess.get_character_num()
    dataloader = build_dataloader(cfg, "Train", logger, task="rec")
    sampler = dataloader.batch_sampler
    if getattr(sampler, "hem_protocol", None) != "NEAR_MISS_HEM_WEIGHTED_RATIO_V1":
        raise ValueError("HEM weighted sampler was not constructed")
    sampler.set_epoch(0)
    planned_batches = list(iter(sampler))
    if len(planned_batches) != 3065:
        raise RuntimeError(
            f"HEM full-epoch steps changed: {len(planned_batches)} != 3065"
        )
    sampler.set_epoch(0)
    selected_indices = [
        int(item[2])
        for batch in planned_batches[: args.steps]
        for item in batch
    ]
    selected_weights = sampler.dataset_sample_weights[selected_indices]
    selected_counts = Counter(str(float(value)) for value in selected_weights)
    selected_hard_rate = float((selected_weights > 1.0).mean())
    population_hard_rate = float((sampler.dataset_sample_weights > 1.0).mean())
    if not selected_counts.get("2.0") or not selected_counts.get("1.5"):
        raise RuntimeError("Smoke prefix did not exercise both HEM weight classes")

    model = build_model(cfg["Architecture"]).to(device).train()
    pretrained = Path(cfg["Global"]["pretrained_model"])
    checkpoint = torch.load(pretrained, map_location="cpu")
    source_state = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    missing = [
        name
        for name, value in model_state.items()
        if name not in source_state or tuple(source_state[name].shape) != tuple(value.shape)
    ]
    unexpected = [name for name in source_state if name not in model_state]
    if missing or unexpected:
        raise ValueError(
            f"HEM initialization is not identical to {args.model} control: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    load_pretrained_params(model, str(pretrained), logger)
    del checkpoint, source_state
    loss_fn = build_loss(cfg["Loss"])
    optimizer, scheduler = build_optimizer(
        cfg["Optimizer"],
        cfg["LRScheduler"],
        epochs=int(cfg["Global"]["epoch_num"]),
        step_each_epoch=len(dataloader),
        model=model,
    )
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[0], find_unused_parameters=False
    )

    losses = []
    gradient_norms = []
    started = time.time()
    iterator = iter(dataloader)
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise RuntimeError("Smoke requested more steps than one epoch") from exc
        tensors = [item.to(device, non_blocking=True) for item in batch]
        optimizer.zero_grad(set_to_none=True)
        predictions = model(tensors[0], data=tensors[1:])
        loss_map = loss_fn(predictions, tensors)
        loss = loss_map["loss"]
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Non-finite loss at smoke step {step}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite gradient norm at smoke step {step}: {grad_norm}"
            )
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(grad_norm.detach().cpu()))
        if rank == 0 and ((step + 1) % 20 == 0 or step + 1 == args.steps):
            print(
                json.dumps(
                    {
                        "status": "HEM_SMOKE_PROGRESS",
                        "model": args.model,
                        "step": step + 1,
                        "steps": args.steps,
                        "loss": losses[-1],
                        "peak_gpu_memory_gib": round(
                            torch.cuda.max_memory_allocated(device) / 1024**3, 3
                        ),
                    }
                ),
                flush=True,
            )

    report = {
        "status": "HEM_DDP_TRAINING_SMOKE_OK",
        "model": args.model,
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config.resolve()),
        "physical_gpu": 1,
        "visible_cuda_device": 0,
        "gpu_name": torch.cuda.get_device_name(0),
        "world_size": world_size,
        "steps_run": args.steps,
        "full_epoch_optimizer_steps": len(dataloader),
        "global_batch_size": 32,
        "initialization_checkpoint": str(pretrained),
        "initialization_checkpoint_sha256": sha256(pretrained),
        "full_initialization_match": True,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "gradient_norm_max": max(gradient_norms),
        "finite_loss_and_gradients": True,
        "population_hard_rate": population_hard_rate,
        "smoke_selected_hard_rate": selected_hard_rate,
        "hard_frequency_multiplier_vs_population": selected_hard_rate / population_hard_rate,
        "selected_weight_counts": dict(selected_counts),
        "hem_manifest_sha256": sampler.sample_weight_manifest_sha256,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "elapsed_seconds": time.time() - started,
        "checkpoint_written": False,
        "test_evaluated": False,
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
