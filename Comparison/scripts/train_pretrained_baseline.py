#!/usr/bin/env python3
"""DDP trainer for the controlled public-pretrained baseline comparison."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from formal_baseline_data import (
    PROJECT_ROOT,
    formal_implementation_hashes,
    load_protocol,
    load_records,
    read_dictionary,
)
from formal_baseline_runtime import (
    LineDataset,
    amp_enabled_for_model,
    collate_lines,
    evaluate_clean_dev,
    training_objective,
)
from pretrained_models import build_model_bundle, sha256_file


BATCH_POLICY = {
    "crnn": {"per_gpu": 16, "accumulate": 2, "eval": 64},
    "svtr": {"per_gpu": 8, "accumulate": 4, "eval": 32},
    "parseq": {"per_gpu": 4, "accumulate": 8, "eval": 16},
    "abinet": {"per_gpu": 2, "accumulate": 16, "eval": 8},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(BATCH_POLICY))
    parser.add_argument("--stage", required=True, choices=("synthetic", "target"))
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--patience-evals", type=int)
    parser.add_argument("--min-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Formal baseline training requires CUDA")
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world, torch.device(f"cuda:{local_rank}")


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def load_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {path}")
    for key in ("model_state_dict", "state_dict"):
        state = payload.get(key)
        if isinstance(state, dict):
            return state, payload
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload, {}
    raise ValueError(f"No model state found in {path}")


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def scheduler_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def main() -> None:
    args = parse_args()
    rank, local_rank, world, device = setup_distributed()
    if world != 2:
        raise ValueError(
            f"Formal comparison requires exactly two GPUs for the frozen global "
            f"batch protocol, got world_size={world}"
        )
    is_main = rank == 0
    protocol = load_protocol(args.protocol)
    protocol_path = args.protocol or (
        PROJECT_ROOT / "Comparison/pretrained_baselines_v1/protocol.json"
    )
    protocol_sha256 = sha256_file(protocol_path)
    implementation_sha256 = formal_implementation_hashes()
    train_spec = protocol["training_protocol"]
    seed = int(args.seed if args.seed is not None else train_spec["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.stage == "synthetic":
        dataset_name = "s50"
        defaults = {
            "max_epochs": int(train_spec["synthetic_max_epochs"]),
            "eval_every": int(train_spec["synthetic_eval_every_epochs"]),
            "patience": int(train_spec["synthetic_early_stop_patience_evals"]),
            "min_epochs": 15,
            "lr": 1e-4,
        }
        if args.source_checkpoint is not None:
            raise ValueError("Synthetic stage must use the audited public initialization")
        source_checkpoint = (
            PROJECT_ROOT
            / f"Comparison/pretrained_baselines_v1/outputs/{args.model}_4891_initialization.pth"
        )
    else:
        dataset_name = "target_train"
        defaults = {
            "max_epochs": int(train_spec["target_max_epochs"]),
            "eval_every": int(train_spec["target_eval_every_epochs"]),
            "patience": int(train_spec["target_early_stop_patience_evals"]),
            "min_epochs": 10,
            "lr": 5e-5,
        }
        if args.source_checkpoint is None:
            raise ValueError("Target stage requires --source-checkpoint from S50 adaptation")
        source_checkpoint = args.source_checkpoint
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)

    max_epochs = args.max_epochs or defaults["max_epochs"]
    eval_every = args.eval_every or defaults["eval_every"]
    patience = args.patience_evals or defaults["patience"]
    min_epochs = args.min_epochs or defaults["min_epochs"]
    learning_rate = args.learning_rate or defaults["lr"]
    characters = read_dictionary(
        PROJECT_ROOT / protocol["dictionary"],
        bool(protocol["use_space_char"]),
    )
    if len(characters) != int(protocol["expected_recognition_symbol_count"]):
        raise ValueError("Runtime recognition-symbol count mismatch")
    char_to_id = {character: index + 1 for index, character in enumerate(characters)}
    model_spec = protocol["models"][args.model]
    public_checkpoint = PROJECT_ROOT / model_spec["checkpoint"]
    bundle = build_model_bundle(
        args.model,
        public_checkpoint,
        len(characters),
        int(protocol["max_label_length"]),
    )
    state, source_payload = load_state(source_checkpoint)
    if source_payload.get("protocol_sha256") != protocol_sha256:
        raise ValueError(
            f"Source checkpoint belongs to a stale protocol: {source_checkpoint}"
        )
    if source_payload.get("implementation_sha256") != implementation_sha256:
        raise ValueError(
            f"Source checkpoint belongs to stale comparison code: {source_checkpoint}"
        )
    bundle.model.load_state_dict(state, strict=True)

    args.run_dir = args.run_dir.resolve()
    args.eval_root = args.eval_root.resolve()
    if is_main:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        args.eval_root.mkdir(parents=True, exist_ok=True)
    barrier()

    train_records = load_records(protocol, dataset_name)
    dev_records = load_records(protocol, "target_dev")
    train_dataset = LineDataset(train_records, bundle.input_size, char_to_id)
    dev_dataset = LineDataset(dev_records, bundle.input_size, char_to_id)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    policy = BATCH_POLICY[args.model]
    generator = torch.Generator().manual_seed(seed + rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=policy["per_gpu"],
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_lines,
        worker_init_fn=worker_seed,
        generator=generator,
        drop_last=False,
    )
    dev_loader = None
    if is_main:
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=policy["eval"],
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_lines,
        )

    bundle.model.to(device)
    model: torch.nn.Module = bundle.model
    if world > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    accumulation = int(policy["accumulate"])
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = optimizer_steps_per_epoch * max_epochs
    warmup_steps = max(round(total_steps * 0.05), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: scheduler_lambda(step, total_steps, warmup_steps),
    )
    amp_enabled = amp_enabled_for_model(args.model)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    best_cer = float("inf")
    best_epoch = 0
    bad_evals = 0
    global_step = 0
    latest_path = args.run_dir / "latest.pth"
    if latest_path.is_file() and not args.resume:
        raise FileExistsError(
            f"Existing stage checkpoint requires --resume or explicit replacement: "
            f"{latest_path}"
        )
    if args.resume and latest_path.is_file():
        resume = torch.load(latest_path, map_location="cpu")
        if resume.get("protocol_sha256") != protocol_sha256:
            raise ValueError("Resume checkpoint belongs to a stale protocol")
        if resume.get("implementation_sha256") != implementation_sha256:
            raise ValueError("Resume checkpoint belongs to stale comparison code")
        if resume.get("model") != args.model or resume.get("stage") != args.stage:
            raise ValueError("Resume checkpoint model/stage mismatch")
        unwrap(model).load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        best_cer = float(resume["best_clean_dev_macro_cer"])
        best_epoch = int(resume["best_epoch"])
        bad_evals = int(resume["bad_evals"])
        global_step = int(resume["global_step"])

    config = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "model": args.model,
        "stage": args.stage,
        "dataset": dataset_name,
        "dataset_rows": len(train_dataset),
        "dev_rows": len(dev_dataset),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "public_checkpoint": str(public_checkpoint),
        "input_size_hw": list(bundle.input_size),
        "max_label_length": int(protocol["max_label_length"]),
        "world_size": world,
        "per_gpu_batch": policy["per_gpu"],
        "gradient_accumulation": accumulation,
        "global_effective_batch": policy["per_gpu"] * accumulation * world,
        "max_epochs": max_epochs,
        "eval_every": eval_every,
        "patience_evals": patience,
        "min_epochs": min_epochs,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "scheduler": "linear_warmup_cosine_decay_by_optimizer_step",
        "total_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "amp_enabled": amp_enabled,
        "precision_policy": (
            "fp32_crnn_ctc" if not amp_enabled else "amp_forward_fp32_ctc_loss"
        ),
        "selection_metric": "clean_dev_macro_cer",
        "seed": seed,
        "test_evaluated": False,
    }
    if is_main:
        (args.run_dir / "training_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    log_path = args.run_dir / "train_metrics.jsonl"
    stop = False
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, max_epochs + 1):
        last_epoch = epoch
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = torch.zeros((), device=device)
        epoch_samples = torch.zeros((), device=device)
        epoch_started = time.time()
        num_batches = len(train_loader)
        group_size = accumulation
        for batch_index, batch in enumerate(train_loader):
            if batch_index % accumulation == 0:
                group_size = min(accumulation, num_batches - batch_index)
            update_now = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == num_batches
            )
            sync_context = (
                contextlib.nullcontext()
                if update_now or not isinstance(model, DistributedDataParallel)
                else model.no_sync()
            )
            images = batch["images"].to(device, non_blocking=True)
            token_rows = batch["token_rows"]
            with sync_context:
                with torch.cuda.amp.autocast(
                    enabled=amp_enabled, dtype=torch.float16
                ):
                    loss = training_objective(
                        args.model,
                        model,
                        images,
                        token_rows,
                        int(protocol["max_label_length"]),
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch} batch={batch_index}: {loss}"
                    )
                scaler.scale(loss / group_size).backward()
            epoch_loss_sum += loss.detach() * images.shape[0]
            epoch_samples += images.shape[0]
            if update_now:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_spec["gradient_clip_norm"])
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch={epoch} batch={batch_index}"
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        if world > 1:
            dist.all_reduce(epoch_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_samples, op=dist.ReduceOp.SUM)
        train_loss = float((epoch_loss_sum / epoch_samples).cpu())
        evaluated = epoch % eval_every == 0 or epoch == max_epochs
        metrics = None
        improved = False
        if is_main and evaluated:
            assert dev_loader is not None
            metrics = evaluate_clean_dev(
                args.model,
                unwrap(model),
                dev_loader,
                characters,
                device,
                args.eval_root / f"epoch_{epoch:04d}",
            )
            cer = float(metrics["clean_dev_macro_cer"])
            if cer < best_cer - 1e-4:
                best_cer = cer
                best_epoch = epoch
                bad_evals = 0
                improved = True
                torch.save(
                    {
                        "model_state_dict": unwrap(model).state_dict(),
                        "protocol_id": protocol["protocol_id"],
                        "protocol_sha256": protocol_sha256,
                        "implementation_sha256": implementation_sha256,
                        "model": args.model,
                        "stage": args.stage,
                        "epoch": epoch,
                        "clean_dev_macro_cer": cer,
                        "source_checkpoint_sha256": sha256_file(source_checkpoint),
                        "test_evaluated": False,
                    },
                    args.run_dir / "best_clean_dev_macro_cer.pth",
                )
                shutil.copy2(
                    args.eval_root / f"epoch_{epoch:04d}" / "metrics.json",
                    args.run_dir / "best_clean_dev_macro_cer.json",
                )
            else:
                bad_evals += 1
            if epoch >= min_epochs and bad_evals >= patience:
                stop = True

        if world > 1:
            state = torch.tensor(
                [best_cer, float(best_epoch), float(bad_evals), float(stop)],
                dtype=torch.float64,
                device=device,
            )
            dist.broadcast(state, src=0)
            best_cer = float(state[0].item())
            best_epoch = int(state[1].item())
            bad_evals = int(state[2].item())
            stop = bool(state[3].item())

        if is_main:
            epoch_row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - epoch_started,
                "evaluated": evaluated,
                "clean_dev_macro_cer": (
                    metrics["clean_dev_macro_cer"] if metrics else None
                ),
                "improved": improved,
                "best_epoch": best_epoch,
                "best_clean_dev_macro_cer": best_cer,
                "bad_evals": bad_evals,
                "early_stop": stop,
            }
            with log_path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(epoch_row, ensure_ascii=False) + "\n")
            torch.save(
                {
                    "model_state_dict": unwrap(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "protocol_id": protocol["protocol_id"],
                    "protocol_sha256": protocol_sha256,
                    "implementation_sha256": implementation_sha256,
                    "model": args.model,
                    "stage": args.stage,
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_epoch": best_epoch,
                    "best_clean_dev_macro_cer": best_cer,
                    "bad_evals": bad_evals,
                    "test_evaluated": False,
                },
                latest_path,
            )
            print(json.dumps(epoch_row, ensure_ascii=False), flush=True)
        barrier()
        if stop:
            break

    if is_main:
        if best_epoch == 0:
            raise RuntimeError("No Clean Dev checkpoint was selected")
        summary = {
            "status": "complete",
            "model": args.model,
            "stage": args.stage,
            "best_epoch": best_epoch,
            "best_clean_dev_macro_cer": best_cer,
            "last_epoch": last_epoch,
            "early_stopped": stop,
            "best_checkpoint": str(args.run_dir / "best_clean_dev_macro_cer.pth"),
            "test_evaluated": False,
        }
        (args.run_dir / "stage_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
