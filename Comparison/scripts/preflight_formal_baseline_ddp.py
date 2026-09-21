#!/usr/bin/env python3
"""Two-GPU long-line optimizer/checkpoint preflight for one public baseline."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

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
    sha256_file,
)
from formal_baseline_runtime import (
    LineDataset,
    amp_enabled_for_model,
    collate_lines,
    training_objective,
)
from pretrained_models import build_model_bundle
from train_pretrained_baseline import BATCH_POLICY, load_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(BATCH_POLICY))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != 2:
        raise ValueError(f"Formal DDP preflight requires exactly two ranks, got {world}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}")

    protocol = load_protocol(args.protocol)
    protocol_path = args.protocol or (
        PROJECT_ROOT / "Comparison/pretrained_baselines_v1/protocol.json"
    )
    protocol_sha256 = sha256_file(protocol_path)
    implementation_sha256 = formal_implementation_hashes()
    characters = read_dictionary(
        PROJECT_ROOT / protocol["dictionary"],
        bool(protocol["use_space_char"]),
    )
    if len(characters) != int(protocol["expected_recognition_symbol_count"]):
        raise ValueError("Runtime recognition-symbol count mismatch")
    char_to_id = {character: index + 1 for index, character in enumerate(characters)}
    model_spec = protocol["models"][args.model]
    bundle = build_model_bundle(
        args.model,
        PROJECT_ROOT / model_spec["checkpoint"],
        len(characters),
        int(protocol["max_label_length"]),
    )
    initialization = (
        PROJECT_ROOT
        / f"Comparison/pretrained_baselines_v1/outputs/{args.model}_4891_initialization.pth"
    )
    state, initialization_payload = load_state(initialization)
    if initialization_payload.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Audited initialization belongs to a stale protocol")
    if initialization_payload.get("implementation_sha256") != implementation_sha256:
        raise ValueError("Audited initialization belongs to stale comparison code")
    bundle.model.load_state_dict(state, strict=True)
    bundle.model.to(device).train()
    model = DistributedDataParallel(
        bundle.model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    all_target = load_records(protocol, "target_train")
    selected = []
    for language in ("zh", "ug", "kk"):
        language_rows = sorted(
            (row for row in all_target if row.language == language),
            key=lambda row: len(row.train_text),
            reverse=True,
        )
        selected.extend(language_rows[:8])
    dataset = LineDataset(selected, bundle.input_size, char_to_id)
    sampler = DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=False, drop_last=False
    )
    loader = DataLoader(
        dataset,
        batch_size=min(BATCH_POLICY[args.model]["per_gpu"], 2),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_lines,
        pin_memory=True,
    )
    batch = next(iter(loader))
    images = batch["images"].to(device, non_blocking=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    amp_enabled = amp_enabled_for_model(args.model)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
        loss = training_objective(
            args.model,
            model,
            images,
            batch["token_rows"],
            int(protocol["max_label_length"]),
        )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite preflight loss: {loss}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError(f"Non-finite preflight gradient norm: {grad_norm}")
    nonzero_gradients = sum(
        int(torch.count_nonzero(parameter.grad).item())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if nonzero_gradients == 0:
        raise ValueError("All DDP preflight gradients are zero")
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    checksum = torch.zeros((), device=device, dtype=torch.float64)
    with torch.no_grad():
        for parameter in model.module.parameters():
            checksum += parameter.detach().double().sum()
    gathered = [torch.zeros_like(checksum) for _ in range(world)]
    dist.all_gather(gathered, checksum)
    checksums = [float(item.cpu()) for item in gathered]
    if max(checksums) - min(checksums) > 1e-6:
        raise ValueError(f"DDP parameters diverged after one step: {checksums}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{args.model}_ddp_roundtrip.pth"
    if rank == 0:
        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "model": args.model,
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha256,
                "implementation_sha256": implementation_sha256,
                "test_evaluated": False,
            },
            checkpoint_path,
        )
    dist.barrier()
    roundtrip, payload = load_state(checkpoint_path)
    model.module.load_state_dict(roundtrip, strict=True)
    dist.barrier()

    if rank == 0:
        report = {
            "status": "passed",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha256,
            "implementation_sha256": implementation_sha256,
            "model": args.model,
            "world_size": world,
            "input_shape_per_rank": list(images.shape),
            "long_label_lengths_per_rank0_batch": [
                len(text) for text in batch["train_texts"]
            ],
            "loss": float(loss.detach().cpu()),
            "amp_enabled": amp_enabled,
            "precision_policy": (
                "fp32_crnn_ctc" if not amp_enabled else "amp_forward_fp32_ctc_loss"
            ),
            "gradient_norm": float(grad_norm.detach().cpu()),
            "nonzero_gradient_numel": nonzero_gradients,
            "parameter_checksums": checksums,
            "optimizer_step": True,
            "scheduler_step": scheduler.last_epoch,
            "checkpoint_roundtrip_strict": True,
            "test_evaluated": False,
            "errors": [],
        }
        report_path = args.output_dir / f"{args.model}_formal_ddp_preflight.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        checkpoint_path.unlink(missing_ok=True)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"FORMAL_DDP_PREFLIGHT_OK model={args.model}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
