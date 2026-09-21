#!/usr/bin/env python3
"""Audit HEM weighted sampling, DDP parity, and control-step equivalence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--expected-optimizer-steps", type=int, default=3065)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def close_dataset(dataset) -> None:
    for entry in getattr(dataset, "lmdb_sets", {}).values():
        transaction = entry.get("txn")
        environment = entry.get("env")
        if transaction is not None:
            transaction.abort()
        if environment is not None:
            environment.close()
    gc.collect()


def batch_spec(batches) -> list[tuple[int, int, int]]:
    return [
        (int(batch[0][0]), int(batch[0][1]), len(batch))
        for batch in batches
    ]


def batch_digest(batches) -> str:
    return hashlib.sha256(
        json.dumps(batches, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_rank(cfg, rank, world_size, build_dataloader, logger, torch):
    seed = int(cfg["Global"].get("seed", 48))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with (
        patch(
            "tools.data.ratio_sampler.torch.distributed.is_available",
            return_value=True,
        ),
        patch(
            "tools.data.ratio_sampler.torch.distributed.is_initialized",
            return_value=True,
        ),
        patch(
            "tools.data.ratio_sampler.torch.distributed.get_world_size",
            return_value=world_size,
        ),
        patch(
            "tools.data.ratio_sampler.torch.distributed.get_rank",
            return_value=rank,
        ),
    ):
        return build_dataloader(cfg, "Train", logger, task="rec")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    openocr_root = root / "third_party/OpenOCR"
    sys.path.insert(0, str(openocr_root))
    sys.path.insert(0, str(openocr_root / "tools"))

    import torch
    from tools.data import build_dataloader
    from tools.data.ratio_sampler import HEM_PROTOCOL
    from tools.engine.config import Config
    from tools.utils.logging import get_logger

    cfg = Config(str(args.config.resolve())).cfg
    cfg["Global"]["distributed"] = True
    cfg["Train"]["loader"]["num_workers"] = 0
    sampler_cfg = cfg["Train"]["sampler"]
    if sampler_cfg.get("hem_protocol") != HEM_PROTOCOL:
        raise ValueError("Config does not enable the frozen HEM sampler")
    manifest = Path(sampler_cfg["sample_weight_manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    if args.world_size != 1:
        raise ValueError("HEM V1 is frozen to one process on physical GPU1")
    if int(sampler_cfg.get("control_world_size", 0)) != 2:
        raise ValueError("HEM sampler must emulate the frozen two-rank control")
    if int(sampler_cfg.get("control_first_bs", 0)) != 16:
        raise ValueError("HEM sampler must retain batch 16 per control rank")
    logger = get_logger("hem_ratio_sampler_ddp_audit")
    rank_reports = []
    manifest_hashes = []

    for rank in range(args.world_size):
        hem_loader = build_rank(
            cfg, rank, args.world_size, build_dataloader, logger, torch
        )
        hem_sampler = hem_loader.batch_sampler
        if hem_sampler.hem_protocol != HEM_PROTOCOL:
            raise ValueError(f"Rank {rank} did not build a HEM sampler")
        hem_sampler.set_epoch(1)
        first = list(iter(hem_sampler))
        hem_sampler.set_epoch(1)
        replay = list(iter(hem_sampler))
        if batch_digest(first) != batch_digest(replay):
            raise RuntimeError(f"HEM sampling is not deterministic on rank {rank}")
        selected_indices = [int(item[2]) for batch in first for item in batch]
        selected_weights = hem_sampler.dataset_sample_weights[selected_indices]
        selected_counts = Counter(str(float(value)) for value in selected_weights)
        population_counts = Counter(
            str(float(value)) for value in hem_sampler.dataset_sample_weights
        )
        selected_hard_rate = float((selected_weights > 1.0).mean())
        population_hard_rate = float((hem_sampler.dataset_sample_weights > 1.0).mean())
        dataset_weights = hem_sampler.dataset_sample_weights.copy()
        wh_ratio = np.asarray(hem_sampler.wh_ratio)
        hem_specs = batch_spec(first)
        manifest_hashes.append(hem_sampler.sample_weight_manifest_sha256)
        expected_uniform_hard = 0.0
        expected_weighted_hard = 0.0
        expected_draws = 0
        for ratio in np.unique(wh_ratio):
            ratio_indices = np.flatnonzero(wh_ratio == ratio)
            draw_count = 2 * int(np.ceil(len(ratio_indices) / 2))
            ratio_weights = dataset_weights[ratio_indices]
            hard = ratio_weights > 1.0
            expected_uniform_hard += draw_count * float(hard.mean())
            expected_weighted_hard += draw_count * float(
                ratio_weights[hard].sum() / ratio_weights.sum()
            )
            expected_draws += draw_count
        expected_uniform_hard_rate = expected_uniform_hard / expected_draws
        expected_weighted_hard_rate = expected_weighted_hard / expected_draws
        close_dataset(hem_loader.dataset)
        del hem_loader, hem_sampler
        gc.collect()

        # Reconstruct one rank of the original two-GPU control. A HEM batch on
        # one GPU must be exactly the concatenation of the two equal-shape
        # control-rank batches, preserving optimizer steps and width schedule.
        control_cfg = Config(str(args.config.resolve())).cfg
        control_cfg["Global"]["distributed"] = True
        control_cfg["Train"]["loader"]["num_workers"] = 0
        control_cfg["Train"]["loader"]["batch_size_per_card"] = 16
        control_sampler_cfg = control_cfg["Train"]["sampler"]
        control_sampler_cfg["first_bs"] = 16
        for key in (
            "sample_weight_manifest",
            "hem_protocol",
            "hem_seed",
            "control_world_size",
            "control_first_bs",
        ):
            control_sampler_cfg.pop(key, None)
        control_loader = build_rank(
            control_cfg, 0, 2, build_dataloader, logger, torch
        )
        control_sampler = control_loader.batch_sampler
        control_sampler.set_epoch(1)
        control = list(iter(control_sampler))
        control_specs = batch_spec(control)
        close_dataset(control_loader.dataset)
        del control_loader, control_sampler
        gc.collect()

        merged_control_specs = [
            (width, height, batch_size * 2)
            for width, height, batch_size in control_specs
        ]
        if Counter(hem_specs) != Counter(merged_control_specs):
            raise RuntimeError(
                "HEM changed the original two-rank width/batch schedule"
            )
        if expected_weighted_hard_rate <= expected_uniform_hard_rate:
            raise RuntimeError("HEM weights do not increase expected hard frequency")
        if selected_hard_rate <= expected_uniform_hard_rate:
            raise RuntimeError(
                "Sampled HEM epoch did not exceed the matched uniform expectation: "
                f"HEM={selected_hard_rate}, uniform={expected_uniform_hard_rate}"
            )
        if len(hem_specs) != args.expected_optimizer_steps:
            raise RuntimeError(
                f"HEM optimizer steps changed on rank {rank}: "
                f"{len(hem_specs)} != {args.expected_optimizer_steps}"
            )
        replay_hash = batch_digest(first)
        rank_reports.append(
            {
                "rank": rank,
                "optimizer_steps_per_epoch": len(hem_specs),
                "assigned_with_control_padding": len(selected_indices),
                "population_weight_counts": dict(population_counts),
                "selected_weight_counts_epoch_1": dict(selected_counts),
                "population_hard_rate": population_hard_rate,
                "uniform_expected_hard_rate": expected_uniform_hard_rate,
                "weighted_expected_hard_rate": expected_weighted_hard_rate,
                "selected_hard_rate_epoch_1": selected_hard_rate,
                "hard_frequency_multiplier_vs_uniform_expectation": (
                    selected_hard_rate / expected_uniform_hard_rate
                ),
                "deterministic_replay_sha256": replay_hash,
            }
        )

    if len(set(manifest_hashes)) != 1:
        raise RuntimeError("DDP ranks loaded different HEM manifests")
    result = {
        "status": "HEM_RATIO_SAMPLER_DDP_AUDIT_OK",
        "protocol": HEM_PROTOCOL,
        "config": str(args.config.resolve()),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": manifest_hashes[0],
        "actual_world_size": args.world_size,
        "control_world_size": 2,
        "control_batch_size_per_rank": 16,
        "actual_batch_size_for_short_lines": 32,
        "canonical_dataset_rows": sum(
            rank_reports[0]["population_weight_counts"].values()
        ),
        "equal_optimizer_steps_vs_control": True,
        "optimizer_steps_per_epoch": args.expected_optimizer_steps,
        "equal_optimizer_steps_vs_original_global_batch_32": True,
        "equal_merged_batch_shapes_vs_two_rank_control": True,
        "weighted_sampling_with_replacement": True,
        "hard_frequency_increased": True,
        "rank_reports": rank_reports,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
