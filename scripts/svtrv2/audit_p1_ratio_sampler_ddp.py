#!/usr/bin/env python3
"""Audit P1 RatioSampler coverage and step parity without allocating GPUs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def digest_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def close_lmdb_dataset(dataset) -> None:
    """Release read-only LMDB handles before simulating the next DDP rank."""

    errors = []
    for dataset_id, lmdb_set in getattr(dataset, "lmdb_sets", {}).items():
        transaction = lmdb_set.get("txn")
        environment = lmdb_set.get("env")
        try:
            if transaction is not None:
                transaction.abort()
                lmdb_set["txn"] = None
        except Exception as error:  # pragma: no cover - defensive cleanup
            errors.append(f"txn[{dataset_id}]: {error}")
        try:
            if environment is not None:
                environment.close()
                lmdb_set["env"] = None
        except Exception as error:  # pragma: no cover - defensive cleanup
            errors.append(f"env[{dataset_id}]: {error}")
    gc.collect()
    if errors:
        raise RuntimeError("Failed to close simulated-rank LMDBs: " + "; ".join(errors))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config_path = args.config.resolve()
    if args.world_size < 1:
        raise ValueError("world-size must be positive")

    openocr_root = root / "third_party" / "OpenOCR"
    sys.path.insert(0, str(openocr_root))
    sys.path.insert(0, str(openocr_root / "tools"))

    import torch
    from tools.data import build_dataloader
    from tools.data.ratio_sampler import SAMPLER_PROTOCOL
    from tools.engine.config import Config
    from tools.utils.logging import get_logger

    cfg = Config(str(config_path)).cfg
    declared_sampler_protocol = cfg["Global"].get("sampler_protocol")
    if declared_sampler_protocol != SAMPLER_PROTOCOL:
        raise RuntimeError(
            "Config/runtime sampler protocol mismatch: "
            f"config={declared_sampler_protocol!r}, runtime={SAMPLER_PROTOCOL!r}"
        )
    cfg["Global"]["distributed"] = True
    cfg["Train"]["loader"]["num_workers"] = 0
    seed = int(cfg["Global"].get("seed", 48))
    logger = get_logger("p1_ratio_sampler_ddp_audit")

    rank_reports = []
    batch_specs = []
    epoch_batch_specs = []
    sampled_sets = []
    traversal_hashes = []
    ratio_hashes = []
    dataset_sizes = []

    for rank in range(args.world_size):
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
                return_value=args.world_size,
            ),
            patch(
                "tools.data.ratio_sampler.torch.distributed.get_rank",
                return_value=rank,
            ),
        ):
            dataloader = build_dataloader(
                cfg,
                "Train",
                logger,
                task="rec",
            )

        sampler = dataloader.batch_sampler
        if getattr(sampler, "sampler_protocol", None) != SAMPLER_PROTOCOL:
            raise RuntimeError(
                f"Unexpected sampler protocol on rank {rank}: "
                f"{getattr(sampler, 'sampler_protocol', None)!r}"
            )
        batches = sampler.batch_list
        specs = [
            (int(batch[0][0]), int(batch[0][1]), len(batch))
            for batch in batches
        ]
        sampled = {
            int(sample[2])
            for batch in batches
            for sample in batch
        }
        dataset = dataloader.dataset
        dataset_size = len(dataset)
        dataset_sizes.append(dataset_size)
        traversal_hashes.append(digest_array(dataset.data_idx_order_list))
        ratio_hashes.append(digest_array(dataset.wh_ratio))
        batch_specs.append(specs)
        rank_epoch_specs = {}
        rank_epoch_replay_hashes = {}
        for epoch in (1, 2):
            sampler.set_epoch(epoch)
            yielded = list(iter(sampler))
            first_hash = hashlib.sha256(
                json.dumps(yielded, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            sampler.set_epoch(epoch)
            replayed = list(iter(sampler))
            replay_hash = hashlib.sha256(
                json.dumps(replayed, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if replay_hash != first_hash:
                raise RuntimeError(
                    "RatioSampler epoch replay is not deterministic: "
                    f"rank={rank}, epoch={epoch}"
                )
            rank_epoch_specs[str(epoch)] = [
                (int(batch[0][0]), int(batch[0][1]), len(batch))
                for batch in yielded
            ]
            rank_epoch_replay_hashes[str(epoch)] = first_hash
        epoch_batch_specs.append(rank_epoch_specs)
        sampled_sets.append(sampled)
        rank_reports.append(
            {
                "rank": rank,
                "dataset_samples": dataset_size,
                "sampler_batches": len(batches),
                "unique_sampled_indices": len(sampled),
                "assigned_with_batch_padding": sum(len(x) for x in batches),
                "distributed_bucket_padding": int(
                    sampler.distributed_padding
                ),
                "traversal_sha256": traversal_hashes[-1],
                "ratio_sha256": ratio_hashes[-1],
                "deterministic_epoch_replay_sha256": rank_epoch_replay_hashes,
            }
        )
        # RatioDataSetTVResize keeps long-lived LMDB transactions. This audit
        # simulates multiple ranks sequentially inside one process, unlike
        # real DDP where each rank has its own process. Explicitly close the
        # current rank before opening the same environments for the next rank.
        close_lmdb_dataset(dataset)
        del sampler, dataset, dataloader
        gc.collect()

    if len(set(dataset_sizes)) != 1:
        raise RuntimeError(f"Dataset lengths differ by rank: {dataset_sizes}")
    if len(set(traversal_hashes)) != 1:
        raise RuntimeError("LMDB traversal differs across DDP ranks")
    if len(set(ratio_hashes)) != 1:
        raise RuntimeError("Width/height ratios differ across DDP ranks")
    batch_counts = [len(item) for item in batch_specs]
    if len(set(batch_counts)) != 1:
        raise RuntimeError(f"DDP step counts differ by rank: {batch_counts}")
    reference_specs = batch_specs[0]
    for rank, specs in enumerate(batch_specs[1:], start=1):
        if specs != reference_specs:
            raise RuntimeError(
                f"DDP batch shapes differ between rank 0 and rank {rank}"
            )
    for epoch in ("1", "2"):
        reference_epoch_specs = epoch_batch_specs[0][epoch]
        for rank, rank_specs in enumerate(epoch_batch_specs[1:], start=1):
            if rank_specs[epoch] != reference_epoch_specs:
                raise RuntimeError(
                    "DDP yielded batch shapes differ after epoch shuffle: "
                    f"epoch={epoch}, ranks=0/{rank}"
                )

    sampled_union = set().union(*sampled_sets)
    expected = set(range(dataset_sizes[0]))
    missing = expected - sampled_union
    unexpected = sampled_union - expected
    if missing or unexpected:
        raise RuntimeError(
            "DDP sample coverage mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    overlap = sum(len(sampled_sets[i] & sampled_sets[j])
                  for i in range(args.world_size)
                  for j in range(i + 1, args.world_size))
    result = {
        "status": "P1_RATIO_SAMPLER_DDP_AUDIT_OK",
        "config": str(config_path),
        "sampler_protocol": SAMPLER_PROTOCOL,
        "world_size": args.world_size,
        "dataset_samples": dataset_sizes[0],
        "equal_step_count": True,
        "equal_batch_shapes": True,
        "equal_yielded_batch_shapes_epochs": [1, 2],
        "deterministic_epoch_replay": True,
        "full_union_coverage": True,
        "cross_rank_padding_overlap": overlap,
        "rank_reports": rank_reports,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
