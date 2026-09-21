#!/usr/bin/env python3
"""CPU-only regression tests for HEM V1 weighted ratio sampling."""

from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import types
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np


class FakeDataset:
    def __init__(self) -> None:
        self.ds_width = True
        self.seed = 1
        self.wh_ratio = np.asarray([1] * 20 + [2] * 20 + [5] * 20 + [9] * 20)
        self.wh_ratio_sort = np.argsort(self.wh_ratio)
        permutation = np.asarray(list(reversed(range(1, 81))))
        self.data_idx_order_list = np.column_stack(
            [np.zeros(80, dtype=np.int64), permutation]
        )
        self.lmdb_sets = {0: {}}

    def __len__(self) -> int:
        return 80


def flatten(batches):
    return [int(item[2]) for batch in batches for item in batch]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    fake_torch = types.ModuleType("torch")
    fake_torch.distributed = types.SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
        get_world_size=lambda: 1,
        get_rank=lambda: 0,
    )
    fake_utils = types.ModuleType("torch.utils")
    fake_data = types.ModuleType("torch.utils.data")
    fake_data.Sampler = object
    fake_utils.data = fake_data
    fake_torch.utils = fake_utils
    sys.modules["torch"] = fake_torch
    sys.modules["torch.utils"] = fake_utils
    sys.modules["torch.utils.data"] = fake_data
    module_path = root / "third_party/OpenOCR/tools/data/ratio_sampler.py"
    spec = importlib.util.spec_from_file_location("hem_ratio_sampler_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    HEM_PROTOCOL = module.HEM_PROTOCOL
    RatioSampler = module.RatioSampler

    freeze_path = root / "scripts/svtrv2/freeze_hem_v1.py"
    freeze_spec = importlib.util.spec_from_file_location(
        "hem_freeze_under_test", freeze_path
    )
    assert freeze_spec is not None and freeze_spec.loader is not None
    freeze_module = importlib.util.module_from_spec(freeze_spec)
    freeze_spec.loader.exec_module(freeze_module)
    canonical, already_ordered = freeze_module.canonicalize_manifest_order(
        [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        [
            {"sample_id": "c", "sample_weight": 1.5},
            {"sample_id": "a", "sample_weight": 1.0},
            {"sample_id": "b", "sample_weight": 2.0},
        ],
    )
    assert [row["sample_id"] for row in canonical] == ["a", "b", "c"]
    assert already_ordered is False

    with tempfile.TemporaryDirectory() as directory:
        manifest = Path(directory) / "weights.jsonl"
        with manifest.open("w", encoding="utf-8") as handle:
            for index in range(80):
                weight = 2.0 if index < 8 else (1.5 if index < 16 else 1.0)
                handle.write(
                    json.dumps(
                        {"sample_id": str(index), "sample_weight": weight}
                    )
                    + "\n"
                )
        kwargs = dict(
            scales=[[128, 32]],
            first_bs=8,
            fix_bs=False,
            divided_factor=[4, 16],
            is_training=True,
            max_ratio=40,
        )
        dataset = FakeDataset()
        standard = RatioSampler(dataset, **kwargs)
        weighted = RatioSampler(
            dataset,
            **kwargs,
            sample_weight_manifest=str(manifest),
            hem_protocol=HEM_PROTOCOL,
            hem_seed=17,
        )
        standard.set_epoch(3)
        standard_batches = list(iter(standard))
        weighted.set_epoch(3)
        first = list(iter(weighted))
        weighted.set_epoch(3)
        replay = list(iter(weighted))
        assert first == replay
        assert len(first) == len(standard_batches)
        assert [
            (batch[0][0], batch[0][1], len(batch)) for batch in first
        ] == [
            (batch[0][0], batch[0][1], len(batch)) for batch in standard_batches
        ]
        sampled = []
        for epoch in range(20):
            weighted.set_epoch(epoch)
            sampled.extend(flatten(list(iter(weighted))))
        sampled_weights = weighted.dataset_sample_weights[sampled]
        selected_hard_rate = float((sampled_weights > 1).mean())
        population_hard_rate = float((weighted.dataset_sample_weights > 1).mean())
        control_weights = weighted.dataset_sample_weights[
            flatten(standard_batches)
        ]
        control_hard_rate = float((control_weights > 1).mean())
        assert selected_hard_rate > control_hard_rate, (
            selected_hard_rate,
            control_hard_rate,
            population_hard_rate,
            np.unique(sampled_weights, return_counts=True),
        )
        assert weighted.hem_selection_summary()["manifest_sha256"]

        # One physical GPU may carry the unchanged global batch by merging the
        # two equal-width batches that the frozen two-rank control would use.
        emulated = RatioSampler(
            FakeDataset(),
            **{**kwargs, "first_bs": 16},
            sample_weight_manifest=str(manifest),
            hem_protocol=HEM_PROTOCOL,
            hem_seed=17,
            control_world_size=2,
            control_first_bs=8,
        )
        emulated.set_epoch(2)
        emulated_batches = list(iter(emulated))
        with (
            patch.object(module.torch.distributed, "is_available", return_value=True),
            patch.object(module.torch.distributed, "is_initialized", return_value=True),
            patch.object(module.torch.distributed, "get_world_size", return_value=2),
            patch.object(module.torch.distributed, "get_rank", return_value=0),
        ):
            control_rank = RatioSampler(FakeDataset(), **kwargs)
        control_rank.set_epoch(2)
        control_rank_batches = list(iter(control_rank))
        emulated_specs = Counter(
            (batch[0][0], batch[0][1], len(batch)) for batch in emulated_batches
        )
        merged_control_specs = Counter(
            (batch[0][0], batch[0][1], len(batch) * 2)
            for batch in control_rank_batches
        )
        assert emulated_specs == merged_control_specs
        assert len(emulated_batches) == len(control_rank_batches)

        unweighted_emulation = RatioSampler(
            FakeDataset(),
            **{**kwargs, "first_bs": 16},
            control_world_size=2,
            control_first_bs=8,
        )
        unweighted_specs = Counter(
            (batch[0][0], batch[0][1], len(batch))
            for batch in unweighted_emulation.batch_list
        )
        assert unweighted_specs == merged_control_specs
        assert len(unweighted_emulation) == len(control_rank_batches)
        assert set(flatten(unweighted_emulation.batch_list)) == set(range(80))

        rank_specs = []
        for rank in (0, 1):
            with (
                patch.object(module.torch.distributed, "is_available", return_value=True),
                patch.object(module.torch.distributed, "is_initialized", return_value=True),
                patch.object(module.torch.distributed, "get_world_size", return_value=2),
                patch.object(module.torch.distributed, "get_rank", return_value=rank),
            ):
                sampler = RatioSampler(
                    FakeDataset(),
                    **kwargs,
                    sample_weight_manifest=str(manifest),
                    hem_protocol=HEM_PROTOCOL,
                    hem_seed=17,
                )
            sampler.set_epoch(2)
            batches = list(iter(sampler))
            rank_specs.append(
                [(batch[0][0], batch[0][1], len(batch)) for batch in batches]
            )
        assert rank_specs[0] == rank_specs[1]
    print(
        {
            "status": "HEM_TRAINING_V1_TESTS_OK",
            "weighted_sampling": True,
            "deterministic_replay": True,
            "control_step_parity": True,
            "single_gpu_two_rank_schedule_emulation": True,
            "unweighted_single_gpu_topology_emulation": True,
            "manifest_canonicalized_by_sample_id": True,
            "ddp_shape_parity": True,
            "test_evaluated": False,
        }
    )


if __name__ == "__main__":
    main()
