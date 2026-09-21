#!/usr/bin/env python3
"""Regression tests for explicit-step B1 scale selection controls."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SVTRV2_SCRIPTS = ROOT / "scripts/svtrv2"
if str(SVTRV2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SVTRV2_SCRIPTS))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    return parser.parse_args()


def load_lr_module():
    path = ROOT / "third_party/OpenOCR/openrec/optimizer/lr.py"
    spec = importlib.util.spec_from_file_location("scale_control_lr", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trajectory(torch, module, epochs: int, steps_per_epoch: int) -> tuple[list[float], dict]:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=2.5e-5)
    scheduler = module.OneCycleLR(
        epochs=epochs,
        step_each_epoch=steps_per_epoch,
        lr=2.5e-5,
        total_steps=100,
        warmup_steps=10,
        cycle_momentum=False,
    )(optimizer)
    values = []
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        values.append(float(scheduler.get_last_lr()[0]))
    return values, scheduler.state_dict()


def test_format_preserving_config_lock() -> None:
    from p1_msr_protocol import DEFAULT_MAX_RATIO, make_rctc_config
    from run_b1_scale_step_control import PROTOCOL_ID_SCALE, lock_step_config

    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        config = temp / "scale.yml"
        config.write_text(
            make_rctc_config(
                root=ROOT,
                run_dir=temp / "run",
                project_name="scale_lock_regression",
                train_lmdbs=[temp / "train"],
                eval_lmdbs=[temp / "dev"],
                pretrained_model=temp / "union14m.pth",
                max_epoch=100,
                first_batch_size=32,
                num_workers=8,
                max_ratio=DEFAULT_MAX_RATIO,
                lr=5e-5,
                internal_eval_every=100000,
                seed=20260731,
            ),
            encoding="utf-8",
        )
        lock_step_config(
            config,
            steps_per_epoch=2307,
            total_steps=115350,
            warmup_steps=4614,
            fixed_terminal=True,
        )
        text = config.read_text(encoding="utf-8")
        required_literals = (
            "padding: False",
            "max_text_length: &max_text_length 120",
            "use_space_char: &use_space_char True",
        )
        missing = [token for token in required_literals if token not in text]
        if missing:
            raise AssertionError(f"P1 literal tokens were not preserved: {missing}")
        cfg = yaml.safe_load(text)
        if cfg["Global"]["scale_control_protocol"] != PROTOCOL_ID_SCALE:
            raise AssertionError("Scale-control protocol marker is missing")
        if cfg["Global"]["max_optimizer_steps"] != 115350:
            raise AssertionError("Exact optimizer-step boundary is missing")
        if cfg["LRScheduler"] != {
            "name": "OneCycleLR",
            "total_steps": 115350,
            "warmup_steps": 4614,
            "cycle_momentum": False,
        }:
            raise AssertionError("Explicit-step scheduler fields changed")
        loader = cfg["Train"]["loader"]
        if (
            loader.get("pin_memory") is not True
            or loader.get("persistent_workers") is not True
            or loader.get("prefetch_factor") != 4
        ):
            raise AssertionError("Train loader performance controls are missing")


def main() -> None:
    args = parse_args()
    test_format_preserving_config_lock()
    trainer = (ROOT / "third_party/OpenOCR/tools/engine/trainer.py").read_text(
        encoding="utf-8-sig"
    )
    required = (
        "max_optimizer_steps",
        "prefix=f'step_{global_step}'",
        "reached_step_boundary",
        "strict_lr_scheduler",
    )
    missing = [token for token in required if token not in trainer]
    if missing:
        raise AssertionError(f"Fixed-step trainer controls missing: {missing}")
    stage_runner = (
        ROOT / "scripts/svtrv2/run_p1_rctc_stage.py"
    ).read_text(encoding="utf-8-sig")
    if "if len(gpu_ids) == 1:" not in stage_runner:
        raise AssertionError("Single-GPU stages still require a DDP launcher")
    if not args.static_only:
        import torch

        module = load_lr_module()
        first, first_state = trajectory(
            torch, module, epochs=10, steps_per_epoch=10
        )
        second, second_state = trajectory(
            torch, module, epochs=25, steps_per_epoch=4
        )
        if first != second:
            raise AssertionError(
                "Explicit-step OneCycleLR still depends on epoch layout"
            )
        if (
            first_state["total_steps"] != 100
            or second_state["total_steps"] != 100
        ):
            raise AssertionError("Explicit scheduler horizon was not retained")
    result = {
        "status": "B1_SCALE_STEP_CONTROL_COMPONENT_TESTS_OK",
        "same_lr_trajectory_for_different_epoch_layouts": True,
        "total_steps": 100,
        "warmup_steps": 10,
        "trainer_exact_step_checkpoint": True,
        "p1_config_literals_preserved": True,
        "scheduler_runtime_tested": not args.static_only,
        "test_evaluated": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
