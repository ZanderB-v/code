#!/usr/bin/env python3
"""Regression test for checkpointed OneCycleLR segment control."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import torch

from run_p1_rctc_stage import validate_checkpoint_horizon


MAX_EPOCH = 50
STEPS_PER_EPOCH = 10


def make_training_state() -> tuple[
    torch.nn.Parameter,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.OneCycleLR,
]:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-5,
        total_steps=MAX_EPOCH * STEPS_PER_EPOCH,
        cycle_momentum=False,
    )
    return parameter, optimizer, scheduler


def advance(
    parameter: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    steps: int,
) -> list[float]:
    lrs: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        lrs.append(float(scheduler.get_last_lr()[0]))
    return lrs


def main() -> None:
    full_parameter, full_optimizer, full_scheduler = make_training_state()
    full_lrs = advance(
        full_parameter,
        full_optimizer,
        full_scheduler,
        MAX_EPOCH * STEPS_PER_EPOCH,
    )

    parameter, optimizer, scheduler = make_training_state()
    split_step = 40 * STEPS_PER_EPOCH
    advance(parameter, optimizer, scheduler, split_step)
    checkpoint = {
        "epoch": 40,
        "global_step": split_step,
        "state_dict": {"parameter": parameter.detach().clone()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "epoch_40.pth"
        torch.save(checkpoint, checkpoint_path)
        validate_checkpoint_horizon(
            {
                "epoch": checkpoint["epoch"],
                "global_step": checkpoint["global_step"],
                "scheduler_total_steps": checkpoint["scheduler"]["total_steps"],
            },
            max_epoch=MAX_EPOCH,
            checkpoint_path=checkpoint_path,
        )

    resumed_parameter, resumed_optimizer, resumed_scheduler = make_training_state()
    resumed_parameter.data.copy_(checkpoint["state_dict"]["parameter"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_scheduler.load_state_dict(checkpoint["scheduler"])
    resumed_lrs = advance(
        resumed_parameter,
        resumed_optimizer,
        resumed_scheduler,
        5 * STEPS_PER_EPOCH,
    )
    expected_lrs = full_lrs[split_step:split_step + 5 * STEPS_PER_EPOCH]
    if resumed_lrs != expected_lrs:
        raise AssertionError("Segmented resume changed the OneCycleLR trajectory")

    invalid_detected = False
    try:
        validate_checkpoint_horizon(
            {
                "epoch": 5,
                "global_step": 5 * STEPS_PER_EPOCH,
                "scheduler_total_steps": 5 * STEPS_PER_EPOCH,
            },
            max_epoch=MAX_EPOCH,
            checkpoint_path=Path("invalid_epoch_5.pth"),
        )
    except ValueError:
        invalid_detected = True
    if not invalid_detected:
        raise AssertionError("Shortened scheduler horizon was not rejected")

    print(
        "SEGMENTED_SCHEDULER_CONTROL_OK "
        f"total_steps={MAX_EPOCH * STEPS_PER_EPOCH} "
        f"resume_step={split_step}"
    )


if __name__ == "__main__":
    main()
