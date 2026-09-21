#!/usr/bin/env python3
"""Run one OpenOCR stage to an exact optimizer-update boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--total-updates", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path, total_updates: int) -> dict:
    import torch

    payload = torch.load(path, map_location=torch.device("cpu"))
    global_step = int(payload.get("global_step", -1))
    scheduler = payload.get("scheduler") or {}
    scheduler_last_epoch = int(scheduler.get("last_epoch", -1))
    scheduler_total_steps = int(scheduler.get("total_steps", -1))
    if global_step != total_updates:
        raise ValueError(
            f"Checkpoint global_step {global_step} != {total_updates}: {path}"
        )
    if scheduler_last_epoch != total_updates:
        raise ValueError(
            "Scheduler and optimizer steps diverged: "
            f"last_epoch={scheduler_last_epoch}, updates={total_updates}"
        )
    if scheduler_total_steps != total_updates:
        raise ValueError(
            "Scheduler horizon mismatch: "
            f"total_steps={scheduler_total_steps}, updates={total_updates}"
        )
    return {
        "epoch_field": int(payload.get("epoch", -1)),
        "global_step": global_step,
        "scheduler_last_epoch": scheduler_last_epoch,
        "scheduler_total_steps": scheduler_total_steps,
    }


def reusable(args: argparse.Namespace, checkpoint: Path) -> dict | None:
    if not args.summary.is_file() or not checkpoint.is_file():
        return None
    summary = json.loads(args.summary.read_text(encoding="utf-8-sig"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    source = Path(str(config.get("Global", {}).get("pretrained_model") or ""))
    if not source.is_file():
        return None
    expected = {
        "status": "FIXED_OPTIMIZER_STEP_STAGE_OK",
        "total_optimizer_updates": args.total_updates,
        "config_sha256": sha256(args.config),
        "checkpoint_sha256": sha256(checkpoint),
        "source_checkpoint_sha256": sha256(source),
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        return None
    verify_checkpoint(checkpoint, args.total_updates)
    return summary


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.config = args.config.resolve()
    args.run_dir = args.run_dir.resolve()
    args.log = args.log.resolve()
    args.summary = args.summary.resolve()
    if args.total_updates < 2:
        raise ValueError("--total-updates must be at least 2")
    if not args.config.is_file():
        raise FileNotFoundError(args.config)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    source_checkpoint = Path(str(config.get("Global", {}).get("pretrained_model") or ""))
    if not source_checkpoint.is_file():
        raise FileNotFoundError(
            f"Configured initialization checkpoint is missing: {source_checkpoint}"
        )
    configured_updates = int(
        config.get("Global", {}).get("max_optimizer_steps", -1)
    )
    scheduler_updates = int(config.get("LRScheduler", {}).get("total_steps", -1))
    if configured_updates != args.total_updates or scheduler_updates != args.total_updates:
        raise ValueError(
            "Config is not locked to the requested update boundary: "
            f"Global={configured_updates}, scheduler={scheduler_updates}, "
            f"requested={args.total_updates}"
        )
    checkpoint = args.run_dir / f"step_{args.total_updates}.pth"
    if not args.replace:
        existing = reusable(args, checkpoint)
        if existing is not None:
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return

    if args.run_dir.exists():
        shutil.rmtree(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "tools/train_rec.py",
        "-c",
        str(args.config),
        "-o",
        f"Global.max_optimizer_steps={args.total_updates}",
        "Global.strict_lr_scheduler=true",
        "Global.strict_epoch_length=true",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    print("+ " + " ".join(command), flush=True)
    with args.log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(args.root / "third_party" / "OpenOCR"),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Training failed with code {return_code}; see {args.log}")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Training exited without exact-step checkpoint: {checkpoint}"
        )

    checkpoint_state = verify_checkpoint(checkpoint, args.total_updates)
    loss_matches = re.findall(
        r"(?:^|, )loss: ([0-9.eE+-]+)",
        args.log.read_text(encoding="utf-8-sig", errors="replace"),
    )
    terminal_logged_loss = float(loss_matches[-1]) if loss_matches else None
    # The generated config records this protocol value for provenance.
    steps_per_epoch = int(config["Global"]["steps_per_epoch_audit"])
    expected_epoch_num = int(math.ceil(args.total_updates / steps_per_epoch))
    if int(config["Global"]["epoch_num"]) != expected_epoch_num:
        raise ValueError("Configured epoch ceiling does not match audited steps")
    summary = {
        "status": "FIXED_OPTIMIZER_STEP_STAGE_OK",
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "physical_gpu": args.physical_gpu,
        "world_size": 1,
        "total_optimizer_updates": args.total_updates,
        "audited_steps_per_epoch": steps_per_epoch,
        "epoch_ceiling": expected_epoch_num,
        "terminal_partial_epoch_allowed": args.total_updates % steps_per_epoch != 0,
        "checkpoint_state": checkpoint_state,
        "terminal_logged_smoothed_loss": terminal_logged_loss,
        "train_log": str(args.log),
        "test_evaluated": False,
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
