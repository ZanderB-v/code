#!/usr/bin/env python3
"""Run isolated target-domain finetune seeds from frozen synthetic checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


RUN_ORDER = (
    ("b1", 20260811),
    ("m3", 20260811),
    ("b1", 20260812),
    ("m3", 20260812),
    ("m1", 20260811),
    ("m2", 20260811),
    ("m1", 20260812),
    ("m2", 20260812),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--ddp-gpus", default="0,1")
    parser.add_argument("--eval-gpu", type=int, default=0)
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--patience-evals", type=int, default=5)
    parser.add_argument("--min-epoch", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--master-port-base", type=int, default=29910)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], cwd: Path, log_path: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"Command failed with code {code}: {' '.join(command)}")


def find_run(plan: dict[str, Any], method: str, seed: int) -> dict[str, Any]:
    for item in plan["methods"][method]["new_runs"]:
        if int(item["seed"]) == seed:
            return item
    raise KeyError((method, seed))


def validate_all(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    for method, seed in RUN_ORDER:
        item = find_run(plan, method, seed)
        config_path = Path(item["config"])
        if not config_path.is_file() or sha256(config_path) != item["config_sha256"]:
            raise ValueError(f"Generated multiseed config hash drift: {config_path}")
        kind = plan["methods"][method]["config_kind"]
        validator = (
            "validate_p1_full_svtrv2_config.py"
            if kind == "full_svtrv2"
            else "validate_dual_order_method_config.py"
        )
        command = [
            sys.executable,
            str(args.root / "scripts" / "svtrv2" / validator),
            "--config",
            item["config"],
            "--expected-initialization",
            "checkpoint",
        ]
        run(
            command,
            args.root,
            args.log_dir / f"preflight_{method}_seed_{seed}.log",
        )
    print("ALL_METHOD_MULTISEED_CONFIG_PREFLIGHTS_OK", flush=True)


def final_summary_is_complete(
    path: Path,
    config_path: str,
    expected_config_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    summary = read_json(path)
    return (
        summary.get("test_policy") == "not_evaluated_during_model_development"
        and Path(summary.get("config", "")).name == Path(config_path).name
        and summary.get("config_sha256") == expected_config_sha256
        and summary.get("preprocess_protocol") == "P1_MSR_V3"
        and Path(summary.get("best_checkpoint", "")).name == "best_clean_dev_macro_cer.pth"
    )


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.plan = args.plan.resolve()
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    plan = read_json(args.plan)
    if plan.get("protocol_id") != "METHOD_TARGET_FINETUNE_MULTISEED_V1":
        raise ValueError("Unexpected multiseed protocol")
    if plan.get("training_scope") != "fixed_synthetic_pretrain_checkpoint_target_finetune_only":
        raise ValueError("This runner must not retrain the synthetic stage")
    if plan.get("status") != "passed" or plan.get("test_policy") != "not_evaluated":
        raise ValueError("Invalid multiseed plan status or test policy")
    if sorted(int(value) for value in plan.get("new_seeds", [])) != [20260811, 20260812]:
        raise ValueError("Unexpected multiseed seed set")
    validate_all(args, plan)
    if args.preflight_only:
        print("METHOD_MULTISEED_PREFLIGHT_ONLY_OK")
        return

    labels = args.root / "04_model_training" / "datasets" / "e1_target_only" / "labels"
    if not labels.is_dir():
        raise FileNotFoundError(labels)
    for index, (method, seed) in enumerate(RUN_ORDER):
        item = find_run(plan, method, seed)
        final_summary = Path(item["final_summary"])
        if not args.replace and final_summary_is_complete(
            final_summary,
            item["config"],
            item["config_sha256"],
        ):
            print(f"REUSE completed {method} seed={seed}: {final_summary}", flush=True)
            continue
        run_dir = Path(item["run_dir"])
        command = [
            sys.executable,
            str(args.root / "scripts" / "svtrv2" / "run_p1_rctc_stage.py"),
            "--root",
            str(args.root),
            "--stage",
            f"multiseed_{method}_seed_{seed}",
            "--config",
            item["config"],
            "--run-dir",
            str(run_dir),
            "--labels-dir",
            str(labels),
            "--label-prefix",
            "target",
            "--eval-prefix",
            item["eval_prefix"],
            "--expected-initialization",
            "checkpoint",
            "--config-kind",
            plan["methods"][method]["config_kind"],
            "--prediction-branch",
            plan["methods"][method]["prediction_branch"],
            "--max-epoch",
            str(args.max_epoch),
            "--eval-every",
            str(args.eval_every),
            "--patience-evals",
            str(args.patience_evals),
            "--min-epoch",
            str(args.min_epoch),
            "--min-delta",
            str(args.min_delta),
            "--ddp-gpus",
            args.ddp_gpus,
            "--eval-gpu",
            str(args.eval_gpu),
            "--master-port",
            str(args.master_port_base + index),
            "--log-dir",
            str(args.log_dir),
        ]
        if args.replace:
            command.append("--replace")
        elif run_dir.is_dir() and any(run_dir.glob("epoch_*.pth")):
            command.append("--resume-existing")
        run(
            command,
            args.root,
            args.log_dir / f"driver_{method}_seed_{seed}.log",
        )
        if not final_summary_is_complete(
            final_summary,
            item["config"],
            item["config_sha256"],
        ):
            raise RuntimeError(
                f"Training returned without a valid frozen-protocol summary: {final_summary}"
            )
    print("ALL_METHOD_TARGET_FINETUNE_SEEDS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
