#!/usr/bin/env python3
"""Complete only the missing M3 alpha=.25 target fine-tuning stage."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


MODEL = "svtrv2_s_m3_alpha025_dual_order_s50_to_target"
SYNTHETIC_MODEL = "svtrv2_s_m3_alpha025_dual_order_s50"
REFERENCE_MODEL = "svtrv2_s_m3_alpha020_dual_order_s50_to_target"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--ddp-gpus", default="0")
    parser.add_argument("--eval-gpu", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=29975)
    parser.add_argument("--reference-world-size", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replace-target", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path, log: Path | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    if log is None:
        subprocess.run(command, cwd=str(cwd), check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
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
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def canonicalize_project_paths(value):
    if isinstance(value, dict):
        return {key: canonicalize_project_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonicalize_project_paths(item) for item in value]
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    marker = "/experiments/multilingual_meme_ocr/svtrv2_line_recognition"
    if marker in normalized:
        return "<ROOT>" + normalized.split(marker, 1)[1]
    return normalized


def controlled_config(config: dict) -> dict:
    value = copy.deepcopy(config)
    for key in (
        "output_dir",
        "project_name",
        "save_res_path",
        "pretrained_model",
        "method_variant",
        "model_protocol",
    ):
        if key in value.get("Global", {}):
            value["Global"][key] = "<CONTROLLED>"
    value["Loss"]["consistency_weight"] = "<TESTED_ALPHA>"
    # The original run used 2 GPUs x 16 samples. A single-GPU completion uses
    # 1 x 32 so the effective batch and optimizer-step schedule stay fixed.
    value["Train"]["sampler"]["first_bs"] = "<WORLD_SIZE_BATCH>"
    value["Train"]["sampler"]["control_world_size"] = "<CONTROL_TOPOLOGY>"
    value["Train"]["sampler"]["control_first_bs"] = "<CONTROL_TOPOLOGY>"
    value["Train"]["loader"]["batch_size_per_card"] = "<WORLD_SIZE_BATCH>"
    return canonicalize_project_paths(value)


def parse_gpu_ids(value: str) -> list[int]:
    try:
        gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"Invalid --ddp-gpus value: {value!r}") from error
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"--ddp-gpus must contain unique GPU ids: {value!r}")
    return gpu_ids


def set_train_batch_size(config: dict, batch_size: int) -> None:
    config["Train"]["sampler"]["first_bs"] = batch_size
    config["Train"]["loader"]["batch_size_per_card"] = batch_size


def read_sampler_audit(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if result.get("status") != "P1_RATIO_SAMPLER_DDP_AUDIT_OK":
        raise ValueError(f"Sampler audit failed: {path}")
    return result


def config_differences(left, right, prefix: str = "") -> list[str]:
    differences = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left:
                differences.append(f"{path}: missing in generated config")
            elif key not in right:
                differences.append(f"{path}: missing in reference config")
            else:
                differences.extend(config_differences(left[key], right[key], path))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differences.append(f"{prefix}: list length {len(left)} != {len(right)}")
            return differences
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(
                config_differences(left_item, right_item, f"{prefix}[{index}]")
            )
        return differences
    if left != right:
        differences.append(f"{prefix}: generated={left!r}, reference={right!r}")
    return differences


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    runs = root / "04_model_training" / "runs"
    configs = root / "04_model_training" / "configs"
    eval_reports = root / "04_model_training" / "eval_reports"

    source = runs / SYNTHETIC_MODEL / "best_clean_dev_macro_cer.pth"
    source_meta = runs / SYNTHETIC_MODEL / "best_clean_dev_macro_cer.json"
    reference_config = configs / f"{REFERENCE_MODEL}.yml"
    for path in (source, source_meta, reference_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_record = json.loads(source_meta.read_text(encoding="utf-8-sig"))
    expected_source_hash = source_record.get("copied_checkpoint_sha256")
    if not expected_source_hash or sha256(source) != expected_source_hash:
        raise ValueError("Transferred alpha=.25 synthetic best checkpoint hash mismatch")

    reference = yaml.safe_load(reference_config.read_text(encoding="utf-8-sig"))
    reference_batch_size = int(reference["Train"]["sampler"]["first_bs"])
    gpu_ids = parse_gpu_ids(args.ddp_gpus)
    target_world_size = len(gpu_ids)
    if args.reference_world_size < 1:
        raise ValueError("--reference-world-size must be positive")
    reference_global_batch = reference_batch_size * args.reference_world_size
    if reference_global_batch % target_world_size:
        raise ValueError(
            "Reference global batch is not divisible by the requested world size: "
            f"{reference_global_batch} / {target_world_size}"
        )
    batch_size = reference_global_batch // target_world_size
    workers = int(reference["Train"]["loader"]["num_workers"])
    learning_rate = float(reference["Optimizer"]["lr"])
    seed = int(reference["Global"].get("seed", 20260731))
    max_epoch = int(reference["Global"].get("epoch_num", 50))
    if seed != 20260731 or max_epoch != 50:
        raise ValueError("Reference alpha=.20 target controls are not canonical")

    target_config = configs / f"{MODEL}.yml"
    run(
        [
            sys.executable,
            str(root / "scripts/svtrv2/prepare_dual_order_method.py"),
            "--root", str(root),
            "--method", "m3_alpha025",
            "--stage", "target",
            "--source-checkpoint", str(source),
            "--source-role", f"{SYNTHETIC_MODEL}_best_clean_target_dev_CTC",
            "--max-epoch", str(max_epoch),
            "--batch-size-per-card", str(batch_size),
            "--eval-batch-size-per-card", str(reference_batch_size),
            "--control-world-size", str(args.reference_world_size),
            "--control-first-batch-size", str(reference_batch_size),
            "--num-workers", str(workers),
            "--lr", str(learning_rate),
            "--seed", str(seed),
        ],
        root,
        log_dir / "prepare.log",
    )
    target = yaml.safe_load(target_config.read_text(encoding="utf-8-sig"))
    differences = config_differences(
        controlled_config(target), controlled_config(reference)
    )
    if differences:
        (log_dir / "controlled_config_diff.json").write_text(
            json.dumps(differences, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise ValueError(
            "Generated alpha=.25 target config differs from alpha=.20 outside "
            "experiment identity, source checkpoint, tested alpha, and server "
            f"root path: {differences[:10]}"
        )
    if float(target["Loss"]["consistency_weight"]) != 0.25:
        raise ValueError("Generated target config is not alpha=.25")
    target_global_batch = batch_size * target_world_size
    if target_global_batch != reference_global_batch:
        raise ValueError(
            f"Global batch changed: {target_global_batch} != {reference_global_batch}"
        )

    # Compare the original 2x16 schedule with the requested 1x32 schedule
    # using the exact same migrated data paths and sampler implementation.
    reference_runtime_config = log_dir / "reference_world_size_sampler_config.yml"
    reference_runtime = copy.deepcopy(target)
    set_train_batch_size(reference_runtime, reference_batch_size)
    reference_runtime_config.write_text(
        yaml.safe_dump(reference_runtime, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    reference_sampler_path = log_dir / "reference_world_size_sampler_audit.json"
    target_sampler_path = log_dir / "target_world_size_sampler_audit.json"
    for config_path, world_size, output_path, log_name in (
        (
            reference_runtime_config,
            args.reference_world_size,
            reference_sampler_path,
            "reference_world_size_sampler_audit.log",
        ),
        (
            target_config,
            target_world_size,
            target_sampler_path,
            "target_world_size_sampler_audit.log",
        ),
    ):
        run(
            [
                sys.executable,
                str(root / "scripts/svtrv2/audit_p1_ratio_sampler_ddp.py"),
                "--root", str(root),
                "--config", str(config_path),
                "--world-size", str(world_size),
                "--output", str(output_path),
            ],
            root,
            log_dir / log_name,
        )
    reference_sampler = read_sampler_audit(reference_sampler_path)
    target_sampler = read_sampler_audit(target_sampler_path)
    reference_steps = {
        int(item["sampler_batches"]) for item in reference_sampler["rank_reports"]
    }
    target_steps = {
        int(item["sampler_batches"]) for item in target_sampler["rank_reports"]
    }
    if len(reference_steps) != 1 or len(target_steps) != 1:
        raise ValueError("Sampler ranks do not have equal optimizer-step counts")
    if reference_steps != target_steps:
        raise ValueError(
            "Single-GPU batch conversion changed optimizer steps per epoch: "
            f"reference={sorted(reference_steps)}, target={sorted(target_steps)}"
        )
    if reference_sampler["dataset_samples"] != target_sampler["dataset_samples"]:
        raise ValueError("Sampler audits used different target datasets")

    run(
        [
            sys.executable,
            str(root / "scripts/svtrv2/validate_dual_order_method_config.py"),
            "--config", str(target_config),
            "--expected-initialization", "checkpoint",
        ],
        root,
        log_dir / "config_validation.log",
    )
    smoke_path = log_dir / "target_smoke.json"
    run(
        [
            sys.executable,
            str(root / "scripts/svtrv2/smoke_dual_order_method.py"),
            "--root", str(root),
            "--config", str(target_config),
            "--device-id", str(args.eval_gpu),
            "--smoke-batch-size", str(batch_size),
            "--output", str(smoke_path),
            "--require-full-initialization",
        ],
        root,
        log_dir / "target_smoke.log",
    )
    smoke = json.loads(smoke_path.read_text(encoding="utf-8-sig"))
    if smoke.get("status") != "DUAL_ORDER_REAL_BATCH_FORWARD_BACKWARD_OK":
        raise ValueError(f"Target smoke failed: {smoke}")

    binding = {
        "status": "M3_ALPHA025_TARGET_PREFLIGHT_OK",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": sha256(source),
        "reference_config": str(reference_config),
        "reference_config_sha256": sha256(reference_config),
        "target_config": str(target_config),
        "target_config_sha256": sha256(target_config),
        "consistency_weight": 0.25,
        "seed": seed,
        "max_epoch": max_epoch,
        "reference_world_size": args.reference_world_size,
        "reference_batch_size_per_card": reference_batch_size,
        "reference_global_batch_size": reference_global_batch,
        "target_world_size": target_world_size,
        "batch_size_per_card": batch_size,
        "target_global_batch_size": target_global_batch,
        "optimizer_steps_per_epoch": next(iter(target_steps)),
        "reference_sampler_audit": str(reference_sampler_path),
        "reference_sampler_audit_sha256": sha256(reference_sampler_path),
        "target_sampler_audit": str(target_sampler_path),
        "target_sampler_audit_sha256": sha256(target_sampler_path),
        "ddp_gpus": args.ddp_gpus,
        "stage_mode": "replace" if args.replace_target else "resume_existing",
        "training_started": False,
        "test_evaluated": False,
    }
    (log_dir / "preflight_binding.json").write_text(
        json.dumps(binding, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(binding, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    run_dir = runs / MODEL
    command = [
        sys.executable,
        str(root / "scripts/svtrv2/run_p1_rctc_stage.py"),
        "--root", str(root),
        "--stage", MODEL,
        "--config", str(target_config),
        "--run-dir", str(run_dir),
        "--labels-dir", str(root / "04_model_training/datasets/e1_target_only/labels"),
        "--label-prefix", "target",
        "--eval-prefix", MODEL,
        "--expected-initialization", "checkpoint",
        "--config-kind", "dual_order",
        "--prediction-branch", "ctc",
        "--max-epoch", str(max_epoch),
        "--eval-every", "2",
        "--patience-evals", "5",
        "--min-epoch", "10",
        "--min-delta", "0.0001",
        "--ddp-gpus", args.ddp_gpus,
        "--eval-gpu", str(args.eval_gpu),
        "--master-port", str(args.master_port),
        "--log-dir", str(log_dir),
    ]
    command.append("--replace" if args.replace_target else "--resume-existing")
    run(command, root)
    final_summary = eval_reports / f"{MODEL}_final_summary.json"
    if not final_summary.is_file():
        raise FileNotFoundError(final_summary)
    result = json.loads(final_summary.read_text(encoding="utf-8-sig"))
    result["completion_protocol"] = binding
    result["test_policy"] = "not_evaluated"
    completion = log_dir / "completion_summary.json"
    completion.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("M3_ALPHA025_TARGET_COMPLETION_OK; Test was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
