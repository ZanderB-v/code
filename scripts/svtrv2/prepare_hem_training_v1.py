#!/usr/bin/env python3
"""Create controlled B1+HEM and M3+HEM target-training configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


HEM_PROTOCOL = "near_miss_hem_training_v1"
SAMPLER_PROTOCOL = "NEAR_MISS_HEM_WEIGHTED_RATIO_V1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_hash(path: Path, expected: str, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{role} SHA-256 mismatch: {actual} != {expected}")


def prepare_one(
    root: Path,
    model: str,
    source_config: Path,
    source_target_summary: Path,
    source_synthetic_summary: Path,
    manifest: Path,
    freeze_report: Path,
) -> dict[str, Any]:
    target_summary = json.loads(source_target_summary.read_text(encoding="utf-8-sig"))
    synthetic_summary = json.loads(source_synthetic_summary.read_text(encoding="utf-8-sig"))
    require_hash(
        source_config,
        target_summary["config_sha256"],
        f"frozen {model} control config",
    )
    source_text = source_config.read_text(encoding="utf-8-sig")
    source_project = yaml.safe_load(source_text)["Global"]["project_name"]
    text = re.sub(
        r"/(?:home/)?data_home/wudayu/experiments/multilingual_meme_ocr/"
        r"svtrv2_line_recognition",
        root.as_posix(),
        source_text,
    )
    cfg = yaml.safe_load(text)
    source_checkpoint = Path(cfg["Global"]["pretrained_model"])
    require_hash(
        source_checkpoint,
        synthetic_summary["best_checkpoint_sha256"],
        f"frozen {model} synthetic initialization",
    )
    if "to_target" in source_checkpoint.as_posix():
        raise ValueError(f"{model}+HEM must not continue from a target-trained checkpoint")

    experiment = f"svtrv2_s_{model}_hem_v1_to_target"
    run_dir = root / "04_model_training/runs" / experiment
    text = text.replace(source_project, experiment)
    global_marker = f"  sampler_protocol: {cfg['Global']['sampler_protocol']}\n"
    hem_global = (
        global_marker
        + f"  hem_protocol: {HEM_PROTOCOL}\n"
        + f"  hem_sampler_protocol: {SAMPLER_PROTOCOL}\n"
        + f"  hem_manifest_sha256: {sha256(manifest)}\n"
        + f"  hem_freeze_sha256: {sha256(freeze_report)}\n"
        + f"  hem_control_model: {model}\n"
        + "  physical_gpu_ids: [1]\n"
        + "  global_batch_size: 32\n"
    )
    if text.count(global_marker) != 1:
        raise ValueError("Cannot place frozen HEM provenance in config")
    text = text.replace(global_marker, hem_global, 1)
    train_text, eval_text = text.split("\nEval:", 1)
    train_text = train_text.replace("    first_bs: 16", "    first_bs: 32", 1)
    sampler_marker = "    max_ratio: 40\n  loader:"
    sampler_replacement = (
        "    max_ratio: 40\n"
        f"    sample_weight_manifest: {manifest}\n"
        f"    hem_protocol: {SAMPLER_PROTOCOL}\n"
        f"    hem_seed: {int(cfg['Global']['seed'])}\n"
        "    control_world_size: 2\n"
        "    control_first_bs: 16\n"
        "  loader:"
    )
    if sampler_marker not in train_text:
        raise ValueError("Cannot place HEM sampler settings in Train section")
    train_text = train_text.replace(sampler_marker, sampler_replacement, 1)
    train_text = train_text.replace(
        "    batch_size_per_card: 16", "    batch_size_per_card: 32", 1
    )
    train_text = train_text.replace(
        "    num_workers: 4",
        "    num_workers: 6\n"
        "    pin_memory: True\n"
        "    persistent_workers: True\n"
        "    prefetch_factor: 3",
        1,
    )
    text = train_text + "\nEval:" + eval_text
    cfg = yaml.safe_load(text)
    if cfg["Train"]["sampler"].get("first_bs") != 32:
        raise ValueError("HEM config did not preserve global batch 32 on GPU1")
    if cfg["Train"]["sampler"].get("control_world_size") != 2:
        raise ValueError("HEM config did not retain the two-rank control schedule")
    if cfg["Train"]["sampler"].get("control_first_bs") != 16:
        raise ValueError("HEM config did not retain control batch 16 per rank")

    output_config = root / "04_model_training/configs" / f"{experiment}.yml"
    output_config.write_text(text, encoding="utf-8")
    return {
        "model": model,
        "config": str(output_config),
        "config_sha256": sha256(output_config),
        "run_dir": str(run_dir),
        "initialization_checkpoint": str(source_checkpoint),
        "initialization_checkpoint_sha256": sha256(source_checkpoint),
        "same_initialization_as_control": True,
        "batch_size_per_card": 32,
        "global_batch_size": 32,
        "control_world_size": 2,
        "control_batch_size_per_rank": 16,
        "physical_gpu_ids": [1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    freeze_dir = args.freeze_dir.resolve()
    manifest = freeze_dir / "formal_hem_manifest.jsonl"
    freeze_report = freeze_dir / "hem_v1_freeze.json"
    freeze = json.loads(freeze_report.read_text(encoding="utf-8-sig"))
    if freeze.get("status") != "HEM_V1_FROZEN":
        raise ValueError("HEM V1 is not frozen")
    if sha256(manifest) != freeze["frozen_file_hashes"]["formal_hem_manifest.jsonl"]:
        raise ValueError("Frozen HEM manifest changed")

    eval_root = root / "04_model_training/eval_reports"
    configs = root / "04_model_training/configs"
    experiments = [
        prepare_one(
            root,
            "b1",
            configs / "svtrv2_s_b1_full_s50_to_target.yml",
            eval_root / "b1_full_s50_to_target_final_summary.json",
            eval_root / "b1_full_s50_final_summary.json",
            manifest,
            freeze_report,
        ),
        prepare_one(
            root,
            "m3",
            configs / "svtrv2_s_m3_dual_order_s50_to_target.yml",
            eval_root / "svtrv2_s_m3_dual_order_s50_to_target_final_summary.json",
            eval_root / "svtrv2_s_m3_dual_order_s50_final_summary.json",
            manifest,
            freeze_report,
        ),
    ]
    report = {
        "status": "HEM_V1_TRAINING_CONFIGS_READY",
        "protocol_id": HEM_PROTOCOL,
        "sampler_protocol": SAMPLER_PROTOCOL,
        "freeze_report": str(freeze_report),
        "freeze_report_sha256": sha256(freeze_report),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "experiments": experiments,
        "controlled_constants": {
            "physical_gpu": 1,
            "world_size": 1,
            "global_batch_size": 32,
            "control_world_size": 2,
            "control_batch_size_per_rank": 16,
            "same_source_initialization_as_each_control": True,
            "same_target_train_dev": True,
            "same_optimizer_scheduler_warmup_epochs": True,
            "selection_metric": "Clean Dev Macro CER",
            "corrupted_dev_for_selection": False,
        },
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
