#!/usr/bin/env python3
"""Sequential S50 -> target -> Clean/Corrupted Dev comparison orchestrator."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from formal_baseline_data import (
    PROJECT_ROOT,
    formal_implementation_hashes,
    load_protocol,
    preflight_implementation_hashes,
    protocol_artifact_hashes,
)
from pretrained_models import sha256_file


MODELS = ("crnn", "svtr", "parseq", "abinet")


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--eval-gpu", default="0")
    args = parser.parse_args()

    protocol = load_protocol()
    protocol_path = (
        PROJECT_ROOT / "Comparison/pretrained_baselines_v1/protocol.json"
    )
    protocol_sha256 = sha256_file(protocol_path)
    preflight_path = (
        PROJECT_ROOT
        / "Comparison/pretrained_baselines_v1/outputs/pretrained_baselines_v1_preflight_summary.json"
    )
    if not preflight_path.is_file():
        raise FileNotFoundError(
            f"Enhanced formal preflight summary is missing: {preflight_path}"
        )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed" or not preflight.get(
        "formal_training_allowed"
    ):
        raise ValueError("Enhanced formal preflight did not authorize training")
    if preflight.get("formal_protocol_audit") != "passed":
        raise ValueError("Full formal data/capacity audit is not passed")
    if preflight.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Enhanced preflight summary belongs to a stale protocol")
    if preflight.get("implementation_sha256") != formal_implementation_hashes():
        raise ValueError("Enhanced preflight summary belongs to stale training code")
    if preflight.get("preflight_implementation_sha256") != (
        preflight_implementation_hashes()
    ):
        raise ValueError("Enhanced preflight summary belongs to stale gate code")
    if preflight.get("artifact_sha256") != protocol_artifact_hashes(protocol):
        raise ValueError("Enhanced preflight summary belongs to stale data or weights")
    for model in args.models:
        item = preflight.get("models", {}).get(model, {})
        if item.get("formal_ddp_preflight") != "passed":
            raise ValueError(f"Formal two-GPU preflight is not passed for {model}")

    run_root = PROJECT_ROOT / "Comparison/pretrained_baselines_v1/formal_runs"
    eval_root = PROJECT_ROOT / "Comparison/pretrained_baselines_v1/formal_eval"
    run_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    python = sys.executable
    train_script = str(
        PROJECT_ROOT / "Comparison/scripts/train_pretrained_baseline.py"
    )
    eval_script = str(
        PROJECT_ROOT / "Comparison/scripts/evaluate_pretrained_baseline.py"
    )

    summaries = {}
    for model in args.models:
        model_run = run_root / model
        model_eval = eval_root / model
        if args.replace:
            shutil.rmtree(model_run, ignore_errors=True)
            shutil.rmtree(model_eval, ignore_errors=True)
        synthetic_run = model_run / "s50_adaptation"
        synthetic_eval = model_eval / "s50_adaptation_clean_dev"
        target_run = model_run / "target_finetune"
        target_eval_epochs = model_eval / "target_finetune_clean_dev_epochs"
        final_eval = model_eval / "best_target_clean_corrupted_dev"

        print(f"===== {model.upper()} 1/3 S50 ADAPTATION =====", flush=True)
        synthetic_summary = synthetic_run / "stage_summary.json"
        synthetic_best = synthetic_run / "best_clean_dev_macro_cer.pth"
        if synthetic_summary.is_file() and synthetic_best.is_file():
            print(f"SKIP completed synthetic stage: {synthetic_summary}", flush=True)
        else:
            synthetic_resume = ["--resume"] if (synthetic_run / "latest.pth").is_file() else []
            if synthetic_run.exists() and any(synthetic_run.iterdir()) and not synthetic_resume:
                raise RuntimeError(
                    f"Incomplete synthetic directory has no resumable checkpoint: {synthetic_run}"
                )
            run(
                [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                train_script,
                "--model",
                model,
                "--stage",
                "synthetic",
                "--run-dir",
                str(synthetic_run),
                "--eval-root",
                str(synthetic_eval),
                "--num-workers",
                str(args.num_workers),
                ]
                + synthetic_resume,
                env,
            )
        source = synthetic_best
        if not source.is_file():
            raise FileNotFoundError(source)

        print(f"===== {model.upper()} 2/3 TARGET FINE-TUNE =====", flush=True)
        target_summary = target_run / "stage_summary.json"
        target_best = target_run / "best_clean_dev_macro_cer.pth"
        if target_summary.is_file() and target_best.is_file():
            print(f"SKIP completed target stage: {target_summary}", flush=True)
        else:
            target_resume = ["--resume"] if (target_run / "latest.pth").is_file() else []
            if target_run.exists() and any(target_run.iterdir()) and not target_resume:
                raise RuntimeError(
                    f"Incomplete target directory has no resumable checkpoint: {target_run}"
                )
            run(
                [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                train_script,
                "--model",
                model,
                "--stage",
                "target",
                "--source-checkpoint",
                str(source),
                "--run-dir",
                str(target_run),
                "--eval-root",
                str(target_eval_epochs),
                "--num-workers",
                str(args.num_workers),
                ]
                + target_resume,
                env,
            )
        best = target_best
        if not best.is_file():
            raise FileNotFoundError(best)

        print(f"===== {model.upper()} 3/3 FROZEN DEV DIAGNOSTICS =====", flush=True)
        eval_env = dict(os.environ)
        eval_env["CUDA_VISIBLE_DEVICES"] = args.eval_gpu
        final_summary_path = final_eval / "final_dev_summary.json"
        if final_summary_path.is_file():
            print(f"SKIP completed Dev diagnostics: {final_summary_path}", flush=True)
        else:
            run(
                [
                python,
                eval_script,
                "--model",
                model,
                "--checkpoint",
                str(best),
                "--output-dir",
                str(final_eval),
                "--device",
                "cuda:0",
                "--num-workers",
                str(args.num_workers),
                ],
                eval_env,
            )
        summary = json.loads(
            final_summary_path.read_text(encoding="utf-8")
        )
        summaries[model] = {
            "clean_dev_macro_cer": summary["clean_dev_macro_cer"],
            "mean_corrupted_macro_cer": summary["mean_corrupted_macro_cer"],
            "absolute_cer_increase": summary["absolute_cer_increase"],
            "relative_cer_increase": summary["relative_cer_increase"],
            "best_checkpoint": str(best),
            "best_checkpoint_sha256": summary["checkpoint_sha256"],
        }

    result = {
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": formal_implementation_hashes(),
        "comparison_scope": (
            "controlled downstream comparison with public initializations from "
            "different disclosed pretraining sources; not a pure architecture-only test"
        ),
        "models": summaries,
        "checkpoint_selection": "Clean Dev Macro CER",
        "corrupted_dev_role": "post-selection diagnostic only",
        "test_evaluated": False,
    }
    output = eval_root / "pretrained_baselines_v1_formal_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("PRETRAINED_BASELINES_V1_FORMAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
