#!/usr/bin/env python3
"""Freeze, preflight, and run the controlled B1/M3 HEM V1 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], cwd: Path, log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"Command failed ({code}): {' '.join(command)}; see {log}")


def metric_row(summary: dict[str, Any]) -> dict[str, Any]:
    dev = summary["dev"]
    languages = dev["languages"]
    return {
        "best_epoch": summary["best_epoch"],
        "macro_cer": dev["macro_cer"],
        "macro_wer": sum(languages[lang]["wer"] for lang in ("zh", "ug", "kk")) / 3,
        "macro_one_minus_ned": sum(
            languages[lang]["one_minus_ned_macro"] for lang in ("zh", "ug", "kk")
        ) / 3,
        "macro_line_accuracy": dev["macro_line_accuracy"],
        "languages": {
            lang: {
                "cer": languages[lang]["cer"],
                "wer": languages[lang]["wer"],
                "one_minus_ned": languages[lang]["one_minus_ned_macro"],
                "line_accuracy": languages[lang]["line_accuracy"],
            }
            for lang in ("zh", "ug", "kk")
        },
    }


def delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_cer": after["macro_cer"] - before["macro_cer"],
        "macro_wer": after["macro_wer"] - before["macro_wer"],
        "macro_one_minus_ned": after["macro_one_minus_ned"] - before["macro_one_minus_ned"],
        "macro_line_accuracy": after["macro_line_accuracy"] - before["macro_line_accuracy"],
        "languages": {
            lang: {
                key: after["languages"][lang][key] - before["languages"][lang][key]
                for key in ("cer", "wer", "one_minus_ned", "line_accuracy")
            }
            for lang in ("zh", "ug", "kk")
        },
    }


def write_comparison_csv(path: Path, table: dict[str, dict[str, Any]]) -> None:
    fields = [
        "model",
        "best_epoch",
        "macro_cer",
        "macro_wer",
        "macro_one_minus_ned",
        "macro_line_accuracy",
    ]
    for language in ("zh", "ug", "kk"):
        for metric in ("cer", "wer", "one_minus_ned", "line_accuracy"):
            fields.append(f"{language}_{metric}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, metrics in table.items():
            row = {
                "model": model,
                **{key: metrics[key] for key in fields[1:6]},
            }
            for language in ("zh", "ug", "kk"):
                for metric in ("cer", "wer", "one_minus_ned", "line_accuracy"):
                    row[f"{language}_{metric}"] = metrics["languages"][language][metric]
            writer.writerow(row)


def smoke_is_current(path: Path, config: Path, manifest_hash: str, steps: int) -> bool:
    if not path.is_file():
        return False
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    return (
        report.get("status") == "HEM_DDP_TRAINING_SMOKE_OK"
        and report.get("config_sha256") == sha256(config)
        and report.get("hem_manifest_sha256") == manifest_hash
        and report.get("steps_run") == steps
        and report.get("finite_loss_and_gradients") is True
        and report.get("test_evaluated") is False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=200)
    parser.add_argument("--master-port", type=int, default=29931)
    args = parser.parse_args()
    root = args.root.resolve()
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable)
    audit_dir = root / "04_model_training/hem_v1/target_train_clean_hard_audit_v1"
    freeze_dir = root / "04_model_training/hem_v1/hem_v1_frozen"
    prep_report = root / "04_model_training/hem_v1/hem_v1_training_configs.json"

    run(
        [
            str(python),
            str(root / "scripts/svtrv2/freeze_hem_v1.py"),
            "--root", str(root),
            "--audit-dir", str(audit_dir),
            "--output", str(freeze_dir),
        ],
        root,
        log_dir / "freeze_hem_v1.log",
    )
    run(
        [
            str(python),
            str(root / "scripts/svtrv2/prepare_hem_training_v1.py"),
            "--root", str(root),
            "--freeze-dir", str(freeze_dir),
            "--output", str(prep_report),
        ],
        root,
        log_dir / "prepare_hem_training_v1.log",
    )
    prepared = json.loads(prep_report.read_text(encoding="utf-8-sig"))
    manifest_hash = prepared["manifest_sha256"]
    specs = {
        "b1": {
            "config": root / "04_model_training/configs/svtrv2_s_b1_hem_v1_to_target.yml",
            "validator": "validate_p1_full_svtrv2_config.py",
            "kind": "full_svtrv2",
            "run": root / "04_model_training/runs/svtrv2_s_b1_hem_v1_to_target",
            "eval_prefix": "b1_hem_v1_to_target",
        },
        "m3": {
            "config": root / "04_model_training/configs/svtrv2_s_m3_hem_v1_to_target.yml",
            "validator": "validate_dual_order_method_config.py",
            "kind": "dual_order",
            "run": root / "04_model_training/runs/svtrv2_s_m3_hem_v1_to_target",
            "eval_prefix": "m3_hem_v1_to_target",
        },
    }

    preflight = {}
    for index, (model, spec) in enumerate(specs.items()):
        config = spec["config"]
        run(
            [
                str(python),
                str(root / "scripts/svtrv2" / spec["validator"]),
                "--config", str(config),
                "--expected-initialization", "checkpoint",
            ],
            root,
            log_dir / f"{model}_config_validation.log",
        )
        sampler_report = log_dir / f"{model}_hem_sampler_audit.json"
        run(
            [
                str(python),
                str(root / "scripts/svtrv2/audit_hem_ratio_sampler_ddp.py"),
                "--root", str(root),
                "--config", str(config),
                "--world-size", "1",
                "--expected-optimizer-steps", "3065",
                "--output", str(sampler_report),
            ],
            root,
            log_dir / f"{model}_hem_sampler_audit.log",
        )
        smoke_report = root / f"04_model_training/hem_v1/{model}_hem_v1_smoke.json"
        if not smoke_is_current(smoke_report, config, manifest_hash, args.smoke_steps):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "1"
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            run(
                [
                    str(python), "-m", "torch.distributed.launch",
                    "--nproc_per_node=1",
                    f"--master_port={args.master_port + index}",
                    str(root / "scripts/svtrv2/smoke_hem_training_ddp.py"),
                    "--root", str(root),
                    "--config", str(config),
                    "--model", model,
                    "--steps", str(args.smoke_steps),
                    "--output", str(smoke_report),
                ],
                root,
                log_dir / f"{model}_hem_training_smoke.log",
                env=env,
            )
        preflight[model] = {
            "config": str(config),
            "config_sha256": sha256(config),
            "sampler_audit": str(sampler_report),
            "sampler_audit_sha256": sha256(sampler_report),
            "smoke": str(smoke_report),
            "smoke_sha256": sha256(smoke_report),
        }
    preflight_report = {
        "status": "HEM_V1_CONTROLLED_PREFLIGHT_OK",
        "physical_gpu": 1,
        "world_size": 1,
        "batch_size_per_card": 32,
        "global_batch_size": 32,
        "optimizer_steps_per_epoch": 3065,
        "models": preflight,
        "formal_training_started": False,
        "test_evaluated": False,
    }
    preflight_path = root / "04_model_training/hem_v1/hem_v1_preflight.json"
    preflight_path.write_text(
        json.dumps(preflight_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(preflight_report, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    labels = root / "04_model_training/datasets/e1_target_only/labels"
    for index, (model, spec) in enumerate(specs.items()):
        command = [
            str(python),
            str(root / "scripts/svtrv2/run_p1_rctc_stage.py"),
            "--root", str(root),
            "--stage", f"{model}_hem_v1_to_target",
            "--config", str(spec["config"]),
            "--run-dir", str(spec["run"]),
            "--labels-dir", str(labels),
            "--label-prefix", "target",
            "--eval-prefix", spec["eval_prefix"],
            "--expected-initialization", "checkpoint",
            "--max-epoch", "50",
            "--eval-every", "2",
            "--patience-evals", "5",
            "--min-epoch", "10",
            "--min-delta", "0.0001",
            "--ddp-gpus", "1",
            "--eval-gpu", "1",
            "--master-port", str(args.master_port + 10 + index),
            "--log-dir", str(log_dir),
            "--config-kind", spec["kind"],
            "--prediction-branch", "ctc",
            "--sampler-audit-kind", "hem",
        ]
        if args.replace:
            command.append("--replace")
        elif spec["run"].is_dir() and any(spec["run"].glob("epoch_*.pth")):
            command.append("--resume-existing")
        run(command, root, log_dir / f"{model}_hem_v1_stage.log")

    eval_root = root / "04_model_training/eval_reports"
    paths = {
        "B1": eval_root / "b1_full_s50_to_target_final_summary.json",
        "B1+HEM": eval_root / "b1_hem_v1_to_target_final_summary.json",
        "SOAR": eval_root / "svtrv2_s_m3_dual_order_s50_to_target_final_summary.json",
        "SOAR+HEM": eval_root / "m3_hem_v1_to_target_final_summary.json",
    }
    table = {
        name: metric_row(json.loads(path.read_text(encoding="utf-8-sig")))
        for name, path in paths.items()
    }
    deltas = {
        "B1_HEM_minus_B1": delta(table["B1+HEM"], table["B1"]),
        "SOAR_HEM_minus_SOAR": delta(table["SOAR+HEM"], table["SOAR"]),
    }
    soar_hem = table["SOAR+HEM"]
    result = {
        "status": "HEM_V1_CONTROLLED_EXPERIMENT_COMPLETE",
        "checkpoint_selection": "argmin Clean Dev Macro CER",
        "models": table,
        "deltas": deltas,
        "decision": {
            "minimum_gate": (
                soar_hem["macro_cer"] < table["SOAR"]["macro_cer"]
                and soar_hem["macro_line_accuracy"] > table["SOAR"]["macro_line_accuracy"]
            ),
            "strong_target": (
                soar_hem["macro_cer"] <= 0.0225
                and soar_hem["macro_line_accuracy"] >= 0.892
            ),
            "run_additional_seeds": (
                soar_hem["macro_cer"] <= 0.0225
                and soar_hem["macro_line_accuracy"] >= 0.892
            ),
            "per_language_guardrail": {
                lang: {
                    "cer_improved": (
                        soar_hem["languages"][lang]["cer"]
                        < table["SOAR"]["languages"][lang]["cer"]
                    ),
                    "line_accuracy_improved": (
                        soar_hem["languages"][lang]["line_accuracy"]
                        > table["SOAR"]["languages"][lang]["line_accuracy"]
                    ),
                }
                for lang in ("zh", "ug", "kk")
            },
            "hem_complementarity": (
                deltas["SOAR_HEM_minus_SOAR"]["macro_cer"]
                < deltas["B1_HEM_minus_B1"]["macro_cer"]
            ),
        },
        "hem_is_training_strategy_not_core_innovation": True,
        "corrupted_dev_used_for_selection": False,
        "test_evaluated": False,
    }
    summary_path = eval_root / "hem_v1_controlled_comparison.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_comparison_csv(
        eval_root / "hem_v1_controlled_comparison.csv",
        table,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
