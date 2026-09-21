#!/usr/bin/env python3
"""Fair B1 S25/S50 scale selection with exact optimizer-step control."""

from __future__ import annotations

import argparse
import copy
import csv
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

from p1_msr_protocol import (
    DEFAULT_MAX_RATIO,
    EVAL_INFERENCE_BATCH_SIZE,
    PROTOCOL_ID,
    assert_p1_config_text,
    ensure_synthetic_scale_lmdbs,
    ensure_target_lmdbs,
    make_full_svtrv2_s_config,
    normalize_spaces,
    read_jsonl,
    u2_text,
)


PROTOCOL_ID_SCALE = "B1_S25_S50_EQUAL_OPTIMIZER_STEPS_V1"
SCALES = ("s25", "s50")
OFFICIAL_RCTC = Path(
    "/home/wudayu/models/openocr_svtrv2/official/svtrv2_s_union14m/best.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0, choices=(0,))
    parser.add_argument("--batch-size", type=int, default=32, choices=(32,))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--synthetic-total-updates", type=int, default=115350)
    parser.add_argument("--synthetic-warmup-steps", type=int, default=4614)
    parser.add_argument("--target-max-epoch", type=int, default=50)
    parser.add_argument("--target-warmup-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    command: list[str], cwd: Path, log: Path | None = None
) -> None:
    print("+ " + " ".join(command), flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    if log is None:
        subprocess.run(command, cwd=str(cwd), env=env, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed ({code}); see {log}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def protocol_paths(root: Path) -> dict[str, Path]:
    base = root / "04_model_training" / "scale_selection_v1"
    return {
        "base": base,
        "audits": base / "audits",
        "fixed": base / "fixed_step_summaries",
        "terminal_eval": base / "terminal_clean_dev",
        "adapters": base / "checkpoint_adapters",
        "summary": base / "b1_scale_step_control_summary.json",
        "csv": base / "b1_scale_step_control_summary.csv",
        "lock": base / "final_scale_lock.json",
    }


def names(scale: str) -> dict[str, str]:
    stem = f"b1_scale_step_v1_{scale}"
    return {
        "rctc_key": f"{stem}_rctc",
        "rctc_model": f"svtrv2_s_{stem}_rctc",
        "full_key": f"{stem}_full",
        "full_model": f"svtrv2_s_{stem}_full",
        "target_key": f"{stem}_to_target",
        "target_model": f"svtrv2_s_{stem}_to_target",
    }


def audit_sampler(
    args: argparse.Namespace, config: Path, output: Path
) -> int:
    run(
        [
            sys.executable,
            str(args.root / "scripts/svtrv2/audit_p1_ratio_sampler_ddp.py"),
            "--root", str(args.root),
            "--config", str(config),
            "--world-size", "1",
            "--output", str(output),
        ],
        args.root,
    )
    report = load_json(output)
    if report.get("status") != "P1_RATIO_SAMPLER_DDP_AUDIT_OK":
        raise RuntimeError(f"Sampler audit failed: {output}")
    reports = report.get("rank_reports") or []
    if len(reports) != 1:
        raise ValueError("GPU0 protocol requires exactly one sampler rank")
    return int(reports[0]["sampler_batches"])


def lock_step_config(
    path: Path,
    *,
    steps_per_epoch: int,
    total_steps: int,
    warmup_steps: int,
    fixed_terminal: bool,
) -> None:
    text = path.read_text(encoding="utf-8-sig")
    cfg = yaml.safe_load(text)
    global_cfg = cfg["Global"]
    if int(global_cfg.get("accumulation_steps", 1)) != 1:
        raise ValueError("Equal-step protocol requires accumulation_steps=1")
    if steps_per_epoch <= 0 or total_steps <= 0:
        raise ValueError("Step-control values must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup < total")

    generated_only_tokens = (
        "steps_per_epoch_audit:",
        "scale_control_protocol:",
        "physical_gpu_ids:",
        "global_batch_size:",
        "strict_lr_scheduler:",
        "strict_epoch_length:",
        "max_optimizer_steps:",
        "pin_memory:",
        "persistent_workers:",
        "prefetch_factor:",
    )
    unexpected = [token for token in generated_only_tokens if token in text]
    if unexpected:
        raise ValueError(
            f"Step-control config was already mutated: {unexpected}"
        )

    epoch_num = int(math.ceil(total_steps / steps_per_epoch))
    global_lines = [
        f"  epoch_num: {epoch_num}",
        f"  steps_per_epoch_audit: {steps_per_epoch}",
        f"  scale_control_protocol: {PROTOCOL_ID_SCALE}",
        "  physical_gpu_ids: [0]",
        "  world_size: 1",
        "  global_batch_size: 32",
        "  strict_lr_scheduler: True",
        "  strict_epoch_length: True",
    ]
    if fixed_terminal:
        global_lines.extend(
            [
                f"  max_optimizer_steps: {total_steps}",
                f"  save_iter_step: [{total_steps + 1}, {total_steps + 1}]",
            ]
        )
    text, count = re.subn(
        r"(?m)^  epoch_num: [^\r\n]+$",
        "\n".join(global_lines),
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Expected exactly one Global.epoch_num line")

    text, count = re.subn(
        r"(?m)^  warmup_epoch: [^\r\n]+$",
        f"  total_steps: {total_steps}\n  warmup_steps: {warmup_steps}",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Expected exactly one LRScheduler.warmup_epoch line")

    train_start = text.find("\nTrain:\n")
    eval_start = text.find("\nEval:\n")
    if train_start < 0 or eval_start <= train_start:
        raise ValueError("Could not isolate Train loader section")
    train_text = text[train_start:eval_start]
    train_text, count = re.subn(
        r"(?m)^    num_workers: ([^\r\n]+)$",
        "    num_workers: \\1\n"
        "    pin_memory: True\n"
        "    persistent_workers: True\n"
        "    prefetch_factor: 4",
        train_text,
        count=1,
    )
    if count != 1:
        raise ValueError("Expected exactly one Train.loader.num_workers line")
    text = text[:train_start] + train_text + text[eval_start:]

    # The frozen P1 validator intentionally checks these literal spellings.
    assert_p1_config_text(text)
    locked = yaml.safe_load(text)
    locked_global = locked["Global"]
    expected_global = {
        "epoch_num": epoch_num,
        "steps_per_epoch_audit": steps_per_epoch,
        "scale_control_protocol": PROTOCOL_ID_SCALE,
        "physical_gpu_ids": [0],
        "world_size": 1,
        "global_batch_size": 32,
        "strict_lr_scheduler": True,
        "strict_epoch_length": True,
    }
    if fixed_terminal:
        expected_global.update(
            {
                "max_optimizer_steps": total_steps,
                "save_iter_step": [total_steps + 1, total_steps + 1],
            }
        )
    for key, expected in expected_global.items():
        if locked_global.get(key) != expected:
            raise ValueError(
                f"Locked Global.{key} mismatch: "
                f"{locked_global.get(key)!r} != {expected!r}"
            )
    scheduler = locked["LRScheduler"]
    if scheduler.get("total_steps") != total_steps:
        raise ValueError("Locked scheduler total_steps mismatch")
    if scheduler.get("warmup_steps") != warmup_steps:
        raise ValueError("Locked scheduler warmup_steps mismatch")
    if "warmup_epoch" in scheduler:
        raise ValueError("Epoch-based scheduler field survived step locking")
    train_loader = locked["Train"]["loader"]
    expected_loader = {
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
    for key, expected in expected_loader.items():
        if train_loader.get(key) != expected:
            raise ValueError(f"Locked Train.loader.{key} mismatch")
    path.write_text(text, encoding="utf-8")


def controlled_config_fingerprint(path: Path) -> str:
    cfg = copy.deepcopy(yaml.safe_load(path.read_text(encoding="utf-8-sig")))
    global_cfg = cfg["Global"]
    for key in (
        "output_dir",
        "project_name",
        "save_res_path",
        "pretrained_model",
        "epoch_num",
        "steps_per_epoch_audit",
    ):
        global_cfg.pop(key, None)
    cfg["Train"]["dataset"]["data_dir_list"] = ["<SCALE_CONTROLLED_DATA>"]
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_controlled_pair(configs: list[Path], stage: str) -> str:
    fingerprints = [controlled_config_fingerprint(path) for path in configs]
    if len(set(fingerprints)) != 1:
        raise ValueError(
            f"{stage} configs differ outside approved scale/path fields: "
            f"{dict(zip(map(str, configs), fingerprints))}"
        )
    return fingerprints[0]


def make_full_config(
    args: argparse.Namespace,
    scale: str,
    source_checkpoint: Path,
    *,
    target: bool,
    max_epoch: int,
) -> Path:
    item = names(scale)
    model_name = item["target_model" if target else "full_model"]
    run_dir = args.root / "04_model_training/runs" / model_name
    config = args.root / "04_model_training/configs" / f"{model_name}.yml"
    target_root = args.root / "01_data_preparation/real_line_dataset_eval_reviewed"
    target_rows = read_jsonl(target_root / "metadata.jsonl")
    target_train, target_dev, _ = ensure_target_lmdbs(
        args.root, target_root, target_rows, replace=False
    )
    if target:
        train_lmdbs = target_train
    else:
        train_lmdbs, _ = ensure_synthetic_scale_lmdbs(
            args.root,
            args.root / "03_synthetic_generation/synthetic_formal_v2",
            scale,
            replace=False,
        )
    config.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    config.write_text(
        make_full_svtrv2_s_config(
            root=args.root,
            run_dir=run_dir,
            project_name=model_name,
            train_lmdbs=train_lmdbs,
            eval_lmdbs=target_dev,
            pretrained_model=source_checkpoint,
            max_epoch=max_epoch,
            first_batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_ratio=DEFAULT_MAX_RATIO,
            lr=2.5e-5,
            internal_eval_every=100000,
            seed=args.seed,
        ),
        encoding="utf-8",
    )
    return config


def ensure_target_dev_labels(args: argparse.Namespace, output: Path) -> Path:
    target_root = args.root / "01_data_preparation/real_line_dataset_eval_reviewed"
    rows = [
        row for row in read_jsonl(target_root / "metadata.jsonl")
        if row.get("split") == "dev"
    ]
    expected = {"zh": 346, "ug": 295, "kk": 310}
    output.mkdir(parents=True, exist_ok=True)
    for language, expected_count in expected.items():
        language_rows = [row for row in rows if row.get("language") == language]
        if len(language_rows) != expected_count:
            raise ValueError(
                f"Target Dev {language} rows changed: "
                f"{len(language_rows)} != {expected_count}"
            )
        logical_lines = []
        u2_lines = []
        for row in language_rows:
            image = str(row.get("image") or "").replace("\\", "/")
            logical = normalize_spaces(
                row.get("logical_text") or row.get("text") or ""
            )
            if not image or not logical:
                raise ValueError(f"Invalid target Dev row: {row.get('id')}")
            logical_lines.append(f"{image}\t{logical}")
            u2_lines.append(f"{image}\t{u2_text(language, logical)}")
        for order, lines in (("logical", logical_lines), ("u2", u2_lines)):
            (output / f"target_dev_{language}_{order}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
    return output


def validate_and_smoke(
    args: argparse.Namespace, config: Path, kind: str
) -> None:
    if kind == "rctc":
        validator = "validate_p1_msr_config.py"
        smoke = "smoke_p1_rctc_config.py"
        expected = "union14m"
    else:
        validator = "validate_p1_full_svtrv2_config.py"
        smoke = "smoke_p1_full_svtrv2_config.py"
        expected = "checkpoint"
    run(
        [
            sys.executable,
            str(args.root / "scripts/svtrv2" / validator),
            "--config", str(config),
            "--expected-initialization", expected,
        ],
        args.root,
    )
    smoke_command = [
            sys.executable,
            str(args.root / "scripts/svtrv2" / smoke),
            "--root", str(args.root),
            "--config", str(config),
            "--device-id", "0",
        ]
    if kind == "full":
        smoke_command.extend(["--smoke-batch-size", str(args.batch_size)])
    run(
        smoke_command,
        args.root,
    )


def fixed_stage(
    args: argparse.Namespace,
    config: Path,
    run_dir: Path,
    tag: str,
    paths: dict[str, Path],
) -> Path:
    checkpoint = run_dir / f"step_{args.synthetic_total_updates}.pth"
    command = [
        sys.executable,
        str(args.root / "scripts/svtrv2/run_fixed_optimizer_step_stage.py"),
        "--root", str(args.root),
        "--config", str(config),
        "--run-dir", str(run_dir),
        "--log", str(args.log_dir / f"{tag}_train.log"),
        "--summary", str(paths["fixed"] / f"{tag}.json"),
        "--total-updates", str(args.synthetic_total_updates),
        "--physical-gpu", "0",
    ]
    if args.replace:
        command.append("--replace")
    run(command, args.root)
    return checkpoint


def evaluate_terminal(
    args: argparse.Namespace,
    scale: str,
    stage: str,
    config: Path,
    checkpoint: Path,
    paths: dict[str, Path],
) -> dict:
    tag = f"{scale}_{stage}_terminal"
    output = paths["terminal_eval"] / tag
    labels = (
        args.root
        / "04_model_training/datasets"
        / names(scale)["rctc_key"]
        / "labels"
    )
    run(
        [
            sys.executable,
            str(args.root / "scripts/svtrv2/evaluate_d2_checkpoint.py"),
            "--root", str(args.root),
            "--config", str(config),
            "--checkpoint", str(checkpoint),
            "--split", "dev",
            "--output-dir", str(output),
            "--labels-dir", str(labels),
            "--label-prefix", "real",
            "--device-id", "0",
            "--batch-size", str(EVAL_INFERENCE_BATCH_SIZE),
            "--require-python-bidi",
            "--expected-preprocess-protocol", PROTOCOL_ID,
            "--prediction-branch", "ctc" if stage == "full" else "auto",
        ],
        args.root,
        args.log_dir / f"{tag}_eval.log",
    )
    return load_json(output / "metrics_macro_summary.json")


def target_stage(
    args: argparse.Namespace,
    scale: str,
    config: Path,
    target_steps: int,
) -> dict:
    item = names(scale)
    run_dir = args.root / "04_model_training/runs" / item["target_model"]
    eval_prefix = item["target_key"]
    final = args.root / "04_model_training/eval_reports" / f"{eval_prefix}_final_summary.json"
    if not args.replace and final.is_file():
        existing = load_json(final)
        best_checkpoint = Path(str(existing.get("best_checkpoint") or ""))
        if (
            existing.get("config_sha256") == sha256(config)
            and best_checkpoint.is_file()
            and existing.get("best_checkpoint_sha256") == sha256(best_checkpoint)
            and existing.get("preprocess_protocol") == PROTOCOL_ID
        ):
            print(f"[{scale}] reusing verified completed target stage: {final}")
            return existing
    command = [
        sys.executable,
        str(args.root / "scripts/svtrv2/run_p1_rctc_stage.py"),
        "--root", str(args.root),
        "--stage", item["target_key"],
        "--config", str(config),
        "--run-dir", str(run_dir),
        "--labels-dir", str(protocol_paths(args.root)["base"] / "target_dev_labels"),
        "--label-prefix", "target",
        "--eval-prefix", eval_prefix,
        "--expected-initialization", "checkpoint",
        "--config-kind", "full_svtrv2",
        "--prediction-branch", "ctc",
        "--max-epoch", str(args.target_max_epoch),
        "--eval-every", "2",
        "--patience-evals", "5",
        "--min-epoch", "10",
        "--min-delta", "0.0001",
        "--ddp-gpus", "0",
        "--eval-gpu", "0",
        "--master-port", "29872",
        "--log-dir", str(args.log_dir),
    ]
    if args.replace:
        command.append("--replace")
    elif any(run_dir.glob("epoch_*.pth")):
        command.append("--resume-existing")
    run(command, args.root)
    summary = load_json(final)
    control = summary.get("training_control") or {}
    if int(control.get("scheduler_horizon_epoch", -1)) != args.target_max_epoch:
        raise ValueError(f"Target scheduler horizon changed for {scale}")
    return summary


def macro_secondary(dev: dict) -> dict[str, float]:
    langs = dev["languages"]
    values = [langs[lang] for lang in ("zh", "ug", "kk")]
    return {
        "macro_cer": float(dev["macro_cer"]),
        "macro_wer": sum(float(item["wer"]) for item in values) / 3,
        "macro_one_minus_ned": sum(
            float(item["one_minus_ned_macro"]) for item in values
        ) / 3,
        "macro_line_accuracy": float(dev["macro_line_accuracy"]),
    }


def remove_protocol_outputs(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    if paths["base"].exists():
        shutil.rmtree(paths["base"])
    for scale in SCALES:
        for model in (
            names(scale)["rctc_model"],
            names(scale)["full_model"],
            names(scale)["target_model"],
        ):
            path = args.root / "04_model_training/runs" / model
            if path.exists():
                shutil.rmtree(path)
            config = args.root / "04_model_training/configs" / f"{model}.yml"
            if config.exists():
                config.unlink()
        dataset_dir = (
            args.root
            / "04_model_training/datasets"
            / names(scale)["rctc_key"]
        )
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        for key in (names(scale)["rctc_key"], names(scale)["target_key"]):
            for candidate in (
                args.root / "04_model_training/eval_reports" / f"{key}_final_summary.json",
                args.root / "04_model_training/eval_reports" / f"{key}_best_clean_dev",
            ):
                if candidate.is_dir():
                    shutil.rmtree(candidate)
                elif candidate.exists():
                    candidate.unlink()


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    paths = protocol_paths(args.root)
    if args.replace:
        remove_protocol_outputs(args, paths)
    for value in paths.values():
        if value.suffix:
            value.parent.mkdir(parents=True, exist_ok=True)
        else:
            value.mkdir(parents=True, exist_ok=True)
    if not OFFICIAL_RCTC.is_file():
        raise FileNotFoundError(OFFICIAL_RCTC)
    if args.synthetic_warmup_steps >= args.synthetic_total_updates:
        raise ValueError("Synthetic warmup must be shorter than total updates")

    run(
        [
            sys.executable,
            str(args.root / "scripts/protocol/freeze_protocol_v2.py"),
            "--root", str(args.root),
            "--mode", "verify",
            "--workers", "8",
        ],
        args.root,
    )
    run(
        [
            sys.executable,
            str(args.root / "scripts/svtrv2/test_b1_scale_step_control.py"),
        ],
        args.root,
    )
    ensure_target_dev_labels(args, paths["base"] / "target_dev_labels")

    configs: dict[str, dict[str, Path]] = {}
    steps: dict[str, int] = {}
    for scale in SCALES:
        item = names(scale)
        run(
            [
                sys.executable,
                str(args.root / "scripts/svtrv2/prepare_d2_synth50k.py"),
                "--root", str(args.root),
                "--scale", scale,
                "--max-epoch", "100",
                "--batch-size-per-card", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--lr", "0.00005",
                "--internal-eval-every", "100000",
                "--seed", str(args.seed),
                "--experiment-key", item["rctc_key"],
                "--model-name", item["rctc_model"],
            ],
            args.root,
        )
        rctc = args.root / "04_model_training/configs" / f"{item['rctc_model']}.yml"
        provisional = paths["audits"] / f"{scale}_rctc_provisional.json"
        steps[scale] = audit_sampler(args, rctc, provisional)
        lock_step_config(
            rctc,
            steps_per_epoch=steps[scale],
            total_steps=args.synthetic_total_updates,
            warmup_steps=args.synthetic_warmup_steps,
            fixed_terminal=True,
        )
        final_audit = paths["audits"] / f"{scale}_rctc_final.json"
        if audit_sampler(args, rctc, final_audit) != steps[scale]:
            raise RuntimeError("Config locking changed sampler length")

        adapter = paths["adapters"] / "union14m_rctc_for_full_preflight.pth"
        if not adapter.exists():
            complete_rctc = paths["adapters"] / "union14m_4891_rctc_preflight.pth"
            run(
                [
                    sys.executable,
                    str(args.root / "scripts/svtrv2/materialize_rctc_initialization.py"),
                    "--root", str(args.root),
                    "--config", str(rctc),
                    "--source", str(OFFICIAL_RCTC),
                    "--output", str(complete_rctc),
                    "--seed", str(args.seed),
                    "--replace",
                ],
                args.root,
            )
            run(
                [
                    sys.executable,
                    str(args.root / "scripts/svtrv2/convert_rctc_checkpoint_to_full_svtrv2.py"),
                    "--input", str(complete_rctc),
                    "--output", str(adapter),
                    "--replace",
                ],
                args.root,
            )
        full = make_full_config(
            args, scale, adapter, target=False,
            max_epoch=math.ceil(args.synthetic_total_updates / steps[scale]),
        )
        lock_step_config(
            full,
            steps_per_epoch=steps[scale],
            total_steps=args.synthetic_total_updates,
            warmup_steps=args.synthetic_warmup_steps,
            fixed_terminal=True,
        )
        if audit_sampler(args, full, paths["audits"] / f"{scale}_full_final.json") != steps[scale]:
            raise RuntimeError("RCTC/full sampler schedules differ")
        configs[scale] = {"rctc": rctc, "full": full}

    if steps["s25"] == steps["s50"]:
        raise ValueError("S25 and S50 unexpectedly have equal epoch lengths")
    controlled_fingerprints = {
        "rctc": assert_controlled_pair(
            [configs[scale]["rctc"] for scale in SCALES], "Stage-1 RCTC"
        ),
        "full": assert_controlled_pair(
            [configs[scale]["full"] for scale in SCALES], "Stage-2 full B1"
        ),
    }
    for scale in SCALES:
        validate_and_smoke(args, configs[scale]["rctc"], "rctc")
        validate_and_smoke(args, configs[scale]["full"], "full")

    preflight = {
        "status": "B1_SCALE_STEP_CONTROL_V1_PREFLIGHT_OK",
        "protocol_id": PROTOCOL_ID_SCALE,
        "physical_gpu": 0,
        "world_size": 1,
        "batch_size_per_card": 32,
        "global_batch_size": 32,
        "use_amp": False,
        "scheduler_runtime_tested": True,
        "full_model_batch32_forward_backward_smoke": True,
        "synthetic_total_updates_per_stage": args.synthetic_total_updates,
        "synthetic_warmup_steps": args.synthetic_warmup_steps,
        "audited_steps_per_epoch": steps,
        "controlled_config_fingerprints": controlled_fingerprints,
        "epoch_ceiling": {
            scale: math.ceil(args.synthetic_total_updates / count)
            for scale, count in steps.items()
        },
        "stage1": "RCTC from the same Union14M checkpoint",
        "stage2": "full B1 from each arm's terminal Stage-1 checkpoint",
        "stage_transition": "exact terminal checkpoint; no Dev-based pretrain checkpoint selection",
        "target_checkpoint_selection": "Clean Dev Macro CER",
        "test_evaluated": False,
        "formal_training_started": False,
    }
    preflight_path = paths["base"] / "preflight_summary.json"
    preflight_path.write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.preflight_only:
        return

    terminal: dict[str, dict[str, object]] = {scale: {} for scale in SCALES}
    for scale in SCALES:
        item = names(scale)
        run_dir = args.root / "04_model_training/runs" / item["rctc_model"]
        checkpoint = fixed_stage(
            args, configs[scale]["rctc"], run_dir,
            f"{scale}_rctc", paths,
        )
        terminal[scale]["rctc_checkpoint"] = str(checkpoint)
        terminal[scale]["rctc_dev_diagnostic"] = evaluate_terminal(
            args, scale, "rctc", configs[scale]["rctc"], checkpoint, paths
        )

    for scale in SCALES:
        item = names(scale)
        rctc_checkpoint = Path(str(terminal[scale]["rctc_checkpoint"]))
        adapter = paths["adapters"] / f"{scale}_terminal_rctc_for_full.pth"
        run(
            [
                sys.executable,
                str(args.root / "scripts/svtrv2/convert_rctc_checkpoint_to_full_svtrv2.py"),
                "--input", str(rctc_checkpoint),
                "--output", str(adapter),
                "--replace",
            ],
            args.root,
        )
        full = make_full_config(
            args, scale, adapter, target=False,
            max_epoch=math.ceil(args.synthetic_total_updates / steps[scale]),
        )
        lock_step_config(
            full,
            steps_per_epoch=steps[scale],
            total_steps=args.synthetic_total_updates,
            warmup_steps=args.synthetic_warmup_steps,
            fixed_terminal=True,
        )
        validate_and_smoke(args, full, "full")
        run_dir = args.root / "04_model_training/runs" / item["full_model"]
        checkpoint = fixed_stage(
            args, full, run_dir, f"{scale}_full", paths
        )
        configs[scale]["full"] = full
        terminal[scale]["full_checkpoint"] = str(checkpoint)
        terminal[scale]["full_dev_diagnostic"] = evaluate_terminal(
            args, scale, "full", full, checkpoint, paths
        )
    controlled_fingerprints["full_formal"] = assert_controlled_pair(
        [configs[scale]["full"] for scale in SCALES],
        "Stage-2 full B1 formal",
    )

    target_steps: dict[str, int] = {}
    target_results: dict[str, dict] = {}
    for scale in SCALES:
        config = make_full_config(
            args,
            scale,
            Path(str(terminal[scale]["full_checkpoint"])),
            target=True,
            max_epoch=args.target_max_epoch,
        )
        provisional = paths["audits"] / f"{scale}_target_provisional.json"
        target_steps[scale] = audit_sampler(args, config, provisional)
        total = target_steps[scale] * args.target_max_epoch
        warmup = target_steps[scale] * args.target_warmup_epochs
        lock_step_config(
            config,
            steps_per_epoch=target_steps[scale],
            total_steps=total,
            warmup_steps=warmup,
            fixed_terminal=False,
        )
        validate_and_smoke(args, config, "full")
        configs[scale]["target"] = config
    if target_steps["s25"] != target_steps["s50"]:
        raise ValueError(f"Target fine-tune steps differ: {target_steps}")
    controlled_fingerprints["target"] = assert_controlled_pair(
        [configs[scale]["target"] for scale in SCALES], "Target fine-tune"
    )
    for scale in SCALES:
        target_results[scale] = target_stage(
            args, scale, configs[scale]["target"], target_steps[scale]
        )

    metrics = {
        scale: macro_secondary(target_results[scale]["dev"])
        for scale in SCALES
    }
    difference = abs(metrics["s25"]["macro_cer"] - metrics["s50"]["macro_cer"])
    if difference < 0.0002:
        selected = "s25"
        reason = "CER difference below 0.02 percentage points; lower-cost S25 tie-break"
    else:
        selected = min(SCALES, key=lambda scale: metrics[scale]["macro_cer"])
        reason = "lower Clean Dev Macro CER"
    result = {
        "status": "B1_SCALE_STEP_CONTROL_V1_COMPLETE",
        "protocol_id": PROTOCOL_ID_SCALE,
        "controlled_compute": {
            "physical_gpu": 0,
            "world_size": 1,
            "batch_size_per_card": 32,
            "global_batch_size": 32,
            "use_amp": False,
            "stage1_rctc_updates": args.synthetic_total_updates,
            "stage2_full_b1_updates": args.synthetic_total_updates,
            "synthetic_warmup_steps": args.synthetic_warmup_steps,
            "target_steps_per_epoch": target_steps["s25"],
            "target_max_epoch": args.target_max_epoch,
            "same_optimizer_scheduler_seed": True,
        },
        "stage_transition": "terminal exact-step checkpoint only",
        "stage_a_dev_metrics_are_diagnostic_only": True,
        "synthetic_validation_note": (
            "No frozen synthetic-validation split exists; none was invented. "
            "Scale selection uses downstream Clean Dev Macro CER only."
        ),
        "results": {
            scale: {
                "terminal_pretraining": terminal[scale],
                "target_finetune": target_results[scale],
                "cer_selected_metrics": metrics[scale],
            }
            for scale in SCALES
        },
        "selection_rule": "argmin Clean Dev Macro CER; choose S25 when absolute difference < 0.02 pp",
        "absolute_cer_difference": difference,
        "selected_scale": selected,
        "selection_reason": reason,
        "test_evaluated": False,
    }
    paths["summary"].write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with paths["csv"].open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "scale", "cer_selected_epoch", "macro_cer", "macro_wer",
            "macro_one_minus_ned", "macro_line_accuracy",
            "zh_cer", "zh_line_accuracy", "ug_cer", "ug_line_accuracy",
            "kk_cer", "kk_line_accuracy", "selected",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scale in SCALES:
            dev = target_results[scale]["dev"]
            row = {
                "scale": scale,
                "cer_selected_epoch": target_results[scale]["best_epoch"],
                **metrics[scale],
                "selected": scale == selected,
            }
            for language in ("zh", "ug", "kk"):
                row[f"{language}_cer"] = dev["languages"][language]["cer"]
                row[f"{language}_line_accuracy"] = (
                    dev["languages"][language]["line_accuracy"]
                )
            writer.writerow(row)
    lock = {
        "status": "FINAL_SYNTHETIC_SCALE_LOCKED",
        "protocol_id": PROTOCOL_ID_SCALE,
        "selected_scale": selected,
        "selection_reason": reason,
        "summary": str(paths["summary"]),
        "summary_sha256": sha256(paths["summary"]),
        "comparison_csv": str(paths["csv"]),
        "comparison_csv_sha256": sha256(paths["csv"]),
        "selected_best_checkpoint": target_results[selected]["best_checkpoint"],
        "selected_best_checkpoint_sha256": target_results[selected]["best_checkpoint_sha256"],
        "apply_to_future_models": [
            "B1", "M1", "M2", "M3", "CRNN", "SVTR", "PARSeq", "ABINet"
        ],
        "test_evaluated": False,
    }
    paths["lock"].write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
