#!/usr/bin/env python3
"""Run one formal P1/MSR RCTC stage with restart-safe external early stopping."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from p1_msr_protocol import EVAL_INFERENCE_BATCH_SIZE, PROTOCOL_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--label-prefix", required=True)
    parser.add_argument("--eval-prefix", required=True)
    parser.add_argument(
        "--expected-initialization",
        choices=("random", "union14m", "checkpoint"),
        required=True,
    )
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--eval-every", type=int, required=True)
    parser.add_argument("--patience-evals", type=int, required=True)
    parser.add_argument("--min-epoch", type=int, required=True)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--ddp-gpus", default="0,1")
    parser.add_argument("--eval-gpu", type=int, default=0)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--config-kind",
        choices=("rctc", "full_svtrv2", "dual_order"),
        default="rctc",
    )
    parser.add_argument(
        "--prediction-branch",
        choices=("auto", "ctc"),
        default="auto",
    )
    parser.add_argument(
        "--sampler-audit-kind",
        choices=("standard", "hem"),
        default="standard",
    )
    parser.add_argument(
        "--selection-protocol-dir",
        type=Path,
        help=(
            "Optional frozen external Clean Dev protocol. When supplied, every "
            "saved epoch is rescored against it and it alone controls early "
            "stopping and checkpoint selection."
        ),
    )
    parser.add_argument(
        "--selection-frozen-manifest",
        help="Frozen manifest filename inside --selection-protocol-dir.",
    )
    parser.add_argument(
        "--selection-protocol-label",
        help="Human-readable name of the external selection protocol.",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Resume a partially completed stage from its last externally "
            "evaluated checkpoint after validating metrics and hashes."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_checked(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
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
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with code {return_code}: {' '.join(command)}\n"
            f"See {log_path}"
        )


def load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    required = ("state_dict", "optimizer", "scheduler", "epoch", "global_step")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Incomplete checkpoint {path}: missing {missing}")
    if not isinstance(checkpoint["state_dict"], dict):
        raise TypeError(f"Checkpoint state_dict is invalid: {path}")
    if checkpoint["optimizer"] is None:
        raise ValueError(f"Checkpoint optimizer state is empty: {path}")
    if checkpoint["scheduler"] is None:
        raise ValueError(f"Checkpoint scheduler state is empty: {path}")
    scheduler_state = checkpoint["scheduler"]
    scheduler_last_epoch = int(scheduler_state.get("last_epoch", -1))
    scheduler_total_steps = scheduler_state.get("total_steps")
    if (
        scheduler_total_steps is not None
        and scheduler_last_epoch > int(scheduler_total_steps)
    ):
        raise ValueError(
            f"Checkpoint scheduler is already out of range: {path}; "
            f"last_epoch={scheduler_last_epoch}, "
            f"total_steps={scheduler_total_steps}"
        )
    if abs(scheduler_last_epoch - int(checkpoint["global_step"])) > 1:
        raise ValueError(
            f"Checkpoint scheduler/global-step mismatch: {path}; "
            f"scheduler.last_epoch={scheduler_last_epoch}, "
            f"global_step={checkpoint['global_step']}"
        )
    # The stage controller only needs progress metadata. Retaining the full
    # model and AdamW state in this long-lived process causes host RAM to grow
    # across repeated train/evaluate segments.
    metadata = {
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "scheduler_last_epoch": scheduler_last_epoch,
        "scheduler_total_steps": (
            None
            if scheduler_total_steps is None
            else int(scheduler_total_steps)
        ),
    }
    del checkpoint
    gc.collect()
    return metadata


def wait_for_checkpoint(
    checkpoint_path: Path,
    process: subprocess.Popen,
    poll_seconds: int = 5,
) -> dict[str, Any]:
    previous_size = -1
    stable_polls = 0
    while True:
        if checkpoint_path.is_file():
            size = checkpoint_path.stat().st_size
            if size > 0 and size == previous_size:
                stable_polls += 1
            else:
                stable_polls = 0
            previous_size = size
            if stable_polls >= 3:
                try:
                    return_code = process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    # The requested checkpoint is complete. Release all DDP
                    # ranks and DataLoader workers before loading it on CPU.
                    stop_process_group(process)
                    return load_checkpoint(checkpoint_path)
                if return_code != 0:
                    raise RuntimeError(
                        f"Training exited with code {return_code} after writing "
                        f"{checkpoint_path}"
                    )
                return load_checkpoint(checkpoint_path)
        return_code = process.poll()
        if return_code is not None:
            if return_code == 0 and checkpoint_path.is_file():
                return load_checkpoint(checkpoint_path)
            raise RuntimeError(
                f"Training exited with code {return_code} before "
                f"{checkpoint_path} appeared"
            )
        time.sleep(poll_seconds)


def validate_checkpoint_horizon(
    checkpoint_payload: dict[str, Any],
    *,
    max_epoch: int,
    checkpoint_path: Path,
) -> None:
    """Prove that a segmented checkpoint retained the full LR horizon."""
    epoch = int(checkpoint_payload["epoch"])
    global_step = int(checkpoint_payload["global_step"])
    total_steps = checkpoint_payload.get("scheduler_total_steps")
    if epoch <= 0 or global_step <= 0 or total_steps is None:
        raise ValueError(
            f"Cannot validate scheduler horizon for {checkpoint_path}: "
            f"epoch={epoch}, global_step={global_step}, "
            f"total_steps={total_steps}"
        )
    if global_step % epoch != 0:
        raise ValueError(
            f"Non-constant epoch step count in {checkpoint_path}: "
            f"global_step={global_step}, epoch={epoch}. This is incompatible "
            "with the fixed-horizon OneCycleLR protocol."
        )
    steps_per_epoch = global_step // epoch
    expected_total_steps = steps_per_epoch * max_epoch
    if int(total_steps) != expected_total_steps:
        raise ValueError(
            f"Scheduler horizon mismatch in {checkpoint_path}: "
            f"stored total_steps={total_steps}, expected={expected_total_steps} "
            f"({steps_per_epoch} steps/epoch x {max_epoch} epochs). The "
            "checkpoint was produced with a shortened LR schedule and must "
            "not be resumed or reported."
        )


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def start_ddp(
    args: argparse.Namespace,
    openocr_root: Path,
    resume_checkpoint: Path | None,
    train_log: Path,
    target_epoch: int,
) -> subprocess.Popen:
    gpu_ids = [item.strip() for item in args.ddp_gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--ddp-gpus must contain at least one GPU")
    if len(gpu_ids) == 1:
        command = [
            sys.executable,
            "tools/train_rec.py",
            "-c",
            str(args.config),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.launch",
            f"--nproc_per_node={len(gpu_ids)}",
            f"--master_port={args.master_port}",
            "tools/train_rec.py",
            "-c",
            str(args.config),
        ]
    # The optimization horizon must stay invariant across resume segments.
    # stop_after_epoch is only a clean process boundary for external dev
    # evaluation; it must never shorten OneCycleLR's total_steps.
    overrides = [
        f"Global.epoch_num={args.max_epoch}",
        f"Global.stop_after_epoch={target_epoch}",
        "Global.strict_lr_scheduler=true",
        "Global.strict_epoch_length=true",
    ]
    if resume_checkpoint is not None:
        overrides.extend(
            [
                f"Global.checkpoints={resume_checkpoint}",
                "Global.pretrained_model=null",
            ]
        )
    command.extend(["-o", *overrides])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    train_log.parent.mkdir(parents=True, exist_ok=True)
    log = train_log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(openocr_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None

    def pump_output() -> None:
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
        finally:
            log.flush()

    thread = threading.Thread(
        target=pump_output,
        name=f"{args.stage}-ddp-log-tee",
        daemon=True,
    )
    thread.start()
    process._codex_log_handle = log  # type: ignore[attr-defined]
    process._codex_log_thread = thread  # type: ignore[attr-defined]
    return process


def close_process_log(process: subprocess.Popen) -> None:
    thread = getattr(process, "_codex_log_thread", None)
    if thread is not None:
        thread.join(timeout=30)
    handle = getattr(process, "_codex_log_handle", None)
    if handle is not None:
        handle.close()


def prune_uncommitted_resume_outputs(
    args: argparse.Namespace,
    last_evaluated_epoch: int,
) -> None:
    """Remove outputs newer than the last externally validated checkpoint."""
    removed: list[str] = []
    for path in args.run_dir.glob("epoch_*.pth"):
        try:
            epoch = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if epoch > last_evaluated_epoch:
            path.unlink()
            removed.append(str(path))

    # latest.pth can point into an interrupted or invalid segment. The
    # controller always resumes from the explicitly validated epoch file.
    latest = args.run_dir / "latest.pth"
    if latest.exists():
        latest.unlink()
        removed.append(str(latest))

    eval_root = args.root / "04_model_training" / "eval_reports"
    for path in eval_root.glob(f"{args.eval_prefix}_dev_epoch_*"):
        try:
            epoch = int(path.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if epoch > last_evaluated_epoch and path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path))

    if removed:
        print(
            f"[{args.stage}] pruned {len(removed)} uncommitted resume outputs "
            f"newer than epoch {last_evaluated_epoch}",
            flush=True,
        )


def evaluate_dev(
    args: argparse.Namespace,
    checkpoint: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    target_root = (
        args.root
        / "01_data_preparation"
        / "real_line_dataset_eval_reviewed"
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    command = [
        sys.executable,
        str(args.root / "scripts" / "svtrv2" / "evaluate_d2_checkpoint.py"),
        "--root",
        str(args.root),
        "--config",
        str(args.config),
        "--checkpoint",
        str(checkpoint),
        "--split",
        "dev",
        "--data-dir",
        str(target_root),
        "--labels-dir",
        str(args.labels_dir),
        "--label-prefix",
        args.label_prefix,
        "--output-dir",
        str(output_dir),
        "--device-id",
        str(args.eval_gpu),
        "--batch-size",
        str(EVAL_INFERENCE_BATCH_SIZE),
        "--require-python-bidi",
        "--expected-preprocess-protocol",
        PROTOCOL_ID,
    ]
    if args.prediction_branch != "auto":
        command.extend(["--prediction-branch", args.prediction_branch])
    run_checked(
        command,
        cwd=args.root,
        log_path=log_path,
        env=os.environ.copy(),
    )
    summary_path = output_dir / "metrics_macro_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8-sig"))


def load_selection_context(args: argparse.Namespace) -> dict[str, Any] | None:
    """Load an optional model-blind external selection protocol once per stage."""
    if args.selection_protocol_dir is None:
        if args.selection_frozen_manifest or args.selection_protocol_label:
            raise ValueError(
                "--selection-frozen-manifest/--selection-protocol-label require "
                "--selection-protocol-dir"
            )
        return None
    if not args.selection_frozen_manifest or not args.selection_protocol_label:
        raise ValueError(
            "An external selection protocol requires both its frozen manifest "
            "and protocol label"
        )

    from rescore_clean_dev_v2 import (
        load_labels,
        load_normalizer,
        sha256 as rescore_sha256,
        verify_protocol,
    )

    verifier_args = argparse.Namespace(
        protocol_dir=args.selection_protocol_dir,
        frozen_manifest=args.selection_frozen_manifest,
        protocol_label=args.selection_protocol_label,
    )
    protocol, frozen, language_counts = verify_protocol(args.root, verifier_args)
    normalize_text, normalization_protocol = load_normalizer(args.root)
    return {
        "protocol": protocol,
        "protocol_id": frozen["protocol_id"],
        "frozen_manifest": protocol / args.selection_frozen_manifest,
        "frozen_manifest_sha256": rescore_sha256(
            protocol / args.selection_frozen_manifest
        ),
        "label": args.selection_protocol_label,
        "language_counts": language_counts,
        "normalization_protocol": normalization_protocol,
        "labels": load_labels(protocol, language_counts),
        "normalize_text": normalize_text,
    }


def selection_metrics(
    report_dir: Path,
    raw_summary: dict[str, Any],
    selection_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the sole metric bundle allowed to control this stage."""
    if selection_context is None:
        return raw_summary

    from rescore_clean_dev_v2 import score_report

    metrics, _ = score_report(
        report_dir,
        selection_context["labels"],
        selection_context["normalize_text"],
    )
    return {
        "macro_cer": metrics["macro_cer"],
        "macro_wer": metrics["macro_wer"],
        "macro_one_minus_ned": metrics["macro_one_minus_ned"],
        "macro_line_accuracy": metrics["macro_line_accuracy"],
        "languages": metrics["languages"],
    }


def clean_outputs(args: argparse.Namespace) -> None:
    if args.run_dir.exists():
        shutil.rmtree(args.run_dir)
    eval_root = args.root / "04_model_training" / "eval_reports"
    for path in eval_root.glob(f"{args.eval_prefix}_dev_epoch_*"):
        if path.is_dir():
            shutil.rmtree(path)
    for suffix in ("best_clean_dev", "final_summary.json"):
        path = eval_root / f"{args.eval_prefix}_{suffix}"
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    # A stable log directory is required for audited resume. Starting a
    # replacement run must therefore clear only this stage's old ledger and
    # logs before the new validation/training records are written.
    for path in args.log_dir.glob(f"{args.stage}_*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def validate_config(args: argparse.Namespace) -> None:
    validators = {
        "rctc": "validate_p1_msr_config.py",
        "full_svtrv2": "validate_p1_full_svtrv2_config.py",
        "dual_order": "validate_dual_order_method_config.py",
    }
    validator = validators[args.config_kind]
    command = [
        sys.executable,
        str(args.root / "scripts" / "svtrv2" / validator),
        "--config",
        str(args.config),
        "--expected-initialization",
        args.expected_initialization,
    ]
    run_checked(
        command,
        cwd=args.root,
        log_path=args.log_dir / f"{args.stage}_p1_msr_validation.log",
    )


def audit_ddp_sampler(args: argparse.Namespace) -> Path:
    gpu_ids = [item.strip() for item in args.ddp_gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--ddp-gpus must contain at least one GPU")
    output = args.log_dir / f"{args.stage}_ddp_sampler_audit.json"
    audit_script = (
        "audit_hem_ratio_sampler_ddp.py"
        if args.sampler_audit_kind == "hem"
        else "audit_p1_ratio_sampler_ddp.py"
    )
    expected_status = (
        "HEM_RATIO_SAMPLER_DDP_AUDIT_OK"
        if args.sampler_audit_kind == "hem"
        else "P1_RATIO_SAMPLER_DDP_AUDIT_OK"
    )
    command = [
        sys.executable,
        str(args.root / "scripts" / "svtrv2" / audit_script),
        "--root",
        str(args.root),
        "--config",
        str(args.config),
        "--world-size",
        str(len(gpu_ids)),
        "--output",
        str(output),
    ]
    run_checked(
        command,
        cwd=args.root,
        log_path=args.log_dir / f"{args.stage}_ddp_sampler_audit.log",
    )
    report = json.loads(output.read_text(encoding="utf-8-sig"))
    if report.get("status") != expected_status:
        raise RuntimeError(f"DDP sampler audit did not pass: {report}")
    return output


def restore_stage_progress(
    args: argparse.Namespace,
    metrics_jsonl: Path,
    best_checkpoint: Path,
    best_json: Path,
    *,
    data_protocol_id: str,
    data_protocol_fingerprint: str,
    selection_binding: dict[str, Any] | None,
) -> dict[str, Any] | None:
    epoch_checkpoints = sorted(args.run_dir.glob("epoch_*.pth"))
    if not epoch_checkpoints and not metrics_jsonl.exists():
        return None
    if not metrics_jsonl.is_file():
        raise RuntimeError(
            f"Cannot safely resume {args.stage}: missing {metrics_jsonl}"
        )

    records = [
        json.loads(line)
        for line in metrics_jsonl.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(
            f"Cannot safely resume {args.stage}: no evaluated epochs recorded"
        )

    expected_epochs = list(
        range(args.eval_every, records[-1]["epoch"] + 1, args.eval_every)
    )
    actual_epochs = [int(record["epoch"]) for record in records]
    if actual_epochs != expected_epochs:
        raise RuntimeError(
            f"Non-contiguous evaluated epochs for {args.stage}: {actual_epochs}"
        )

    config_sha256 = sha256(args.config)
    best_cer = float("inf")
    best_epoch = 0
    bad_evals = 0
    best_record: dict[str, Any] | None = None
    for record in records:
        if record.get("stage") != args.stage:
            raise ValueError(f"Stage mismatch in {metrics_jsonl}: {record}")
        if record.get("config_sha256") != config_sha256:
            raise ValueError(f"Config hash mismatch in {metrics_jsonl}")
        if record.get("preprocess_protocol") != PROTOCOL_ID:
            raise ValueError(f"Preprocess protocol mismatch in {metrics_jsonl}")
        if record.get("data_protocol_id") != data_protocol_id:
            raise ValueError(f"Data protocol mismatch in {metrics_jsonl}")
        if record.get("data_protocol_fingerprint") != data_protocol_fingerprint:
            raise ValueError(f"Data fingerprint mismatch in {metrics_jsonl}")
        if record.get("selection_binding") != selection_binding:
            raise ValueError(
                f"External selection protocol mismatch in {metrics_jsonl}"
            )

        checkpoint_path = Path(record["checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if sha256(checkpoint_path) != record.get("checkpoint_sha256"):
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")

        current_cer = float(record["clean_dev_macro_cer"])
        previous_best_cer = best_cer
        if current_cer < best_cer:
            best_cer = current_cer
            best_epoch = int(record["epoch"])
            best_record = record

        if current_cer < previous_best_cer - args.min_delta:
            bad_evals = 0
        else:
            bad_evals += 1

    if best_record is None or not best_checkpoint.is_file() or not best_json.is_file():
        raise RuntimeError(f"Incomplete best-checkpoint state for {args.stage}")
    saved_best = json.loads(best_json.read_text(encoding="utf-8-sig"))
    if int(saved_best.get("epoch", -1)) != best_epoch:
        raise ValueError(f"Best epoch mismatch for {args.stage}")
    if sha256(best_checkpoint) != saved_best.get("copied_checkpoint_sha256"):
        raise ValueError(f"Best checkpoint hash mismatch for {args.stage}")

    last_epoch = actual_epochs[-1]
    resume_checkpoint = args.run_dir / f"epoch_{last_epoch}.pth"
    payload = load_checkpoint(resume_checkpoint)
    if int(payload["epoch"]) != last_epoch:
        raise ValueError(f"Resume checkpoint epoch mismatch: {resume_checkpoint}")
    validate_checkpoint_horizon(
        payload,
        max_epoch=args.max_epoch,
        checkpoint_path=resume_checkpoint,
    )

    return {
        "best_cer": best_cer,
        "best_epoch": best_epoch,
        "bad_evals": bad_evals,
        "evaluated_epochs": actual_epochs,
        "stopped_epoch": last_epoch,
        "resume_checkpoint": resume_checkpoint,
        "next_target_epoch": last_epoch + args.eval_every,
    }


def main() -> None:
    args = parse_args()
    if args.replace and args.resume_existing:
        raise ValueError("--replace and --resume-existing are mutually exclusive")
    args.root = args.root.resolve()
    args.config = args.config.resolve()
    args.run_dir = args.run_dir.resolve()
    args.labels_dir = args.labels_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    if args.selection_protocol_dir is not None:
        args.selection_protocol_dir = args.selection_protocol_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    data_protocol_manifest_path = (
        args.root
        / "00_docs"
        / "frozen_protocol_v2"
        / "protocol_v2_manifest.json"
    )
    data_protocol_manifest = json.loads(
        data_protocol_manifest_path.read_text(encoding="utf-8-sig")
    )
    data_protocol_id = str(data_protocol_manifest.get("protocol_id") or "")
    data_protocol_fingerprint = str(
        data_protocol_manifest.get("verification_fingerprint_sha256") or ""
    )
    if not data_protocol_id or not data_protocol_fingerprint:
        raise ValueError(
            f"Incomplete frozen data protocol: {data_protocol_manifest_path}"
        )
    selection_context = load_selection_context(args)
    selection_binding = None
    if selection_context is not None:
        selection_binding = {
            "protocol_id": selection_context["protocol_id"],
            "frozen_manifest": str(selection_context["frozen_manifest"]),
            "frozen_manifest_sha256": selection_context[
                "frozen_manifest_sha256"
            ],
            "normalization_protocol": selection_context[
                "normalization_protocol"
            ],
            "selection_label": selection_context["label"],
        }
    if args.replace:
        clean_outputs(args)
    elif (
        not args.resume_existing
        and args.run_dir.is_dir()
        and any(args.run_dir.glob("epoch_*.pth"))
    ):
        raise RuntimeError(
            f"Existing epoch checkpoints found in {args.run_dir}. "
            "Use --replace to start a controlled fresh stage; stale checkpoints "
            "are never consumed implicitly."
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    validate_config(args)
    sampler_audit_path = audit_ddp_sampler(args)

    openocr_root = args.root / "third_party" / "OpenOCR"
    eval_root = args.root / "04_model_training" / "eval_reports"
    train_log = args.log_dir / f"{args.stage}_train.log"
    metrics_jsonl = args.log_dir / f"{args.stage}_clean_dev_epoch_metrics.jsonl"
    best_checkpoint = args.run_dir / "best_clean_dev_macro_cer.pth"
    best_json = args.run_dir / "best_clean_dev_macro_cer.json"

    best_cer = float("inf")
    best_epoch = 0
    bad_evals = 0
    resume_checkpoint: Path | None = None
    evaluated_epochs: list[int] = []
    stopped_epoch = 0
    early_stopped = False
    first_target_epoch = args.eval_every

    if args.resume_existing:
        restored = restore_stage_progress(
            args,
            metrics_jsonl,
            best_checkpoint,
            best_json,
            data_protocol_id=data_protocol_id,
            data_protocol_fingerprint=data_protocol_fingerprint,
            selection_binding=selection_binding,
        )
        if restored is not None:
            best_cer = float(restored["best_cer"])
            best_epoch = int(restored["best_epoch"])
            bad_evals = int(restored["bad_evals"])
            evaluated_epochs = list(restored["evaluated_epochs"])
            stopped_epoch = int(restored["stopped_epoch"])
            resume_checkpoint = Path(restored["resume_checkpoint"])
            first_target_epoch = int(restored["next_target_epoch"])
            print(
                f"[{args.stage}] validated resume at epoch={stopped_epoch}; "
                f"next={first_target_epoch}; best={best_epoch} "
                f"CER={best_cer:.8f}; bad_evals={bad_evals}",
                flush=True,
            )
            prune_uncommitted_resume_outputs(args, stopped_epoch)

    for target_epoch in range(
        first_target_epoch,
        args.max_epoch + 1,
        args.eval_every,
    ):
        print(
            f"[{args.stage}] training to epoch {target_epoch}; "
            f"resume={resume_checkpoint}",
            flush=True,
        )
        process = start_ddp(
            args,
            openocr_root,
            resume_checkpoint,
            train_log,
            target_epoch,
        )
        checkpoint_path = args.run_dir / f"epoch_{target_epoch}.pth"
        try:
            checkpoint_payload = wait_for_checkpoint(checkpoint_path, process)
        finally:
            stop_process_group(process)
            close_process_log(process)
        if int(checkpoint_payload["epoch"]) != target_epoch:
            raise ValueError(
                f"Checkpoint epoch mismatch for {checkpoint_path}: "
                f"{checkpoint_payload['epoch']}"
            )
        validate_checkpoint_horizon(
            checkpoint_payload,
            max_epoch=args.max_epoch,
            checkpoint_path=checkpoint_path,
        )
        time.sleep(8)

        epoch_tag = f"{target_epoch:04d}"
        report_dir = eval_root / f"{args.eval_prefix}_dev_epoch_{epoch_tag}"
        raw_summary = evaluate_dev(
            args,
            checkpoint_path,
            report_dir,
            args.log_dir / f"{args.stage}_eval_epoch_{epoch_tag}.log",
        )
        summary = selection_metrics(report_dir, raw_summary, selection_context)
        current_cer = float(summary["macro_cer"])
        evaluated_epochs.append(target_epoch)
        stopped_epoch = target_epoch
        record = {
            "stage": args.stage,
            "epoch": target_epoch,
            "global_step": int(checkpoint_payload["global_step"]),
            "scheduler_last_epoch": int(
                checkpoint_payload["scheduler_last_epoch"]
            ),
            "scheduler_total_steps": checkpoint_payload[
                "scheduler_total_steps"
            ],
            "clean_dev_macro_cer": current_cer,
            "selection_metrics": summary,
            "raw_evaluator_macro_cer": float(raw_summary["macro_cer"]),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "report_dir": str(report_dir),
            "config_sha256": sha256(args.config),
            "preprocess_protocol": PROTOCOL_ID,
            "data_protocol_id": data_protocol_id,
            "data_protocol_fingerprint": data_protocol_fingerprint,
            "selection_binding": selection_binding,
        }
        with metrics_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        previous_best_cer = best_cer
        # Checkpoint selection must use the actual minimum CER.  min_delta is
        # only an early-stop tolerance; using it for selection can discard a
        # numerically better checkpoint and misreport the best epoch.
        if current_cer < best_cer:
            best_cer = current_cer
            best_epoch = target_epoch
            shutil.copy2(checkpoint_path, best_checkpoint)
            best_record = {
                **record,
                "copied_checkpoint": str(best_checkpoint),
                "copied_checkpoint_sha256": sha256(best_checkpoint),
                "selection_metric": "clean_target_dev_macro_CER",
            }
            best_json.write_text(
                json.dumps(best_record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[{args.stage}] new best epoch={best_epoch} "
                f"clean_dev_macro_CER={best_cer:.8f}",
                flush=True,
            )

        if current_cer < previous_best_cer - args.min_delta:
            bad_evals = 0
        else:
            bad_evals += 1
            if current_cer < previous_best_cer:
                print(
                    f"[{args.stage}] tiny best improvement below min_delta; "
                    f"patience={bad_evals}/{args.patience_evals}; "
                    f"best epoch={best_epoch} CER={best_cer:.8f}",
                    flush=True,
                )
            else:
                print(
                    f"[{args.stage}] no improvement {bad_evals}/"
                    f"{args.patience_evals}; best epoch={best_epoch} "
                    f"CER={best_cer:.8f}",
                    flush=True,
                )
        if (
            target_epoch >= args.min_epoch
            and bad_evals >= args.patience_evals
        ):
            print(f"[{args.stage}] early stop at epoch {target_epoch}", flush=True)
            early_stopped = True
            break
        resume_checkpoint = checkpoint_path

    if not best_checkpoint.is_file():
        raise RuntimeError(f"No best checkpoint selected for {args.stage}")
    best_report_dir = eval_root / f"{args.eval_prefix}_best_clean_dev"
    best_raw_summary = evaluate_dev(
        args,
        best_checkpoint,
        best_report_dir,
        args.log_dir / f"{args.stage}_best_clean_dev.log",
    )
    best_summary = selection_metrics(
        best_report_dir, best_raw_summary, selection_context
    )
    if abs(float(best_summary["macro_cer"]) - best_cer) > 1e-12:
        raise ValueError(
            "Best checkpoint selection metric changed after final evaluation"
        )
    final = {
        "experiment": args.stage,
        "checkpoint_selection": (
            f"argmin_epoch_{selection_context['label'].replace(' ', '_')}_Macro_CER"
            if selection_context is not None
            else "clean_target_dev_macro_CER"
        ),
        "config_kind": args.config_kind,
        "prediction_branch": args.prediction_branch,
        "preprocess_protocol": PROTOCOL_ID,
        "external_eval_batch_size": EVAL_INFERENCE_BATCH_SIZE,
        "data_protocol_id": data_protocol_id,
        "data_protocol_fingerprint": data_protocol_fingerprint,
        "data_protocol_manifest": str(data_protocol_manifest_path),
        "data_protocol_manifest_sha256": sha256(
            data_protocol_manifest_path
        ),
        "external_selection_protocol": selection_binding,
        "ddp_sampler_audit": str(sampler_audit_path),
        "ddp_sampler_audit_sha256": sha256(sampler_audit_path),
        "sampler_audit_kind": args.sampler_audit_kind,
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "best_checkpoint": str(best_checkpoint),
        "best_checkpoint_sha256": sha256(best_checkpoint),
        "best_epoch": best_epoch,
        "best_clean_dev_macro_cer": best_cer,
        "training_control": {
            "max_epoch": args.max_epoch,
            "scheduler_horizon_epoch": args.max_epoch,
            "segment_stop_mode": "clean_stop_after_checkpoint",
            "strict_lr_scheduler": True,
            "strict_epoch_length": True,
            "stopped_epoch": stopped_epoch,
            "early_stopped": early_stopped,
            "eval_every_epochs": args.eval_every,
            "patience_evals": args.patience_evals,
            "min_epoch": args.min_epoch,
            "min_delta": args.min_delta,
            "evaluated_epochs": evaluated_epochs,
            "metrics_jsonl": str(metrics_jsonl),
        },
        "dev": best_summary,
        "raw_evaluator_dev": best_raw_summary,
        "test_policy": "not_evaluated_during_model_development",
    }
    final_path = eval_root / f"{args.eval_prefix}_final_summary.json"
    final_path.write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
