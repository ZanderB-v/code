#!/usr/bin/env python3
"""Sequential controlled training for B2, M1, M2, M3, and Full."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from dual_order_protocol import (
    METHODS,
    METHOD_PROTOCOL,
    consistency_weight_path_tag,
)
from p1_msr_protocol import EVAL_INFERENCE_BATCH_SIZE, PROTOCOL_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=tuple(METHODS), default=["b2", "m1", "m2", "m3", "full"]
    )
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--synthetic-eval-every", type=int, default=5)
    parser.add_argument("--target-eval-every", type=int, default=2)
    parser.add_argument("--synthetic-patience", type=int, default=4)
    parser.add_argument("--target-patience", type=int, default=5)
    parser.add_argument("--synthetic-min-epoch", type=int, default=15)
    parser.add_argument("--target-min-epoch", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size-per-card", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--m2-consistency-weight", type=float)
    parser.add_argument(
        "--summary-name",
        default="dual_order_method_suite_clean_dev_summary",
    )
    parser.add_argument("--ddp-gpus", default="0,1")
    parser.add_argument("--eval-gpu", type=int, default=0)
    parser.add_argument("--master-port-base", type=int, default=29820)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume hash-validated stage outputs without deleting completed work.",
    )
    parser.add_argument("--rebuild-dual-order-lmdb", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def run(command: list[str], cwd: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def read_required_summary(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Controlled baseline summary is missing: {path}"
        )
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require_current_preprocess_protocol(
    root: Path,
    name: str,
    summary: dict,
    path: Path,
    checkpoint_path: Path,
) -> None:
    actual = summary.get("preprocess_protocol")
    if actual != PROTOCOL_ID:
        raise RuntimeError(
            f"{name} baseline uses preprocess_protocol={actual!r}, but the "
            f"method suite requires {PROTOCOL_ID!r}: {path}. Rebuild the "
            "formal RCTC baseline chain and B1 after synchronizing the fixed "
            "RatioSampler; checkpoints from older P1/MSR protocols are not "
            "comparable."
        )
    protocol_manifest = json.loads(
        (
            root
            / "00_docs"
            / "frozen_protocol_v2"
            / "protocol_v2_manifest.json"
        ).read_text(encoding="utf-8-sig")
    )
    config_path = Path(str(summary.get("config") or ""))
    expected = {
        "external_eval_batch_size": EVAL_INFERENCE_BATCH_SIZE,
        "data_protocol_fingerprint": protocol_manifest.get(
            "verification_fingerprint_sha256"
        ),
        "best_checkpoint_sha256": sha256(checkpoint_path),
        "config_sha256": sha256(config_path) if config_path.is_file() else None,
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if value is None or summary.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"{name} baseline provenance mismatch at {path}: "
            + json.dumps(mismatches, ensure_ascii=False)
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_comparison_csv(path: Path, baselines: dict, methods: dict) -> None:
    rows = []
    for name, summary in {**baselines, **methods}.items():
        dev = summary.get("dev") or {}
        languages = dev.get("languages") or {}
        rows.append(
            {
                "method": name,
                "best_epoch": summary.get("best_epoch"),
                "macro_cer": dev.get(
                    "macro_cer", summary.get("best_clean_dev_macro_cer")
                ),
                "macro_line_accuracy": dev.get("macro_line_accuracy"),
                "zh_cer": (languages.get("zh") or {}).get("cer"),
                "ug_cer": (languages.get("ug") or {}).get("cer"),
                "kk_cer": (languages.get("kk") or {}).get("cer"),
                "checkpoint_selection": summary.get("checkpoint_selection"),
                "test_policy": summary.get("test_policy"),
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(
    args,
    method,
    stage,
    checkpoint,
    checkpoint_role,
    seed,
    rebuild_lmdb,
):
    command = [
        sys.executable,
        str(args.root / "scripts/svtrv2/prepare_dual_order_method.py"),
        "--root",
        str(args.root),
        "--method",
        method,
        "--stage",
        stage,
        "--source-checkpoint",
        str(checkpoint),
        "--source-role",
        checkpoint_role,
        "--max-epoch",
        str(args.max_epoch),
        "--batch-size-per-card",
        str(args.batch_size_per_card),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--seed",
        str(seed),
    ]
    if rebuild_lmdb:
        command.append("--replace-dual-order-lmdb")
    if method == "m2" and args.m2_consistency_weight is not None:
        command.extend(
            [
                "--consistency-weight",
                str(args.m2_consistency_weight),
            ]
        )
    run(command, args.root)


def model_name(args, method, stage):
    method_name = f"{method}_dual_order"
    if method == "m2" and args.m2_consistency_weight is not None:
        method_name = (
            "m2_alpha_"
            + consistency_weight_path_tag(args.m2_consistency_weight)
        )
    suffix = "s50" if stage == "synthetic" else "s50_to_target"
    return f"svtrv2_s_{method_name}_{suffix}"


def train_stage(
    args,
    method,
    stage,
    config,
    run_dir,
    eval_every,
    patience,
    min_epoch,
    port,
):
    command = [
        sys.executable,
        str(args.root / "scripts/svtrv2/run_p1_rctc_stage.py"),
        "--root",
        str(args.root),
        "--stage",
        stage,
        "--config",
        str(config),
        "--run-dir",
        str(run_dir),
        "--labels-dir",
        str(
            args.root / "04_model_training/datasets/e1_target_only/labels"
        ),
        "--label-prefix",
        "target",
        "--eval-prefix",
        stage,
        "--expected-initialization",
        "checkpoint",
        "--config-kind",
        "dual_order",
        "--prediction-branch",
        "ctc",
        "--max-epoch",
        str(args.max_epoch),
        "--eval-every",
        str(eval_every),
        "--patience-evals",
        str(patience),
        "--min-epoch",
        str(min_epoch),
        "--min-delta",
        str(args.min_delta),
        "--ddp-gpus",
        args.ddp_gpus,
        "--eval-gpu",
        str(args.eval_gpu),
        "--master-port",
        str(port),
        "--log-dir",
        str(args.log_dir),
    ]
    if args.replace:
        command.append("--replace")
    elif getattr(args, "resume_existing", False):
        command.append("--resume-existing")
    run(command, args.root)


def smoke(args, method, config, stage):
    sampler_audit_path = (
        args.log_dir / f"sampler_audit_{method}_{stage}.json"
    )
    if sampler_audit_path.exists():
        sampler_audit_path.unlink()
    run(
        [
            sys.executable,
            str(
                args.root
                / "scripts"
                / "svtrv2"
                / "audit_p1_ratio_sampler_ddp.py"
            ),
            "--root",
            str(args.root),
            "--config",
            str(config),
            "--world-size",
            str(len([item for item in args.ddp_gpus.split(",") if item.strip()])),
            "--output",
            str(sampler_audit_path),
        ],
        args.root,
    )
    sampler_report = json.loads(
        sampler_audit_path.read_text(encoding="utf-8-sig")
    )
    if sampler_report.get("status") != "P1_RATIO_SAMPLER_DDP_AUDIT_OK":
        raise RuntimeError(
            f"DDP sampler audit did not pass for {method}/{stage}: "
            f"{sampler_report}"
        )
    output_path = args.log_dir / f"preflight_{method}_{stage}.json"
    if output_path.exists():
        output_path.unlink()
    command = [
        sys.executable,
        str(args.root / "scripts/svtrv2/smoke_dual_order_method.py"),
        "--root",
        str(args.root),
        "--config",
        str(config),
        "--device-id",
        str(args.eval_gpu),
        "--output",
        str(output_path),
    ]
    if stage == "target":
        command.append("--require-full-initialization")
        if method in ("dir_d1_v2", "dir_d2_v2"):
            command.append("--allow-new-semantic-parameters")
    run(command, args.root)
    if not output_path.is_file():
        raise RuntimeError(
            f"Smoke command exited without producing its report: {output_path}"
        )
    report = json.loads(output_path.read_text(encoding="utf-8-sig"))
    if report.get("status") != "DUAL_ORDER_REAL_BATCH_FORWARD_BACKWARD_OK":
        raise RuntimeError(
            f"Smoke report did not pass for {method}/{stage}: {report}"
        )


def main() -> None:
    args = parse_args()
    if args.replace and args.resume_existing:
        raise ValueError("--replace and --resume-existing are mutually exclusive")
    args.root = args.root.resolve()
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    if args.m2_consistency_weight is not None:
        if "m2" not in args.methods:
            raise ValueError(
                "--m2-consistency-weight requires --methods to include m2"
            )
        if not 0.0 <= args.m2_consistency_weight <= 1.0:
            raise ValueError("--m2-consistency-weight must be in [0, 1]")
    if not args.summary_name or Path(args.summary_name).name != args.summary_name:
        raise ValueError("--summary-name must be a plain file stem")
    run(
        [
            sys.executable,
            str(args.root / "scripts/protocol/freeze_protocol_v2.py"),
            "--root",
            str(args.root),
            "--mode",
            "verify",
            "--workers",
            "8",
        ],
        args.root,
    )
    run(
        [
            sys.executable,
            str(args.root / "scripts/svtrv2/test_dual_order_components.py"),
            "--root",
            str(args.root),
            "--output",
            str(args.log_dir / "dual_order_component_invariants.json"),
        ],
        args.root,
    )

    configs = args.root / "04_model_training/configs"
    runs = args.root / "04_model_training/runs"
    eval_root = args.root / "04_model_training/eval_reports"
    b0_summary_path = eval_root / "e5_d2_to_target_final_summary.json"
    b1_summary_path = (
        eval_root / "b1_full_s50_to_target_final_summary.json"
    )
    controlled_baselines = {
        "b0": read_required_summary(b0_summary_path),
        "b1": read_required_summary(b1_summary_path),
    }
    baseline_checkpoints = {
        "b0": runs / "svtrv2_s_e5_d2_to_target/best_clean_dev_macro_cer.pth",
        "b1": runs / "svtrv2_s_b1_full_s50_to_target/best_clean_dev_macro_cer.pth",
    }
    for baseline_name, baseline_path in (
        ("b0", b0_summary_path),
        ("b1", b1_summary_path),
    ):
        checkpoint_path = baseline_checkpoints[baseline_name]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        require_current_preprocess_protocol(
            args.root,
            baseline_name,
            controlled_baselines[baseline_name],
            baseline_path,
            checkpoint_path,
        )
    adapter_checkpoint = (
        args.root
        / "04_model_training/checkpoint_adapters/b1_full_s50"
        / "d2_rctc_for_full_svtrv2.pth"
    )
    d2_best = runs / "svtrv2_s_d2_synth50k/best_clean_dev_macro_cer.pth"
    if not d2_best.is_file():
        raise FileNotFoundError(d2_best)
    conversion_report = adapter_checkpoint.with_suffix(".conversion.json")
    conversion_is_current = False
    if adapter_checkpoint.is_file() and conversion_report.is_file():
        report = json.loads(conversion_report.read_text(encoding="utf-8-sig"))
        conversion_is_current = (
            report.get("input_sha256") == sha256(d2_best)
            and report.get("output_sha256") == sha256(adapter_checkpoint)
        )
    if not conversion_is_current:
        command = [
            sys.executable,
            str(
                args.root
                / "scripts/svtrv2/convert_rctc_checkpoint_to_full_svtrv2.py"
            ),
            "--input",
            str(d2_best),
            "--output",
            str(adapter_checkpoint),
            "--replace",
        ]
        run(command, args.root)

    preflight = {}
    rebuild_pending = args.rebuild_dual_order_lmdb
    for method in args.methods:
        synthetic_name = model_name(args, method, "synthetic")
        synthetic_config = configs / f"{synthetic_name}.yml"
        prepare(
            args,
            method,
            "synthetic",
            adapter_checkpoint,
            "same_B0_D2_RCTC_checkpoint_remapped_to_nested_CTC",
            20260731,
            rebuild_pending,
        )
        rebuild_pending = False
        smoke(args, method, synthetic_config, "synthetic")
        preflight[method] = str(
            args.log_dir / f"preflight_{method}_synthetic.json"
        )
    print(
        json.dumps(
            {
                "status": "ALL_DUAL_ORDER_METHOD_PREFLIGHTS_OK",
                "preflight": preflight,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.preflight_only:
        return

    results = {}
    target_smoke_reports = {}
    port = args.master_port_base
    for method in args.methods:
        seed = 20260731
        synthetic_name = model_name(args, method, "synthetic")
        synthetic_config = configs / f"{synthetic_name}.yml"
        synthetic_run = runs / synthetic_name
        prepare(
            args,
            method,
            "synthetic",
            adapter_checkpoint,
            "same_B0_D2_RCTC_checkpoint_remapped_to_nested_CTC",
            seed,
            False,
        )
        train_stage(
            args,
            method,
            synthetic_name,
            synthetic_config,
            synthetic_run,
            args.synthetic_eval_every,
            args.synthetic_patience,
            args.synthetic_min_epoch,
            port,
        )
        port += 1
        synthetic_best = synthetic_run / "best_clean_dev_macro_cer.pth"
        if not synthetic_best.is_file():
            raise FileNotFoundError(synthetic_best)

        target_name = model_name(args, method, "target")
        target_config = configs / f"{target_name}.yml"
        target_run = runs / target_name
        prepare(
            args,
            method,
            "target",
            synthetic_best,
            f"{synthetic_name}_best_clean_target_dev_CTC",
            seed,
            False,
        )
        smoke(args, method, target_config, "target")
        target_smoke_reports[method] = str(
            args.log_dir / f"preflight_{method}_target.json"
        )
        train_stage(
            args,
            method,
            target_name,
            target_config,
            target_run,
            args.target_eval_every,
            args.target_patience,
            args.target_min_epoch,
            port,
        )
        port += 1
        final_path = eval_root / f"{target_name}_final_summary.json"
        results[method] = json.loads(
            final_path.read_text(encoding="utf-8-sig")
        )

    summary = {
        "experiment": "dual_order_method_suite",
        "method_protocol": METHOD_PROTOCOL,
        "preprocess_protocol": PROTOCOL_ID,
        "external_dev_inference_batch_size": EVAL_INFERENCE_BATCH_SIZE,
        "methods": args.methods,
        "m2_consistency_weight": args.m2_consistency_weight,
        "controlled_initialization": (
            "same converted B0/D2 RCTC checkpoint for every synthetic stage"
        ),
        "preflight_reports": preflight,
        "target_stage_smoke_reports": target_smoke_reports,
        "controlled_baselines": controlled_baselines,
        "checkpoint_selection": "clean_target_dev_macro_CER",
        "test_policy": "not_evaluated",
        "results": results,
    }
    output = eval_root / f"{args.summary_name}.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_comparison_csv(
        eval_root / f"{args.summary_name}.csv",
        controlled_baselines,
        results,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
