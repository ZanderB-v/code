#!/usr/bin/env python3
"""Verify the final M3 consistency weight on the fixed four-point grid."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import yaml

from run_dual_order_method_suite import run, sha256
from run_semantic_direction_study import checked_summary, local_path, write
from run_semantic_direction_v2 import export_epochs


STUDY = "m3_alpha_verification_v1"
NEW_METHODS = ("m3_alpha020", "m3_alpha025")
TARGET_STEMS = {
    0.15: "svtrv2_s_m3_dual_order_s50_to_target",
    0.20: "svtrv2_s_m3_alpha020_dual_order_s50_to_target",
    0.25: "svtrv2_s_m3_alpha025_dual_order_s50_to_target",
    0.30: "M3_anchor_alpha030_v2",
}
SYNTHETIC_STEMS = {
    0.15: "svtrv2_s_m3_dual_order_s50",
    0.20: "svtrv2_s_m3_alpha020_dual_order_s50",
    0.25: "svtrv2_s_m3_alpha025_dual_order_s50",
    0.30: "M3_shared_S50_alpha030_v2",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalized_config(path: Path) -> dict:
    """Remove only experiment identity, source pointer, and tested alpha."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    normalized = copy.deepcopy(cfg)
    for key in (
        "output_dir",
        "project_name",
        "save_res_path",
        "pretrained_model",
        "method_variant",
        "model_protocol",
    ):
        normalized["Global"][key] = "<CONTROLLED>"
    normalized["Loss"]["consistency_weight"] = "<TESTED_ALPHA>"
    return normalized


def select_alpha(rows: list[dict]) -> dict:
    if [row["alpha"] for row in rows] != [0.15, 0.20, 0.25, 0.30]:
        raise ValueError("The M3 alpha grid must be exactly .15/.20/.25/.30")
    return min(rows, key=lambda row: (row["macro_cer"], row["alpha"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_row(alpha: float, selected: dict, summary: dict) -> dict:
    if not math.isclose(float(summary["audited_alpha"]), alpha, abs_tol=1e-12):
        raise ValueError(f"Config alpha mismatch for M3 alpha={alpha:.2f}")
    return {
        "alpha": alpha,
        "epoch": selected["epoch"],
        "macro_cer": selected["macro_cer"],
        "macro_wer": selected["macro_wer"],
        "macro_1_ned": selected["macro_1_ned"],
        "macro_line_accuracy": selected["macro_line_accuracy"],
        "zh_cer": selected["zh_cer"],
        "ug_cer": selected["ug_cer"],
        "kk_cer": selected["kk_cer"],
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "config": summary["config"],
        "config_sha256": summary["config_sha256"],
        "selection_rule": "minimum_clean_dev_macro_CER_over_all_saved_epochs",
    }


def verify_controls(root: Path, target: dict[float, dict], synthetic: dict[float, dict]) -> dict:
    target_fingerprints = {
        alpha: normalized_config(local_path(root, summary["config"]))
        for alpha, summary in target.items()
    }
    synthetic_fingerprints = {
        alpha: normalized_config(local_path(root, summary["config"]))
        for alpha, summary in synthetic.items()
    }
    if len({json.dumps(value, sort_keys=True) for value in target_fingerprints.values()}) != 1:
        raise ValueError("Target configs differ outside identity, source, and alpha")
    if len({json.dumps(value, sort_keys=True) for value in synthetic_fingerprints.values()}) != 1:
        raise ValueError("Synthetic configs differ outside identity, source, and alpha")

    control_keys = (
        "max_epoch",
        "eval_every_epochs",
        "patience_evals",
        "min_epoch",
        "min_delta",
        "scheduler_horizon_epoch",
        "strict_lr_scheduler",
        "strict_epoch_length",
    )
    for stage_name, summaries in (("target", target), ("synthetic", synthetic)):
        reference = {
            key: summaries[0.15]["training_control"].get(key)
            for key in control_keys
        }
        for alpha, summary in summaries.items():
            actual = {key: summary["training_control"].get(key) for key in control_keys}
            if actual != reference:
                raise ValueError(f"{stage_name} training control drift at alpha={alpha:.2f}")
        if len({summary["data_protocol_fingerprint"] for summary in summaries.values()}) != 1:
            raise ValueError(f"{stage_name} data protocol fingerprint drift")
        if len({summary["external_eval_batch_size"] for summary in summaries.values()}) != 1:
            raise ValueError(f"{stage_name} evaluator batch-size drift")

    synthetic_sources = {}
    for alpha, summary in synthetic.items():
        cfg = yaml.safe_load(local_path(root, summary["config"]).read_text(encoding="utf-8-sig"))
        source = local_path(root, cfg["Global"]["pretrained_model"])
        synthetic_sources[alpha] = sha256(source)
    if len(set(synthetic_sources.values())) != 1:
        raise ValueError("M3 S50 stages do not share the same converted RCTC initialization")

    for alpha, summary in target.items():
        cfg = yaml.safe_load(local_path(root, summary["config"]).read_text(encoding="utf-8-sig"))
        target_source = local_path(root, cfg["Global"]["pretrained_model"])
        if sha256(target_source) != synthetic[alpha]["best_checkpoint_sha256"]:
            raise ValueError(f"Target alpha={alpha:.2f} is not initialized from its matching S50 stage")
        if cfg["Global"].get("seed") != 20260731:
            raise ValueError("Canonical seed changed")
    return {
        "same_s50_data_and_training_config": True,
        "same_target_data_and_training_config": True,
        "same_external_training_controls": True,
        "same_data_protocol_fingerprint": True,
        "same_clean_dev_evaluator_batch_size": True,
        "same_converted_rctc_initialization_sha256": next(iter(synthetic_sources.values())),
        "canonical_seed": 20260731,
        "preprocess_protocol": "P1_MSR_V3",
        "data_protocol": "Protocol V2",
    }


def plot_grid(output: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = [row["alpha"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(alphas, [100 * row["macro_cer"] for row in rows], marker="o")
    axes[0].set(xlabel="Consistency weight alpha", ylabel="Macro CER (%)")
    axes[1].plot(alphas, [100 * row["macro_line_accuracy"] for row in rows], marker="o")
    axes[1].set(xlabel="Consistency weight alpha", ylabel="Macro Line Accuracy (%)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xticks(alphas)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"m3_alpha_verification.{extension}", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.eval_gpu = 0
    root = args.root
    output = root / "04_model_training/eval_reports" / STUDY

    historical_target = checked_summary(root, TARGET_STEMS[0.15])
    historical_synthetic = checked_summary(root, SYNTHETIC_STEMS[0.15])
    alpha030_target = checked_summary(root, TARGET_STEMS[0.30])
    alpha030_synthetic = checked_summary(root, SYNTHETIC_STEMS[0.30])
    if historical_target["audited_alpha"] != 0.15 or alpha030_target["audited_alpha"] != 0.30:
        raise ValueError("Existing M3 alpha anchors are mislabeled")

    bound_files = (
        "scripts/svtrv2/run_m3_alpha_verification.py",
        "scripts/svtrv2/run_dual_order_method_suite.py",
        "scripts/svtrv2/prepare_dual_order_method.py",
        "scripts/svtrv2/dual_order_protocol.py",
        "scripts/svtrv2/validate_dual_order_method_config.py",
        "scripts/svtrv2/run_p1_rctc_stage.py",
        "scripts/svtrv2/evaluate_d2_checkpoint.py",
        "scripts/svtrv2/infer_label_file_metrics.py",
        "scripts/svtrv2/run_semantic_direction_v2.py",
        "third_party/OpenOCR/openrec/losses/dual_order_gtc_loss.py",
    )
    binding = {
        "study": STUDY,
        "candidate_alphas": [0.15, 0.20, 0.25, 0.30],
        "new_training_alphas": [0.20, 0.25],
        "selection": "argmin_alpha_clean_dev_macro_CER",
        "code_sha256": {path: sha256(root / path) for path in bound_files},
        "historical_alpha015_summary_sha256": historical_target["audited_source_sha256"],
        "matched_alpha030_summary_sha256": alpha030_target["audited_source_sha256"],
        "corrupted_dev_used": False,
        "test_evaluated": False,
    }
    binding_path = args.log_dir / "study_binding.json"
    if binding_path.exists() and read(binding_path) != binding:
        raise ValueError("Study code or fixed anchors changed during resume")
    write(binding_path, binding)

    control = historical_target["training_control"]
    synthetic_control = historical_synthetic["training_control"]
    template_cfg = yaml.safe_load(local_path(root, historical_target["config"]).read_text(encoding="utf-8-sig"))
    suite_command = [
        sys.executable,
        str(root / "scripts/svtrv2/run_dual_order_method_suite.py"),
        "--root", str(root),
        "--log-dir", str(args.log_dir / "new_runs"),
        "--methods", *NEW_METHODS,
        "--summary-name", "m3_alpha_verification_new_runs",
        "--max-epoch", str(control["max_epoch"]),
        "--synthetic-eval-every", str(synthetic_control["eval_every_epochs"]),
        "--target-eval-every", str(control["eval_every_epochs"]),
        "--synthetic-patience", str(synthetic_control["patience_evals"]),
        "--target-patience", str(control["patience_evals"]),
        "--synthetic-min-epoch", str(synthetic_control["min_epoch"]),
        "--target-min-epoch", str(control["min_epoch"]),
        "--min-delta", str(control["min_delta"]),
        "--batch-size-per-card", str(template_cfg["Train"]["sampler"]["first_bs"]),
        "--num-workers", str(template_cfg["Train"]["loader"]["num_workers"]),
        "--lr", str(template_cfg["Optimizer"]["lr"]),
        "--ddp-gpus", "0,1",
        "--eval-gpu", "0",
        "--master-port-base", "29970",
        "--resume-existing",
    ]
    if args.preflight_only:
        run([*suite_command, "--preflight-only"], root)
        print("M3_ALPHA_VERIFICATION_PREFLIGHT_OK; no training was started.", flush=True)
        return
    if not args.summarize_only:
        run(suite_command, root)

    target = {0.15: historical_target, 0.30: alpha030_target}
    synthetic = {0.15: historical_synthetic, 0.30: alpha030_synthetic}
    for alpha in (0.20, 0.25):
        target[alpha] = checked_summary(root, TARGET_STEMS[alpha])
        synthetic[alpha] = checked_summary(root, SYNTHETIC_STEMS[alpha])
    controls = verify_controls(root, target, synthetic)

    selected = {}
    histories = {}
    for alpha in (0.15, 0.20, 0.25):
        label = f"M3_alpha{int(round(alpha * 100)):03d}"
        selected[alpha], histories[alpha] = export_epochs(args, label, target[alpha], output)
    semantic_output = root / "04_model_training/eval_reports/semantic_direction_v2"
    selected[0.30], histories[0.30] = export_epochs(
        args, "M3", target[0.30], semantic_output
    )

    rows = [selected_row(alpha, selected[alpha], target[alpha]) for alpha in (0.15, 0.20, 0.25, 0.30)]
    winner = select_alpha(rows)
    decision = {
        "status": "M3_ALPHA_VERIFICATION_COMPLETE",
        "model": "SOAR-SVTR / M3",
        "candidate_alphas": [0.15, 0.20, 0.25, 0.30],
        "new_training_alphas": [0.20, 0.25],
        "selected_alpha": winner["alpha"],
        "selected_checkpoint": winner["checkpoint"],
        "selected_checkpoint_sha256": winner["checkpoint_sha256"],
        "primary_metric": "Clean Dev Macro CER",
        "tie_breaking": "smaller_alpha",
        "ablation_alpha": 0.30,
        "ablation_reason": "Keep alpha=.30 fixed for clean M2-to-M3 module attribution.",
        "direction_branch": "closed; D1 and D2 remain diagnostic experiments",
        "further_alpha_search_allowed": False,
        "controls": controls,
        "rows": rows,
        "corrupted_dev_used": False,
        "test_evaluated": False,
    }
    write(output / "m3_alpha_verification_summary.json", decision)
    write_csv(output / "m3_alpha_verification_summary.csv", rows)
    lock_path = output / "final_alpha_lock.json"
    lock = {
        "status": "FINAL_SOAR_SVTR_ALPHA_LOCKED",
        "selected_alpha": winner["alpha"],
        "selection_rule": "argmin over the predeclared .15/.20/.25/.30 Clean Dev Macro CER grid",
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "further_alpha_search_allowed": False,
        "test_evaluated": False,
    }
    if lock_path.exists() and read(lock_path) != lock:
        raise ValueError("An incompatible final alpha lock already exists")
    write(lock_path, lock)
    plot_grid(output, rows)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    print("M3_ALPHA_VERIFICATION_COMPLETE; Direction branch closed; Test was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
