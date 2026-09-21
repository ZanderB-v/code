#!/usr/bin/env python3
"""Controlled M3 alpha=.30, D1, conditional D2, and Clean Dev selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import yaml

from p1_msr_protocol import PROTOCOL_ID
from run_dual_order_method_suite import (
    model_name, prepare, require_current_preprocess_protocol, run, sha256,
    smoke, train_stage,
)

STUDY = "semantic_direction_v1"
METHODS = ("m3_alpha030", "dir_d1", "dir_d2")
HISTORICAL = {
    "B1": "b1_full_s50_to_target",
    "M1": "svtrv2_s_m1_dual_order_s50_to_target",
    "M2": "svtrv2_s_m2_alpha_030_s50_to_target",
    "M3 (alpha=0.15, historical)": "svtrv2_s_m3_dual_order_s50_to_target",
    "M3 + LDC (Diagnostic Ablation)": "svtrv2_s_full_dual_order_s50_to_target",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def local_path(root, raw):
    # Synchronized reports can use either /data_home or /home/data_home.
    raw = str(raw).replace("\\", "/")
    marker = "svtrv2_line_recognition/"
    if marker in raw:
        return root / raw.split(marker, 1)[1]
    path = Path(raw)
    return path if path.is_absolute() else root / path


def metric_row(name, summary):
    dev = summary["dev"]
    languages = dev["languages"]
    ctc_branch = dev.get("prediction_branch") == "ctc" or (
        dev.get("prediction_branch") == "auto" and summary.get("config_kind") == "rctc")
    if dev.get("split") != "dev" or not ctc_branch:
        raise ValueError(f"{name}: requires Clean Dev CTC predictions")
    if summary.get("checkpoint_selection") != "clean_target_dev_macro_CER":
        raise ValueError(f"{name}: invalid checkpoint selection rule")
    row = {"model": name, "epoch": summary["best_epoch"]}
    for output, key in (("macro_cer", "cer"), ("macro_wer", "wer"),
                        ("macro_1_ned", "one_minus_ned_macro"),
                        ("macro_line_accuracy", "line_accuracy")):
        values = [float(languages[lang][key]) for lang in ("zh", "ug", "kk")]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{name}: non-finite {key}")
        row[output] = sum(values) / 3
    if not math.isclose(row["macro_cer"], summary["best_clean_dev_macro_cer"], abs_tol=1e-12):
        raise ValueError(f"{name}: selected CER and report disagree")
    for key in ("macro_cer", "macro_line_accuracy"):
        if not math.isclose(row[key], dev[key], abs_tol=1e-12):
            raise ValueError(f"{name}: language macro and {key} disagree")
    row.update({f"{lang}_cer": languages[lang]["cer"] for lang in ("zh", "ug", "kk")})
    row["checkpoint_sha256"] = summary["best_checkpoint_sha256"]
    return row


def checked_summary(root, stem):
    path = root / "04_model_training/eval_reports" / f"{stem}_final_summary.json"
    summary = read(path)
    if summary.get("test_policy") != "not_evaluated_during_model_development":
        raise ValueError(f"Development-only policy missing: {path}")
    checkpoint = local_path(root, summary["best_checkpoint"])
    summary["config"] = str(local_path(root, summary["config"]))
    require_current_preprocess_protocol(root, stem, summary, path, checkpoint)
    metric_row(stem, summary)
    selected = read(checkpoint.with_suffix(".json"))
    if (selected["epoch"] != summary["best_epoch"]
            or selected["checkpoint_sha256"] != summary["best_checkpoint_sha256"]
            or not math.isclose(selected["clean_dev_macro_cer"],
                                summary["best_clean_dev_macro_cer"], abs_tol=1e-12)):
        raise ValueError(f"Selection/checkpoint summary mismatch: {path}")
    history_path = local_path(root, summary["training_control"]["metrics_jsonl"])
    history = [json.loads(line) for line in history_path.read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    if not history or [r["epoch"] for r in history] != summary["training_control"]["evaluated_epochs"]:
        raise ValueError(f"Missing or inconsistent evaluated epochs: {history_path}")
    best = min(history, key=lambda r: (r["clean_dev_macro_cer"], r["epoch"]))
    if (best["epoch"] != summary["best_epoch"]
            or best["checkpoint_sha256"] != summary["best_checkpoint_sha256"]):
        raise ValueError(f"Selected checkpoint is not the actual minimum CER: {history_path}")
    report = local_path(root, summary["dev"]["output_dir"]) / "metrics_macro_summary.json"
    current = read(report)
    for lang in ("zh", "ug", "kk"):
        for key in ("samples", "chars", "edit_distance", "cer", "wer",
                    "one_minus_ned_macro", "line_accuracy"):
            if current["languages"][lang][key] != summary["dev"]["languages"][lang][key]:
                raise ValueError(f"Stale or mixed reports: {report}, {lang}.{key}")
    cfg = yaml.safe_load(Path(summary["config"]).read_text(encoding="utf-8-sig"))
    if (dev_cfg_hash := current.get("config_sha256")) != summary["config_sha256"]:
        raise ValueError(f"Evaluation config hash mismatch: {report}: {dev_cfg_hash}")
    summary["audited_alpha"] = cfg["Loss"].get("consistency_weight", 0.0)
    summary["audited_source_sha256"] = sha256(path)
    return summary


def select_structure(rows):
    baseline = next(row for row in rows if row["model"] == "B1")
    eligible = [row for row in rows if row.get("candidate", False)
                and row["macro_cer"] < baseline["macro_cer"]
                and row["macro_line_accuracy"] > baseline["macro_line_accuracy"]]
    # Stable ties prefer the earlier, simpler candidate; LineAcc is not a tie breaker.
    winner = min(eligible, key=lambda row: row["macro_cer"]) if eligible else None
    return {"selected_model": winner["model"] if winner else None,
            "status": "candidate_selected" if winner else "no_candidate_passes_guardrail",
            "primary": "clean_target_dev_macro_CER",
            "guardrail": "CER < B1 AND Macro Line Accuracy > B1",
            "test_evaluated": False}


def gates(checkpoint):
    import torch
    state = torch.load(checkpoint, map_location="cpu")["state_dict"]
    logits = state["decoder.direction_gate_logits"].detach().float()
    if not torch.isfinite(logits).all() or tuple(logits.shape) != (4,):
        raise ValueError("Invalid learned script gate parameters")
    return {"initial_gate_all_scripts": 0.01,
            "learned_sigmoid_gates": dict(zip(
                ("Han", "Arabic", "Cyrillic", "Common_Latin"), logits.sigmoid().tolist())),
            "interpretation": "Learned gates are descriptive, not causal proof of usefulness."}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--ddp-gpus", default="0,1")
    p.add_argument("--eval-gpu", type=int, default=0)
    args = p.parse_args()
    # Controlled constants: no alpha/gate sweep is exposed by this entry point.
    args.max_epoch = 50
    args.batch_size_per_card = 16
    args.num_workers = 4
    args.lr = 2.5e-5
    args.min_delta = 1e-4
    args.replace = False
    args.resume_existing = True
    args.m2_consistency_weight = None
    return args


def main():
    args = parse_args()
    args.root = args.root.resolve()
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    root = args.root
    output = root / "04_model_training/eval_reports" / STUDY
    historical = {name: checked_summary(root, stem) for name, stem in HISTORICAL.items()}
    if historical["M2"]["audited_alpha"] != 0.30:
        raise ValueError("M2 reference must use the locked alpha=0.30")
    parent_cfg = yaml.safe_load(Path(historical[
        "M3 (alpha=0.15, historical)"]["config"]).read_text(encoding="utf-8-sig"))
    expected = {"seed": 20260731, "epoch_num": 50}
    for key, value in expected.items():
        if parent_cfg["Global"].get(key) != value:
            raise ValueError(f"M3 controlled training constant changed: Global.{key}")
    if (parent_cfg["Optimizer"]["lr"] != args.lr
            or parent_cfg["Train"]["sampler"]["first_bs"] != args.batch_size_per_card):
        raise ValueError("M3 optimizer LR or per-device batch differs from direction study")

    source = root / "04_model_training/checkpoint_adapters/b1_full_s50/d2_rctc_for_full_svtrv2.pth"
    conversion = read(source.with_suffix(".conversion.json"))
    d2 = checked_summary(root, "d2_synth50k")
    if (conversion["input_sha256"] != d2["best_checkpoint_sha256"]
            or conversion["output_sha256"] != sha256(source)):
        raise ValueError("D2 RCTC initialization conversion is stale")

    bound_files = [
        "scripts/svtrv2/run_semantic_direction_study.py",
        "scripts/svtrv2/dual_order_protocol.py",
        "scripts/svtrv2/run_p1_rctc_stage.py",
        "scripts/svtrv2/test_semantic_direction.py",
        "third_party/OpenOCR/openrec/modeling/decoders/dual_order_gtc_decoder.py",
        "third_party/OpenOCR/openrec/losses/dual_order_gtc_loss.py",
    ]
    binding = {"study": STUDY, "alpha": 0.30, "initial_gate": 0.01,
               "seed": 20260731, "source_sha256": sha256(source),
               "preprocess_protocol": PROTOCOL_ID,
               "code_sha256": {p: sha256(root / p) for p in bound_files},
               "reference_sha256": {n: s["audited_source_sha256"] for n, s in historical.items()},
               "ddp_gpus": args.ddp_gpus,
               "test_policy": "not_evaluated"}
    binding_path = args.log_dir / "study_binding.json"
    if binding_path.exists() and read(binding_path) != binding:
        raise ValueError("Study inputs changed; use a new log directory for a new revision")
    write(binding_path, binding)
    if not args.summarize_only:
        run([sys.executable, str(root / "scripts/protocol/freeze_protocol_v2.py"),
             "--root", str(root), "--mode", "verify", "--workers", "8"], root)
        run([sys.executable, str(root / "scripts/svtrv2/test_semantic_direction.py"),
             "--root", str(root)], root)
        for method in METHODS:
            prepare(args, method, "synthetic", source,
                    "same_B0_D2_RCTC_checkpoint_remapped_to_nested_CTC", 20260731, False)
            config = root / "04_model_training/configs" / f"{model_name(args, method, 'synthetic')}.yml"
            run([sys.executable, str(root / "scripts/svtrv2/validate_dual_order_method_config.py"),
                 "--config", str(config)], root)
            smoke(args, method, config, "synthetic")
        print("SEMANTIC_DIRECTION_PREFLIGHT_OK", flush=True)
        if args.preflight_only:
            return

    results = {}
    for index, method in enumerate(METHODS):
        if method == "dir_d2" and not (
            results["dir_d1"]["best_clean_dev_macro_cer"]
            < results["m3_alpha030"]["best_clean_dev_macro_cer"]
        ):
            write(output / "d2_decision.json", {
                "run_d2": False, "reason": "D1 did not improve matched M3 Clean Dev Macro CER",
                "d1_cer": results["dir_d1"]["best_clean_dev_macro_cer"],
                "m3_cer": results["m3_alpha030"]["best_clean_dev_macro_cer"]})
            print("D2_SKIPPED: D1 did not improve M3 alpha=.30", flush=True)
            break
        if method == "dir_d2":
            write(output / "d2_decision.json", {"run_d2": True, "reason": "D1 CER < matched M3 CER"})
        checkpoint = source
        for stage in ("synthetic", "target"):
            name = model_name(args, method, stage)
            config = root / "04_model_training/configs" / f"{name}.yml"
            run_dir = root / "04_model_training/runs" / name
            final_path = root / "04_model_training/eval_reports" / f"{name}_final_summary.json"
            if not final_path.exists():
                if args.summarize_only:
                    raise FileNotFoundError(final_path)
                prepare(args, method, stage, checkpoint, "controlled_same_source" if stage == "synthetic"
                        else "own_synthetic_best_clean_dev", 20260731, False)
                smoke(args, method, config, stage)
                # Evaluate every target epoch; retain a ten-epoch target patience window.
                train_stage(args, method, name, config, run_dir,
                            5 if stage == "synthetic" else 1,
                            4 if stage == "synthetic" else 10,
                            15 if stage == "synthetic" else 10, 29930 + index * 2 + (stage == "target"))
            summary = checked_summary(root, name)
            cfg = yaml.safe_load(config.read_text(encoding="utf-8-sig"))
            if (cfg["Global"]["method_variant"] != method
                    or summary["audited_alpha"] != 0.30
                    or sha256(local_path(root, cfg["Global"]["pretrained_model"])) != sha256(checkpoint)):
                raise ValueError(f"Completed stage has mismatched method/alpha/source: {name}")
            checkpoint = local_path(root, summary["best_checkpoint"])
        results[method] = summary
        write(output / "progress.json", {"completed": list(results), "test_evaluated": False})
        if method == "dir_d2":
            write(output / "d2_learned_gates.json", {"checkpoint_sha256": sha256(checkpoint),
                                                   **gates(checkpoint)})

    rows = []
    for name, summary in historical.items():
        row = metric_row(name, summary)
        row.update(candidate=name in ("M1", "M2"), alpha=summary["audited_alpha"])
        rows.append(row)
    for method, summary in results.items():
        name = {"m3_alpha030": "M3 / SOAR-SVTR (alpha=0.30)", "dir_d1": "D1", "dir_d2": "D2"}[method]
        row = metric_row(name, summary)
        row.update(candidate=True, alpha=0.30)
        rows.append(row)
    selection = select_structure(rows)
    write(output / "structure_selection.json", {**selection, "rows": rows,
          "old_full_label": "M3 + LDC (Diagnostic Ablation)",
          "decision_scope": "Clean Dev candidate selection; existing method freeze is not overwritten"})
    with (output / "structure_selection.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(selection, indent=2), flush=True)
    print("SEMANTIC_DIRECTION_STUDY_COMPLETE; Test was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
