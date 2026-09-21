#!/usr/bin/env python3
"""Shared-S50, no-new-loss D1/D2 study with complete saved-epoch reporting."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import re
import sys
from pathlib import Path

import yaml

from dual_order_protocol import METHODS, METHOD_PROTOCOL
from p1_msr_protocol import assert_p1_config_text
from prepare_method_multiseed import replace_global_value
from run_dual_order_method_suite import run, sha256, smoke, train_stage
from run_p1_rctc_stage import evaluate_dev
from run_semantic_direction_study import (
    HISTORICAL, checked_summary, local_path, metric_row, read, select_structure, write,
)

STUDY = "semantic_direction_v2"
SYNTHETIC_NAME = "M3_shared_S50_alpha030_v2"
NAMES = {"M3": "M3_anchor_alpha030_v2",
         "D1": "D1_sgm_only_direction_alpha030",
         "D2": "D2_script_gated_sgm_direction_alpha030"}
VARIANTS = {"M3": "m3_alpha030", "D1": "dir_d1_v2", "D2": "dir_d2_v2"}
LANGUAGES = ("zh", "ug", "kk")


def clone_config(text, root, name, method, source):
    """Preserve the template's named YAML anchors; verify a parsed allowlist diff."""
    original = yaml.safe_load(text)
    if original["Global"]["method_variant"] not in ("m3", "m3_alpha030"):
        raise ValueError("Direction experiments must clone an M3 template")
    if original["Architecture"]["Decoder"].get("use_local_direction"):
        raise ValueError("Old LDC cannot be used as the anchor")
    if original["Architecture"]["Decoder"].get("semantic_direction", "none") != "none":
        raise ValueError("Anchor already contains a semantic direction module")
    if not original["Architecture"]["Decoder"].get("use_script_adapter"):
        raise ValueError("M3 Script Adaptation is required")
    expected = copy.deepcopy(original)
    run_dir = root / "04_model_training/runs" / name
    changes = {"output_dir": str(run_dir), "project_name": name,
               "save_res_path": str(run_dir / "predicts.txt"),
               "pretrained_model": str(source), "method_variant": method,
               "model_protocol": f"{METHOD_PROTOCOL}_{method.upper()}"}
    for key, value in changes.items():
        text = replace_global_value(text, key, value)
        expected["Global"][key] = value
    text, count = re.subn(r"(?m)^  consistency_weight: [^\r\n]*", "  consistency_weight: 0.30", text)
    if count != 1:
        raise ValueError("Expected exactly one consistency weight")
    expected["Loss"]["consistency_weight"] = .30
    if expected["Loss"].get("direction_weight") != 0:
        raise ValueError("M3 loss must have direction_weight=0")
    if method in ("dir_d1_v2", "dir_d2_v2"):
        mode = METHODS[method]["semantic_direction"]
        gate_config = ""
        if method == "dir_d2_v2":
            gate_config = "    direction_gate_init_logit: -4.0\n"
        text, count = re.subn(
            r"(?m)^(    script_bottleneck_ratio: [^\r\n]*)(\r?\n)",
            rf"\1\2    semantic_direction: {mode}\n{gate_config}", text)
        if count != 1:
            raise ValueError("Expected one script adapter config")
        expected["Architecture"]["Decoder"]["semantic_direction"] = mode
        if method == "dir_d2_v2":
            expected["Architecture"]["Decoder"]["direction_gate_init_logit"] = -4.0
    actual = yaml.safe_load(text)
    if actual != expected:
        raise ValueError("Config clone changed fields outside the direction/output/alpha allowlist")
    assert_p1_config_text(text)
    if actual["Global"].get("uyghur_ctc_order") != "visual":
        raise ValueError("CTC must retain U2 visual order")
    for section in ("Train", "Eval"):
        transform = next(t["DualOrderGTCLabelEncode"] for t in actual[section]["dataset"]["transforms"]
                         if "DualOrderGTCLabelEncode" in t)
        if (transform["ctc_order"], transform["sgm_order"]) != ("visual", "logical"):
            raise ValueError("CTC/SGM target order drift")
    return text


def save_config(root, name, content):
    path = root / "04_model_training/configs" / f"{name}.yml"
    if path.exists() and path.read_text(encoding="utf-8-sig") != content:
        raise ValueError(f"Existing study config differs; refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    return path


def control_args(args, config):
    cfg = yaml.safe_load(config.read_text(encoding="utf-8-sig"))
    args.max_epoch = cfg["Global"]["epoch_num"]
    args.batch_size_per_card = cfg["Train"]["sampler"]["first_bs"]
    args.num_workers = cfg["Train"]["loader"]["num_workers"]
    args.lr = cfg["Optimizer"]["lr"]
    return cfg


def run_stage(args, method, name, config, control, stage, port):
    cfg = control_args(args, config)
    args.min_delta = control["min_delta"]
    final = args.root / "04_model_training/eval_reports" / f"{name}_final_summary.json"
    if not final.exists():
        if args.summarize_only:
            raise FileNotFoundError(final)
        smoke(args, method, config, stage)
        train_stage(args, method, name, config, args.root / "04_model_training/runs" / name,
                    control["eval_every_epochs"], control["patience_evals"],
                    control["min_epoch"], port)
    result = checked_summary(args.root, name)
    if result["audited_alpha"] != .30:
        raise ValueError(f"Locked alpha changed: {name}")
    for key in ("eval_every_epochs", "patience_evals", "min_epoch", "min_delta"):
        if result["training_control"][key] != control[key]:
            raise ValueError(f"Completed stage control differs: {name}.{key}")
    if result["training_control"]["max_epoch"] != cfg["Global"]["epoch_num"]:
        raise ValueError("Scheduler horizon drift")
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gate_values(checkpoint):
    import torch
    payload = torch.load(checkpoint, map_location="cpu")
    values = payload["state_dict"]["decoder.direction_gate_logits"].float().sigmoid().tolist()
    del payload
    gc.collect()
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid D2 gate values")
    return dict(zip(("gate_Han", "gate_Arabic", "gate_Cyrillic", "gate_Common_Latin"), values))


def epoch_row(model, epoch, summary, checkpoint_hash):
    synthetic_summary = {"dev": summary, "best_epoch": epoch,
        "checkpoint_selection": "clean_target_dev_macro_CER",
        "best_clean_dev_macro_cer": summary["macro_cer"],
        "best_checkpoint_sha256": checkpoint_hash}
    row = metric_row(model, synthetic_summary)
    for lang in LANGUAGES:
        row[f"{lang}_line_accuracy"] = summary["languages"][lang]["line_accuracy"]
    return row


def export_epochs(args, model, summary, output):
    root = args.root
    run_dir = local_path(root, summary["best_checkpoint"]).parent
    stopped = summary["training_control"]["stopped_epoch"]
    checkpoints = {int(p.stem.split("_")[-1]): p for p in run_dir.glob("epoch_*.pth")}
    if set(range(1, stopped + 1)) - set(checkpoints):
        raise ValueError(f"Missing saved epochs for complete selection: {run_dir}")
    args.config = local_path(root, summary["config"])
    args.labels_dir = root / "04_model_training/datasets/e1_target_only/labels"
    args.label_prefix = "target"
    args.prediction_branch = "ctc"
    config_hash = sha256(args.config)
    evaluator_hashes = {name: sha256(root / "scripts/svtrv2" / name) for name in (
        "evaluate_d2_checkpoint.py", "infer_label_file_metrics.py", "p1_msr_protocol.py")}
    rows = []
    for epoch in range(1, stopped + 1):
        checkpoint = checkpoints[epoch]
        digest = sha256(checkpoint)
        report_dir = output / "all_epoch_dev" / model / f"epoch_{epoch:04d}"
        meta = report_dir / "epoch_binding.json"
        report = report_dir / "metrics_macro_summary.json"
        expected = {"checkpoint_sha256": digest, "config_sha256": config_hash,
                    "epoch": epoch, "evaluators": evaluator_hashes}
        if meta.exists():
            binding = read(meta)
            if any(binding.get(k) != v for k, v in expected.items()) or binding["report_sha256"] != sha256(report):
                raise ValueError(f"Cached epoch provenance changed: {meta}")
            dev = read(report)
        else:
            dev = evaluate_dev(args, checkpoint, report_dir,
                               args.log_dir / f"{model}_all_epoch_{epoch:04d}.log")
            write(meta, {**expected, "report_sha256": sha256(report)})
        if dev["config_sha256"] != config_hash:
            raise ValueError("Epoch evaluation used a different configuration")
        row = epoch_row(model, epoch, dev, digest)
        row["checkpoint"] = str(checkpoint)
        row["report_dir"] = str(report_dir)
        if model == "D2":
            row.update(gate_values(checkpoint))
        rows.append(row)
        write_csv(output / f"{model}_epoch_metrics.csv", rows)
        write(output / f"{model}_epoch_metrics.json", rows)
        print(f"{model} epoch={epoch}: CER={100 * row['macro_cer']:.4f}% "
              f"LineAcc={100 * row['macro_line_accuracy']:.4f}%", flush=True)
    selected = min(rows, key=lambda r: (r["macro_cer"], r["epoch"]))
    write(output / f"{model}_cer_selected.json", {**selected,
          "selection_rule": "minimum_clean_dev_macro_CER_over_all_saved_epochs",
          "test_evaluated": False})
    return selected, rows


def compare_languages(selected):
    anchor = selected["M3"]
    rows = []
    for model, metrics in selected.items():
        for lang in (*LANGUAGES, "macro"):
            cer_key = "macro_cer" if lang == "macro" else f"{lang}_cer"
            acc_key = "macro_line_accuracy" if lang == "macro" else f"{lang}_line_accuracy"
            rows.append({"model": model, "language": lang, "epoch": metrics["epoch"],
                "cer": metrics[cer_key], "line_accuracy": metrics[acc_key],
                "cer_delta_vs_M3_pp": 100 * (metrics[cer_key] - anchor[cer_key]),
                "line_accuracy_delta_vs_M3_pp": 100 * (metrics[acc_key] - anchor[acc_key])})
    return rows


def plot_epochs(output, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric, label in (("macro_cer", "Macro CER (%)"),
                          ("macro_line_accuracy", "Macro Line Accuracy (%)")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for model, rows in histories.items():
            ax.plot([r["epoch"] for r in rows], [100 * r[metric] for r in rows], label=model)
        ax.set(xlabel="Epoch", ylabel=label)
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"epoch_{metric}.{ext}", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ("gate_Han", "gate_Arabic", "gate_Cyrillic", "gate_Common_Latin"):
        ax.plot([r["epoch"] for r in histories["D2"]], [r[key] for r in histories["D2"]], label=key[5:])
    ax.axhline(1 / (1 + math.exp(4)), color="black", linestyle="--", linewidth=.8, label="Initialization")
    ax.set(xlabel="Epoch", ylabel="Learned sigmoid gate")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"D2_gate_curves.{ext}", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--anchor-summary", type=Path,
                   help="Optional completed M3 alpha=.30 target final summary in eval_reports")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    args.root = args.root.resolve()
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.ddp_gpus, args.eval_gpu = "0,1", 0
    args.replace, args.resume_existing = False, True
    root = args.root
    output = root / "04_model_training/eval_reports" / STUDY
    import matplotlib  # Fail before training if curve export cannot run.

    historical = {name: checked_summary(root, stem) for name, stem in HISTORICAL.items()}
    if historical["M2"]["audited_alpha"] != .30:
        raise ValueError("The M2 reference must be the frozen alpha=.30 run")
    if historical["M3 (alpha=0.15, historical)"]["audited_alpha"] != .15:
        raise ValueError("The historical M3 reference is mislabeled; expected alpha=.15")
    parent = historical["M3 (alpha=0.15, historical)"]
    anchor = None
    anchor_path = args.anchor_summary or root / (
        "04_model_training/eval_reports/svtrv2_s_m3_alpha030_dual_order_s50_to_target_final_summary.json")
    if anchor_path.exists():
        anchor = checked_summary(root, anchor_path.stem.removesuffix("_final_summary"))
        cfg = yaml.safe_load(Path(anchor["config"]).read_text(encoding="utf-8-sig"))
        if anchor["audited_alpha"] != .30 or cfg["Global"]["method_variant"] != "m3_alpha030":
            raise ValueError("Supplied anchor is not M3 alpha=.30")
        parent = anchor
    elif args.anchor_summary:
        raise FileNotFoundError(args.anchor_summary)
    target_text = Path(parent["config"]).read_text(encoding="utf-8-sig")
    target_cfg = yaml.safe_load(target_text)
    if target_cfg["Global"]["seed"] != 20260731:
        raise ValueError("Canonical seed changed")
    source = local_path(root, target_cfg["Global"]["pretrained_model"]) if anchor else None
    binding = {"study": STUDY, "alpha": .30, "direction_weight": 0,
               "d2_gate_init_logit": -4.0,
               "d2_gate_type": "scalar_per_predicted_sample_script",
               "run_D2_even_if_D1_fails_to_improve": True,
               "target_template_sha256": sha256(Path(parent["config"])),
               "target_training_control": {k: parent["training_control"][k] for k in
                   ("max_epoch", "eval_every_epochs", "patience_evals", "min_epoch", "min_delta")},
               "anchor_summary": str(anchor_path) if anchor else None,
               "initialization_sha256": sha256(source) if source else None,
               "test_evaluated": False}
    bound_code = ("scripts/svtrv2/run_semantic_direction_v2.py",
                  "scripts/svtrv2/dual_order_protocol.py", "scripts/svtrv2/run_p1_rctc_stage.py",
                  "scripts/svtrv2/run_semantic_direction_study.py",
                  "scripts/svtrv2/run_dual_order_method_suite.py",
                  "scripts/svtrv2/smoke_dual_order_method.py",
                  "scripts/svtrv2/validate_dual_order_method_config.py",
                  "scripts/svtrv2/prepare_method_multiseed.py",
                  "scripts/svtrv2/evaluate_d2_checkpoint.py",
                  "scripts/svtrv2/infer_label_file_metrics.py",
                  "scripts/svtrv2/p1_msr_protocol.py",
                  "scripts/svtrv2/test_dual_order_components.py",
                  "scripts/svtrv2/test_semantic_direction.py",
                  "scripts/svtrv2/test_semantic_direction_v2_workflow.py",
                  "third_party/OpenOCR/openrec/preprocess/dual_order_gtc_label_encode.py",
                  "third_party/OpenOCR/openrec/modeling/decoders/dual_order_gtc_decoder.py",
                  "third_party/OpenOCR/openrec/losses/dual_order_gtc_loss.py")
    binding["code_sha256"] = {f: sha256(root / f) for f in bound_code}
    binding_file = args.log_dir / "study_binding.json"
    if binding_file.exists() and read(binding_file) != binding:
        raise ValueError("Study code/template changed during resume")
    write(binding_file, binding)
    if not args.summarize_only:
        run([sys.executable, str(root / "scripts/protocol/freeze_protocol_v2.py"),
             "--root", str(root), "--mode", "verify", "--workers", "8"], root)

    if source is None:
        synth_parent = checked_summary(root, "svtrv2_s_m3_dual_order_s50")
        text = Path(synth_parent["config"]).read_text(encoding="utf-8-sig")
        original_source = local_path(root, yaml.safe_load(text)["Global"]["pretrained_model"])
        conversion = read(original_source.with_suffix(".conversion.json"))
        d2_reference = checked_summary(root, "d2_synth50k")
        if (sha256(original_source) != conversion["output_sha256"]
                or conversion["input_sha256"] != d2_reference["best_checkpoint_sha256"]):
            raise ValueError("M3 initial converted checkpoint hash mismatch")
        synth_config = save_config(root, SYNTHETIC_NAME, clone_config(
            text, root, SYNTHETIC_NAME, "m3_alpha030", original_source))
        if args.preflight_only:
            smoke(args, "m3_alpha030", synth_config, "synthetic")
            print("DIRECTION_V2_ANCHOR_PREFLIGHT_OK; shared S50 alpha=.30 training required; "
                  "target preflights will run after the source exists.", flush=True)
            return
        synth = run_stage(args, "m3_alpha030", SYNTHETIC_NAME, synth_config,
                          synth_parent["training_control"], "synthetic", 29940)
        source = local_path(root, synth["best_checkpoint"])
    write(output / "shared_initialization.json", {"checkpoint": str(source),
          "checkpoint_sha256": sha256(source), "alpha": .30,
          "all_target_stages_share_this_source": True})

    configs = {}
    for model in NAMES:
        configs[model] = save_config(root, NAMES[model], clone_config(
            target_text, root, NAMES[model], VARIANTS[model], source))
        run([sys.executable, str(root / "scripts/svtrv2/validate_dual_order_method_config.py"),
             "--config", str(configs[model])], root)
        if not args.summarize_only:
            smoke(args, VARIANTS[model], configs[model], "target")
    print("DIRECTION_V2_ALL_TARGET_PREFLIGHTS_OK", flush=True)
    if args.preflight_only:
        return

    summaries, selected, histories = {}, {}, {}
    for index, model in enumerate(NAMES):
        summary = anchor if model == "M3" and anchor else run_stage(
            args, VARIANTS[model], NAMES[model], configs[model], parent["training_control"], "target", 29941 + index)
        cfg = yaml.safe_load(Path(summary["config"]).read_text(encoding="utf-8-sig"))
        if sha256(local_path(root, cfg["Global"]["pretrained_model"])) != sha256(source):
            raise ValueError(f"Shared S50 initialization differs: {model}")
        summaries[model] = summary
        selected[model], histories[model] = export_epochs(args, model, summary, output)
        write(output / "progress.json", {"completed": list(selected), "test_evaluated": False})

    comparison = compare_languages(selected)
    write_csv(output / "language_comparison.csv", comparison)
    write(output / "language_comparison.json", comparison)
    rows = []
    for name, summary in historical.items():
        row = metric_row(name, summary)
        row.update(alpha=summary["audited_alpha"], candidate=name in ("M1", "M2"))
        rows.append(row)
    for name, result in selected.items():
        row = {k: result[k] for k in rows[0] if k not in ("alpha", "candidate")}
        row.update(alpha=.30, candidate=True)
        rows.append(row)
    decision = select_structure(rows)
    b1 = next(r for r in rows if r["model"] == "B1")
    improves_anchor = any(
        selected[m]["macro_cer"] < selected["M3"]["macro_cer"]
        and selected[m]["macro_cer"] < b1["macro_cer"]
        and selected[m]["macro_line_accuracy"] > b1["macro_line_accuracy"] for m in ("D1", "D2"))
    decision.update(alpha=.30, rows=rows,
        direction_route="candidate_requires_three_seeds" if improves_anchor else "stop_direction_route",
        automatic_method_promotion=False,
        note="A single-seed improvement does not replace the frozen SOAR-SVTR model.")
    write(output / "structure_selection.json", decision)
    write_csv(output / "structure_selection.csv", rows)
    plot_epochs(output, histories)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    print("DIRECTION_V2_PHASE4_PHASE5_COMPLETE; Test was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
