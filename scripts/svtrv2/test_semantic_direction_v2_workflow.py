#!/usr/bin/env python3
"""Fixture tests for controlled config cloning and complete-epoch selection."""

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

import run_semantic_direction_v2 as study


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    template = (root / "04_model_training/configs/svtrv2_s_m3_dual_order_s50_to_target.yml").read_text(encoding="utf-8-sig")
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        (work / "scripts/svtrv2").mkdir(parents=True)
        for filename in ("evaluate_d2_checkpoint.py", "infer_label_file_metrics.py", "p1_msr_protocol.py"):
            shutil.copyfile(root / "scripts/svtrv2" / filename, work / "scripts/svtrv2" / filename)
        source = work / "shared_source.pth"
        parsed = {}
        for model, method in study.VARIANTS.items():
            parsed[model] = yaml.safe_load(study.clone_config(template, work, model, method, source))
        original = yaml.safe_load(template)
        for model, cfg in parsed.items():
            assert cfg["Loss"]["consistency_weight"] == .30
            assert cfg["Loss"]["direction_weight"] == 0
            for section in ("Optimizer", "LRScheduler", "Train", "Eval", "PostProcess"):
                assert cfg[section] == original[section]
            assert cfg["Global"]["pretrained_model"] == str(source)
        assert parsed["M3"]["Loss"] == parsed["D1"]["Loss"] == parsed["D2"]["Loss"]
        assert "direction_gate_init_logit" not in parsed["D1"]["Architecture"]["Decoder"]
        assert parsed["D2"]["Architecture"]["Decoder"]["direction_gate_init_logit"] == -4.0

        run_dir = work / "run"
        run_dir.mkdir()
        config = work / "config.yml"
        config.write_text("fixture: true\n", encoding="utf-8")
        for epoch in (1, 2, 3):
            torch.save({"epoch": epoch, "state_dict": {
                "decoder.direction_gate_logits": torch.full((4,), -4.0 + epoch * .1)
            }}, run_dir / f"epoch_{epoch}.pth")
        summary = {"best_checkpoint": str(run_dir / "best.pth"), "config": str(config),
                   "training_control": {"stopped_epoch": 3}}
        call_args = SimpleNamespace(root=work, log_dir=work / "logs", eval_gpu=0)
        output = work / "output"

        def fake_eval(_, checkpoint, report_dir, log):
            epoch = int(checkpoint.stem.split("_")[-1])
            cer, acc = {1: (.03, .92), 2: (.02, .89), 3: (.025, .90)}[epoch]
            dev = {"split": "dev", "prediction_branch": "ctc", "config_sha256": study.sha256(config),
                   "macro_cer": cer, "macro_line_accuracy": acc,
                   "languages": {lang: {"cer": cer, "wer": cer * 2,
                       "one_minus_ned_macro": 1 - cer, "line_accuracy": acc}
                       for lang in study.LANGUAGES}}
            study.write(report_dir / "metrics_macro_summary.json", dev)
            return dev

        with patch.object(study, "evaluate_dev", side_effect=fake_eval) as evaluator:
            selected, rows = study.export_epochs(call_args, "D2", summary, output)
            assert evaluator.call_count == 3
        assert selected["epoch"] == 2 and selected["macro_line_accuracy"] == .89
        assert len(rows) == 3 and all("gate_Arabic" in r for r in rows)
        with patch.object(study, "evaluate_dev", side_effect=AssertionError("Valid cache reran evaluation")):
            assert study.export_epochs(call_args, "D2", summary, output)[0] == selected
        histories = {model: copy.deepcopy(rows) for model in ("M3", "D1", "D2")}
        study.plot_epochs(output, histories)
        for path in output.glob("*.pdf"):
            assert path.stat().st_size > 1000
        baseline = copy.deepcopy(selected)
        candidate = copy.deepcopy(selected)
        candidate["ug_cer"] -= .001
        candidate["ug_line_accuracy"] += .01
        language_rows = study.compare_languages({"M3": baseline, "D1": candidate})
        ug = next(r for r in language_rows if r["model"] == "D1" and r["language"] == "ug")
        assert abs(ug["cer_delta_vs_M3_pp"] + .1) < 1e-8
        assert abs(ug["line_accuracy_delta_vs_M3_pp"] - 1.) < 1e-8
        bad = output / "all_epoch_dev/D2/epoch_0002/metrics_macro_summary.json"
        bad.write_text("{}", encoding="utf-8")
        try:
            study.export_epochs(call_args, "D2", summary, output)
        except ValueError as exc:
            assert "provenance" in str(exc)
        else:
            raise AssertionError("Tampered epoch report accepted")
    print(json.dumps({"status": "DIRECTION_V2_WORKFLOW_TESTS_OK",
        "config_only_allowed_fields_changed": True,
        "shared_S50_initialization": True, "no_new_loss": True,
        "all_epoch_CER_selection": True, "four_gates_each_epoch": True,
        "cache_provenance_validated": True, "language_delta_units_checked": True,
        "plot_exports": True, "test_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
