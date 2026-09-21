#!/usr/bin/env python3
"""Evaluate one SVTRv2 checkpoint on target-domain dev/test by language.

The config declares whether Uyghur CTC predicts U2 visual order or Unicode
logical order. Visual-order predictions retain both direct U2 metrics and the
frozen heuristic logical-order report; logical-order diagnostics are evaluated
directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


LANGS = ("zh", "ug", "kk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Directory containing real_<split>_<lang>_logical.txt. Defaults to the D2 label directory.",
    )
    parser.add_argument(
        "--label-prefix",
        default="real",
        help="Label filename prefix, for example 'real' (D2) or 'target' (E1).",
    )
    parser.add_argument(
        "--label-template",
        default="{prefix}_{split}_{lang}_logical.txt",
        help=(
            "Filename template under --labels-dir. Clean Dev V2 uses "
            "'{lang}.txt'. Available fields: prefix, split, lang."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Target-domain image root. Defaults to 01_data_preparation/real_line_dataset_eval_reviewed.",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--require-python-bidi", action="store_true")
    parser.add_argument(
        "--expected-preprocess-protocol",
        default=None,
        help="Forwarded to every per-language inference process.",
    )
    parser.add_argument(
        "--prediction-branch",
        choices=("auto", "ctc"),
        default="auto",
        help="Use ctc for full SVTRv2 checkpoints with training-only SGM.",
    )
    parser.add_argument(
        "--metric-normalization",
        choices=("none", "normalization_v2"),
        default="none",
    )
    parser.add_argument(
        "--frozen-clean-dev-manifest",
        type=Path,
        default=None,
        help=(
            "Required with normalization_v2. Verifies the frozen protocol, "
            "label hashes, and normalization implementation before inference."
        ),
    )
    return parser.parse_args()


def run(cmd: list[str], log_path: Path, device_id: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path.cwd()),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with code {return_code}: {' '.join(cmd)}\nSee log: {log_path}")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def metric_for(summary: dict, lang: str) -> dict:
    metrics = summary.get("metrics", {})
    return metrics.get(lang) or metrics.get("all") or {}


def uyghur_ctc_order(config_path: Path) -> str:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    order = config.get("Global", {}).get("uyghur_ctc_order", "visual")
    if order not in {"visual", "logical"}:
        raise ValueError(f"Unsupported uyghur_ctc_order: {order}")
    return order


def label_path(args: argparse.Namespace, labels_dir: Path, lang: str) -> Path:
    try:
        filename = args.label_template.format(
            prefix=args.label_prefix,
            split=args.split,
            lang=lang,
        )
    except KeyError as exc:
        raise ValueError(f"Unsupported --label-template field: {exc}") from exc
    path = labels_dir / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def add_common_inference_args(cmd: list[str], args: argparse.Namespace) -> None:
    if args.expected_preprocess_protocol:
        cmd.extend(
            ["--expected-preprocess-protocol", args.expected_preprocess_protocol]
        )
    if args.prediction_branch != "auto":
        cmd.extend(["--prediction-branch", args.prediction_branch])
    if args.metric_normalization != "none":
        cmd.extend(["--metric-normalization", args.metric_normalization])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_clean_dev_v2(
    args: argparse.Namespace,
    labels_dir: Path,
) -> dict | None:
    if args.metric_normalization == "none":
        if args.frozen_clean_dev_manifest is not None:
            raise ValueError(
                "--frozen-clean-dev-manifest requires --metric-normalization "
                "normalization_v2"
            )
        return None
    if args.split != "dev":
        raise ValueError("normalization_v2 is frozen for Clean Dev V2 only")
    if args.frozen_clean_dev_manifest is None:
        raise ValueError(
            "normalization_v2 requires --frozen-clean-dev-manifest"
        )
    manifest_path = args.frozen_clean_dev_manifest.resolve()
    frozen = read_json(manifest_path)
    if frozen.get("status") != "CLEAN_DEV_V2_VERIFIED_FROZEN":
        raise ValueError(f"Not a frozen Clean Dev V2 manifest: {manifest_path}")
    if frozen.get("test_evaluated") is not False:
        raise ValueError("Frozen Clean Dev V2 manifest violates the no-Test policy")
    if args.label_template != "{lang}.txt":
        raise ValueError("Clean Dev V2 requires --label-template '{lang}.txt'")
    for lang in LANGS:
        path = label_path(args, labels_dir, lang)
        expected = frozen["artifact_sha256"][f"labels/{lang}.txt"]
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"Frozen Clean Dev V2 label hash mismatch for {lang}: "
                f"expected={expected}, actual={actual}"
            )
    implementation = (
        args.root.resolve() / "scripts" / "protocol" / "text_normalization_v2.py"
    )
    actual_implementation_hash = sha256(implementation)
    expected_implementation_hash = frozen["source"][
        "normalization_implementation_sha256"
    ]
    if actual_implementation_hash != expected_implementation_hash:
        raise ValueError(
            "Frozen normalization implementation hash mismatch: "
            f"expected={expected_implementation_hash}, "
            f"actual={actual_implementation_hash}"
        )
    return {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "protocol_id": frozen["protocol_id"],
        "rows": frozen["rows"],
        "patch_rows": frozen["patch_rows"],
    }


def main() -> None:
    args = parse_args()
    root = args.root
    openocr_root = root / "third_party" / "OpenOCR"
    target_root = args.data_dir or (
        root / "01_data_preparation" / "real_line_dataset_eval_reviewed"
    )
    labels_dir = args.labels_dir or (
        root / "04_model_training" / "datasets" / "d2_synth50k" / "labels"
    )
    frozen_clean_dev = verify_frozen_clean_dev_v2(args, labels_dir)
    ug_ctc_order = uyghur_ctc_order(args.config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_lang = {}

    for lang in LANGS:
        if lang == "ug":
            if ug_ctc_order == "logical":
                logical_dir = args.output_dir / "ug_logical"
                label_file = label_path(args, labels_dir, "ug")
                cmd = [
                    sys.executable,
                    str(
                        root
                        / "scripts"
                        / "svtrv2"
                        / "infer_label_file_metrics.py"
                    ),
                    "--openocr-root",
                    str(openocr_root),
                    "--config",
                    str(args.config),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--data-dir",
                    str(target_root),
                    "--label-file",
                    str(label_file),
                    "--metadata-csv",
                    str(target_root / "metadata.csv"),
                    "--output-dir",
                    str(logical_dir),
                    "--batch-size",
                    str(args.batch_size),
                    "--device-id",
                    "0",
                ]
                add_common_inference_args(cmd, args)
                run(
                    cmd,
                    args.output_dir / "logs" / "ug_logical_eval.log",
                    args.device_id,
                )
                summary = read_json(logical_dir / "metrics_summary.json")
                per_lang[lang] = metric_for(summary, "ug")
                per_lang[lang]["report_dir"] = str(logical_dir)
                continue

            visual_dir = args.output_dir / "ug_visual"
            logical_dir = args.output_dir / "ug_logical"
            label_file = label_path(args, labels_dir, "ug")
            cmd = [
                sys.executable,
                str(root / "scripts" / "svtrv2" / "infer_label_file_metrics.py"),
                "--openocr-root",
                str(openocr_root),
                "--config",
                str(args.config),
                "--checkpoint",
                str(args.checkpoint),
                "--data-dir",
                str(target_root),
                "--label-file",
                str(label_file),
                "--metadata-csv",
                str(target_root / "metadata.csv"),
                "--output-dir",
                str(visual_dir),
                "--batch-size",
                str(args.batch_size),
                "--device-id",
                "0",
                "--gt-transform",
                "ug_logical_to_visual",
            ]
            add_common_inference_args(cmd, args)
            run(cmd, args.output_dir / "logs" / "ug_visual_eval.log", args.device_id)
            visual_summary = read_json(
                visual_dir / "metrics_summary.json"
            )
            visual_metrics = metric_for(visual_summary, "ug")

            cmd = [
                sys.executable,
                str(root / "scripts" / "svtrv2" / "u2_visual_predictions_to_logical_metrics.py"),
                "--input",
                str(visual_dir / "predictions.jsonl"),
                "--output-dir",
                str(logical_dir),
                "--require-python-bidi",
            ]
            if args.metric_normalization != "none":
                cmd.extend(["--metric-normalization", args.metric_normalization])
            run(cmd, args.output_dir / "logs" / "ug_logical_metrics.log", args.device_id)
            summary = read_json(logical_dir / "metrics_summary_logical.json")
            per_lang[lang] = metric_for(summary, "ug")
            per_lang[lang]["report_dir"] = str(logical_dir)
            per_lang[lang]["visual_report_dir"] = str(visual_dir)
            per_lang[lang]["u2_visual_order_metrics"] = visual_metrics
        else:
            out_dir = args.output_dir / lang
            label_file = label_path(args, labels_dir, lang)
            cmd = [
                sys.executable,
                str(root / "scripts" / "svtrv2" / "infer_label_file_metrics.py"),
                "--openocr-root",
                str(openocr_root),
                "--config",
                str(args.config),
                "--checkpoint",
                str(args.checkpoint),
                "--data-dir",
                str(target_root),
                "--label-file",
                str(label_file),
                "--metadata-csv",
                str(target_root / "metadata.csv"),
                "--output-dir",
                str(out_dir),
                "--batch-size",
                str(args.batch_size),
                "--device-id",
                "0",
            ]
            add_common_inference_args(cmd, args)
            run(cmd, args.output_dir / "logs" / f"{lang}_eval.log", args.device_id)
            summary = read_json(out_dir / "metrics_summary.json")
            per_lang[lang] = metric_for(summary, lang)
            per_lang[lang]["report_dir"] = str(out_dir)

    cer_values = [per_lang[lang].get("cer") for lang in LANGS]
    valid_cer_values = [value for value in cer_values if value is not None]
    macro_cer = sum(valid_cer_values) / len(valid_cer_values) if valid_cer_values else None
    macro_line_acc_values = [per_lang[lang].get("line_accuracy") for lang in LANGS]
    valid_line_acc = [value for value in macro_line_acc_values if value is not None]
    macro_line_acc = sum(valid_line_acc) / len(valid_line_acc) if valid_line_acc else None
    macro_wer_values = [per_lang[lang].get("wer") for lang in LANGS]
    valid_wer = [value for value in macro_wer_values if value is not None]
    macro_wer = sum(valid_wer) / len(valid_wer) if valid_wer else None
    macro_ned_values = [per_lang[lang].get("one_minus_ned_macro") for lang in LANGS]
    valid_ned = [value for value in macro_ned_values if value is not None]
    macro_one_minus_ned = sum(valid_ned) / len(valid_ned) if valid_ned else None

    result = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "preprocess_protocol": args.expected_preprocess_protocol,
        "metric_normalization": args.metric_normalization,
        "frozen_clean_dev": frozen_clean_dev,
        "label_template": args.label_template,
        "label_files": {
            lang: {
                "path": str(label_path(args, labels_dir, lang)),
                "sha256": hashlib.sha256(label_path(args, labels_dir, lang).read_bytes()).hexdigest(),
            }
            for lang in LANGS
        },
        "prediction_branch": args.prediction_branch,
        "uyghur_ctc_order": ug_ctc_order,
        "dataset_role": "target_domain",
        "uyghur_reporting": {
            "primary_in_macro": (
                "direct logical-order CTC metrics"
                if ug_ctc_order == "logical"
                else (
                    "heuristic logical-order metrics from a second "
                    "python-bidi display pass"
                )
            ),
            "direct_ctc_metric": (
                "languages.ug"
                if ug_ctc_order == "logical"
                else "languages.ug.u2_visual_order_metrics"
            ),
            "is_general_visual_to_logical_inverse": (
                None if ug_ctc_order == "logical" else False
            ),
            "research_note": (
                "B2 directly predicts Unicode logical order."
                if ug_ctc_order == "logical"
                else (
                    "Report both values. The heuristic is exact for every "
                    "frozen dev ground-truth label, but not for every possible "
                    "mixed RTL/LTR prediction or test label."
                )
            ),
        },
        "macro_cer": macro_cer,
        "macro_wer": macro_wer,
        "macro_one_minus_ned": macro_one_minus_ned,
        "macro_line_accuracy": macro_line_acc,
        "languages": per_lang,
        "output_dir": str(args.output_dir),
    }
    if args.split == "dev":
        result["selection_metric"] = "clean_target_dev_macro_CER"
    else:
        result["evaluation_metric"] = "target_test_macro_CER"
        result["checkpoint_selection_split"] = "dev"
    out_path = args.output_dir / "metrics_macro_summary.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
