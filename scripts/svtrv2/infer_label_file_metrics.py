#!/usr/bin/env python3
"""Run OpenOCR recognition on a label file and report per-language metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from metrics_v1 import (
    edit_distance,
    finalize_metric_bucket,
    update_metric_bucket,
)


LANGS = {"zh", "ug", "kk"}
NORMALIZATION_CHOICES = ("none", "normalization_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--label-file", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--expected-preprocess-protocol",
        default=None,
        help="Abort when the config does not declare this preprocessing protocol.",
    )
    parser.add_argument(
        "--gt-transform",
        choices=["none", "ug_logical_to_visual"],
        default="none",
        help="Transform ground-truth text before metric computation.",
    )
    parser.add_argument(
        "--pred-transform",
        choices=["none", "ug_logical_to_visual"],
        default="none",
        help="Transform prediction text before metric computation.",
    )
    parser.add_argument(
        "--prediction-branch",
        choices=["auto", "ctc"],
        default="auto",
        help=(
            "Use ctc for a full GTCDecoder checkpoint when SGM is training-only. "
            "The full model is loaded first, then inference is switched to RCTC."
        ),
    )
    parser.add_argument(
        "--metric-normalization",
        choices=NORMALIZATION_CHOICES,
        default="none",
        help="Apply the same frozen normalization to GT and predictions before scoring.",
    )
    return parser.parse_args()


def load_openocr(openocr_root: Path):
    openocr_root = openocr_root.resolve()
    sys.path.insert(0, str(openocr_root))
    sys.path.insert(0, str(openocr_root / "tools"))
    from tools.engine.config import Config
    from tools.infer_rec import OpenRecognizer

    return Config, OpenRecognizer


def read_label_file(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(f"Bad label line {line_no}: no tab separator")
            image, text = line.split("\t", 1)
            rows.append({"image": image, "gt_text": text})
    return rows


def get_bidi_display():
    try:
        from bidi.algorithm import get_display  # type: ignore

        return get_display
    except Exception:
        return None


GET_DISPLAY = get_bidi_display()


def normalize_spaces(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def ug_logical_to_visual(text: str) -> str:
    text = normalize_spaces(text)
    if GET_DISPLAY is None:
        raise RuntimeError(
            "python-bidi is required for U2 conversion. "
            "A simple character reverse is not protocol-compatible."
        )
    return GET_DISPLAY(text, base_dir="R")


def transform_text(text: str, transform: str) -> str:
    if transform == "none":
        return text
    if transform == "ug_logical_to_visual":
        return ug_logical_to_visual(text)
    raise ValueError(f"Unsupported text transform: {transform}")


def load_metric_normalizer(name: str):
    if name == "none":
        return lambda text: text, None, None
    protocol_dir = Path(__file__).resolve().parents[1] / "protocol"
    sys.path.insert(0, str(protocol_dir))
    from text_normalization_v2 import PROTOCOL_ID, normalize_text_v2

    implementation = protocol_dir / "text_normalization_v2.py"
    return (
        normalize_text_v2,
        PROTOCOL_ID,
        hashlib.sha256(implementation.read_bytes()).hexdigest(),
    )


def load_language_map(metadata_csv: Path | None) -> dict[str, str]:
    if not metadata_csv:
        return {}
    mapping = {}
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image = (row.get("image") or "").replace("\\", "/")
            language = row.get("language") or row.get("lang") or ""
            if image and language:
                mapping[image] = language
                mapping[Path(image).name] = language
    return mapping


def infer_language(image: str, language_map: dict[str, str]) -> str:
    key = image.replace("\\", "/")
    if key in language_map:
        return language_map[key]
    name = Path(key).name
    if name in language_map:
        return language_map[name]
    lower = key.lower()
    for lang in LANGS:
        if f"/{lang}/" in lower or f"_{lang}_" in lower or name.lower().startswith(f"{lang}_"):
            return lang
    return "unknown"


def main() -> None:
    args = parse_args()
    normalize_metric_text, normalization_protocol, normalization_sha256 = (
        load_metric_normalizer(args.metric_normalization)
    )
    Config, OpenRecognizer = load_openocr(args.openocr_root)

    cfg = Config(str(args.config)).cfg
    preprocess_protocol = cfg["Global"].get("preprocess_protocol")
    if (
        args.expected_preprocess_protocol
        and preprocess_protocol != args.expected_preprocess_protocol
    ):
        raise ValueError(
            "Preprocessing protocol mismatch: "
            f"expected {args.expected_preprocess_protocol!r}, "
            f"got {preprocess_protocol!r}"
        )
    cfg["Global"]["checkpoints"] = str(args.checkpoint)
    cfg["Global"]["pretrained_model"] = None
    cfg["Global"]["use_amp"] = False
    cfg["Global"]["infer_img"] = None

    rows = read_label_file(args.label_file)
    language_map = load_language_map(args.metadata_csv)

    model = OpenRecognizer(config=cfg, mode="server", backend="torch", use_gpu="true", numId=args.device_id)
    if args.prediction_branch == "ctc":
        decoder = getattr(model.model, "decoder", None)
        if decoder is None or not hasattr(decoder, "ctc_decoder"):
            raise ValueError(
                "--prediction-branch ctc requires a full GTCDecoder model"
            )
        decoder.infer_gtc = False
        from openrec.postprocess import build_post_process

        model.post_process_class = build_post_process(
            {
                "name": "CTCLabelDecode",
                "character_dict_path": cfg["Global"]["character_dict_path"],
                "use_space_char": cfg["Global"].get("use_space_char", False),
            },
            cfg["Global"],
        )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "metrics_summary.json"

    metrics = defaultdict(lambda: defaultdict(float))
    status_counts = Counter()
    wall_start = time.perf_counter()
    model_inference_seconds = 0.0

    with pred_path.open("w", encoding="utf-8") as out:
        for row_index, row in enumerate(rows, 1):
            image_path = args.data_dir / row["image"]
            try:
                pred = model(img_path=str(image_path), batch_num=1)[0]
            except Exception as exc:
                language = infer_language(row["image"], language_map)
                result = {
                    "status": "error",
                    "error": repr(exc),
                    "image": row["image"],
                    "language": language,
                    "gt_text": row["gt_text"],
                    "pred_text": "",
                }
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                status_counts["error"] += 1
                continue

            language = infer_language(row["image"], language_map)
            pred_text = pred.get("text", "")
            inference_seconds = float(pred.get("elapse") or 0.0)
            model_inference_seconds += inference_seconds
            gt_text = row["gt_text"]
            normalized_gt_text = normalize_metric_text(gt_text)
            normalized_pred_text = normalize_metric_text(pred_text)
            eval_gt_text = transform_text(normalized_gt_text, args.gt_transform)
            eval_pred_text = transform_text(normalized_pred_text, args.pred_transform)
            result = {
                "status": "ok",
                "image": row["image"],
                "language": language,
                "gt_text": gt_text,
                "pred_text": pred_text,
                "normalized_gt_text": normalized_gt_text,
                "normalized_pred_text": normalized_pred_text,
                "eval_gt_text": eval_gt_text,
                "eval_pred_text": eval_pred_text,
                "score": pred.get("score"),
                "inference_seconds": inference_seconds,
                "char_edit_distance": edit_distance(eval_gt_text, eval_pred_text),
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            status_counts["ok"] += 1
            update_metric_bucket(metrics[language], eval_gt_text, eval_pred_text)
            update_metric_bucket(metrics["all"], eval_gt_text, eval_pred_text)
            if row_index % 100 == 0 or row_index == len(rows):
                print(
                    json.dumps(
                        {
                            "evaluation_progress": row_index,
                            "rows": len(rows),
                            "language": language,
                            "errors": status_counts["error"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    wall_seconds = time.perf_counter() - wall_start
    ok_samples = status_counts["ok"]
    summary = {
        "label_file": str(args.label_file),
        "data_dir": str(args.data_dir),
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "checkpoint": str(args.checkpoint),
        "label_file_sha256": hashlib.sha256(args.label_file.read_bytes()).hexdigest(),
        "preprocess_protocol": preprocess_protocol,
        "metric_normalization": normalization_protocol,
        "normalization_implementation_sha256": normalization_sha256,
        "prediction_branch": args.prediction_branch,
        "gt_transform": args.gt_transform,
        "pred_transform": args.pred_transform,
        "rows": len(rows),
        "status_counts": dict(status_counts),
        "timing": {
            "wall_seconds": wall_seconds,
            "model_inference_seconds": model_inference_seconds,
            "wall_samples_per_second": (
                ok_samples / wall_seconds if wall_seconds else None
            ),
            "model_samples_per_second": (
                ok_samples / model_inference_seconds
                if model_inference_seconds
                else None
            ),
            "note": (
                "Evaluation currently invokes OpenRecognizer per sample; "
                "model_samples_per_second excludes image IO and Python loop overhead."
            ),
        },
        "language_counts": dict(Counter(infer_language(row["image"], language_map) for row in rows)),
        "metrics": {
            lang: finalize_metric_bucket(bucket)
            for lang, bucket in sorted(metrics.items())
        },
        "outputs": {
            "predictions": str(pred_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
