#!/usr/bin/env python3
"""Export auditable, pre-collapse CTC paths for frozen Clean-Dev predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


LANGUAGES = ("ug", "kk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--m3-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-preprocess-protocol", default="P1_MSR_V3")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if row.get("status") != "ok":
                    raise ValueError(f"Invalid frozen prediction at {path}:{line_number}")
                rows.append(row)
    return rows


def relocate(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: relocate(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item, root) for item in value]
    if not isinstance(value, str):
        return value
    marker = "svtrv2_line_recognition"
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if marker in parts:
        suffix = parts[parts.index(marker) + 1 :]
        return str(root.joinpath(*suffix))
    return value


def frozen_rows(report: Path) -> dict[str, list[dict[str, Any]]]:
    files = {
        "ug": report / "ug_visual/predictions.jsonl",
        "kk": report / "kk/predictions.jsonl",
    }
    return {language: read_jsonl(path) for language, path in files.items()}


def compress_path(ids: np.ndarray, probabilities: np.ndarray, characters: list[str]) -> dict[str, Any]:
    runs = []
    start = 0
    for index in range(1, len(ids) + 1):
        if index < len(ids) and ids[index] == ids[start]:
            continue
        token_id = int(ids[start])
        run_probs = probabilities[start:index]
        runs.append(
            {
                "run_index": len(runs),
                "token_id": token_id,
                "token": characters[token_id] if token_id else "<blank>",
                "is_blank": token_id == 0,
                "start_t": start,
                "end_t_exclusive": index,
                "max_probability": float(run_probs.max()),
                "mean_probability": float(run_probs.mean()),
            }
        )
        start = index

    collapsed = []
    for run in runs:
        if run["is_blank"]:
            continue
        collapsed.append(
            {
                "output_index": len(collapsed),
                "token_id": run["token_id"],
                "token": run["token"],
                "raw_run_index": run["run_index"],
                "max_probability": run["max_probability"],
                "mean_probability": run["mean_probability"],
            }
        )
    return {
        "time_steps": int(len(ids)),
        "runs": runs,
        "collapsed_segments": collapsed,
        "decoded_visual_text": "".join(item["token"] for item in collapsed),
    }


class RawPathCapture:
    def __init__(self, characters: list[str]):
        self.characters = characters

    def __call__(self, preds, **kwargs):
        if hasattr(preds, "detach"):
            preds = preds.detach().float().cpu().numpy()
        ids = np.asarray(preds).argmax(axis=2)
        probabilities = np.asarray(preds).max(axis=2)
        return [
            (compress_path(sample_ids, sample_probs, self.characters), 1.0)
            for sample_ids, sample_probs in zip(ids, probabilities)
        ]


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    report = args.m3_report.resolve()
    output = args.output.resolve()
    for path in (config, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and not args.replace:
        raise FileExistsError(f"Raw-path output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    openocr_root = root / "third_party/OpenOCR"
    sys.path.insert(0, str(openocr_root))
    sys.path.insert(0, str(openocr_root / "tools"))
    from openrec.postprocess import build_post_process
    from tools.engine.config import Config
    from tools.infer_rec import OpenRecognizer

    cfg = relocate(Config(str(config)).cfg, root)
    if cfg["Global"].get("preprocess_protocol") != args.expected_preprocess_protocol:
        raise ValueError("Frozen M3 preprocessing protocol changed")
    cfg["Global"]["checkpoints"] = str(checkpoint)
    cfg["Global"]["pretrained_model"] = None
    cfg["Global"]["use_amp"] = False
    cfg["Global"]["infer_img"] = None
    recognizer = OpenRecognizer(
        config=cfg,
        mode="server",
        backend="torch",
        use_gpu="true",
        numId=args.device_id,
    )
    decoder = getattr(recognizer.model, "decoder", None)
    if decoder is None or not hasattr(decoder, "ctc_decoder"):
        raise ValueError("Frozen M3 does not expose its CTC branch")
    decoder.infer_gtc = False
    ctc_post = build_post_process(
        {
            "name": "CTCLabelDecode",
            "character_dict_path": cfg["Global"]["character_dict_path"],
            "use_space_char": cfg["Global"].get("use_space_char", False),
        },
        cfg["Global"],
    )
    recognizer.post_process_class = RawPathCapture(ctc_post.character)

    target_root = root / "01_data_preparation/real_line_dataset_eval_reviewed"
    rows_by_language = frozen_rows(report)
    output_hashes = {}
    for language in LANGUAGES:
        destination = output / f"{language}_ctc_raw_paths.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for row_index, row in enumerate(rows_by_language[language], 1):
                image = str(row["image"]).replace("\\", "/")
                payload = recognizer(
                    img_path=str(target_root / image), batch_num=1
                )[0]["text"]
                expected = row.get("eval_pred_text", row.get("pred_text", ""))
                if payload["decoded_visual_text"] != expected:
                    raise ValueError(
                        f"Frozen/raw prediction mismatch for {language}:{image}: "
                        f"{payload['decoded_visual_text']!r} != {expected!r}"
                    )
                record = {
                    "language": language,
                    "image": image,
                    "frozen_visual_prediction": expected,
                    **payload,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if row_index % 100 == 0 or row_index == len(rows_by_language[language]):
                    print(
                        json.dumps(
                            {
                                "language": language,
                                "completed": row_index,
                                "rows": len(rows_by_language[language]),
                            }
                        ),
                        flush=True,
                    )
        output_hashes[language] = sha256(destination)

    summary = {
        "status": "M3_CTC_RAW_PATH_EXPORT_OK",
        "split": "clean_dev",
        "languages": list(LANGUAGES),
        "config": str(config),
        "config_sha256": sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "frozen_report": str(report),
        "output_sha256": output_hashes,
        "ctc_path_stage": "argmax_before_standard_CTC_collapse",
        "test_evaluated": False,
    }
    (output / "raw_path_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
