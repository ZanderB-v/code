#!/usr/bin/env python3
"""Mine and review clean hard examples from the frozen target-domain train split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from metrics_v1 import edit_distance


PROTOCOL_ID = "target_train_clean_hard_audit_v1"
EXPECTED_PREPROCESS = "P1_MSR_V3"
EXPECTED_METHOD = "m3"
EXPECTED_ALPHA = 0.15
EXPECTED_TRAIN_ROWS = 66854
EXPECTED_LANGUAGE_ROWS = {"zh": 23954, "ug": 21899, "kk": 21001}
LANGUAGES = ("zh", "ug", "kk")
ALLOWED_DECISIONS = {
    "genuine_ocr_error",
    "gt_annotation_error",
    "normalization_issue",
    "ambiguous_image",
}
REVIEW_FIELDS = [
    "sample_id",
    "language",
    "ed",
    "ed_bucket",
    "image_path",
    "gt_text",
    "prediction",
    "confidence",
    "width",
    "height",
    "contrast_span",
    "blur_variance",
    "edge_contact",
    "risk_flags",
    "manual_decision",
    "manual_notes",
]
MEMBERSHIP_FIELDS = (
    "sample_id",
    "language",
    "ed",
    "ed_bucket",
    "image_path",
    "gt_text",
    "prediction",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = {
            field: "" if row.get(field) is None else str(row.get(field, ""))
            for field in fields
        }
        digest.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def read_resumable_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read an append-only partial file and discard only a torn final line."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    rows = []
    valid_lines = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
            valid_lines.append(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError(f"Corrupt non-final JSONL line in {path}")
    if len(valid_lines) != len([line for line in lines if line.strip()]):
        path.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
    return rows


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def normalize_for_risk(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    return " ".join(normalized.replace("\u00a0", " ").split())


def char_script(char: str) -> str:
    if not char or char.isspace():
        return "Common"
    codepoint = ord(char)
    if (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    ):
        return "Han"
    if (
        0x0600 <= codepoint <= 0x06FF
        or 0x0750 <= codepoint <= 0x077F
        or 0x08A0 <= codepoint <= 0x08FF
        or 0xFB50 <= codepoint <= 0xFDFF
        or 0xFE70 <= codepoint <= 0xFEFF
    ):
        return "Arabic"
    if 0x0400 <= codepoint <= 0x052F or 0x2DE0 <= codepoint <= 0x2DFF:
        return "Cyrillic"
    if "LATIN" in unicodedata.name(char, ""):
        return "Latin"
    return "Common"


def text_risk_flags(text: str, dictionary: set[str]) -> list[str]:
    flags = []
    if any(char not in dictionary and char != " " for char in text):
        flags.append("dictionary_oov")
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in text):
        flags.append("unicode_control_or_surrogate")
    if unicodedata.normalize("NFC", text) != text:
        flags.append("non_nfc_label")
    scripts = {char_script(char) for char in text} - {"Common"}
    if len(scripts) > 1:
        flags.append("suspicious_mixed_scripts")
    return flags


def image_quality(path: Path) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
        from PIL import Image

        with Image.open(path) as source:
            image = source.convert("RGB")
            width, height = image.size
            rgb = np.asarray(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        p05, p95 = np.percentile(gray, [5, 95])
        contrast = float(p95 - p05)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        border = np.concatenate(
            [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]], axis=0
        )
        background = float(np.median(border))
        foreground = np.abs(gray.astype(np.float32) - background) > max(12.0, contrast * 0.18)
        strip = max(1, min(3, width // 8, height // 8))
        contacts = [
            foreground[:, :strip].mean(),
            foreground[:, -strip:].mean(),
            foreground[:strip, :].mean(),
            foreground[-strip:, :].mean(),
        ]
        edge_contact = float(max(contacts))
        flags = []
        aspect = width / max(height, 1)
        if width < 16 or height < 12 or aspect < 0.4 or aspect > 40.0:
            flags.append("abnormal_image_dimensions")
        if contrast < 20.0:
            flags.append("low_contrast")
        if blur < 12.0:
            flags.append("severe_blur_or_unreadable")
        if edge_contact > 0.75:
            flags.append("possible_bad_crop")
        return {
            "status": "ok",
            "width": width,
            "height": height,
            "aspect_ratio": aspect,
            "contrast_span": contrast,
            "blur_variance": blur,
            "edge_contact": edge_contact,
            "image_risk_flags": flags,
            "image": image,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": repr(exc),
            "width": 0,
            "height": 0,
            "aspect_ratio": 0.0,
            "contrast_span": 0.0,
            "blur_variance": 0.0,
            "edge_contact": 1.0,
            "image_risk_flags": ["image_decode_error"],
            "image": None,
        }


def load_dictionary(path: Path) -> set[str]:
    characters = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            char = line.rstrip("\r\n")
            if char:
                characters.add(char)
    if len(characters) < 1000:
        raise ValueError(f"Character dictionary is unexpectedly small: {len(characters)}")
    return characters


def logical_prediction(prediction: str, language: str) -> str:
    if language != "ug":
        return prediction
    try:
        from bidi.algorithm import get_display
    except ImportError as exc:
        raise RuntimeError("python-bidi is required for frozen U2 recovery") from exc
    normalized = " ".join((prediction or "").replace("\u00a0", " ").split())
    return get_display(normalized, base_dir="R")


def validate_frozen_inputs(
    root: Path, config: Path, checkpoint: Path, metadata: Path, dictionary: Path
) -> dict[str, Any]:
    for path in (config, checkpoint, metadata, dictionary):
        if not path.is_file():
            raise FileNotFoundError(path)
    verification_path = root / "00_docs/frozen_protocol_v2/verification_report.json"
    manifest_path = root / "00_docs/frozen_protocol_v2/protocol_v2_manifest.json"
    for path in (verification_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    verification = json.loads(verification_path.read_text(encoding="utf-8-sig"))
    if verification.get("status") != "passed" or verification.get("errors"):
        raise ValueError("Protocol V2 verification is not passed with errors=[]")
    final_summary_path = (
        root
        / "04_model_training/eval_reports"
        / "svtrv2_s_m3_dual_order_s50_to_target_final_summary.json"
    )
    if not final_summary_path.is_file():
        raise FileNotFoundError(final_summary_path)
    final_summary = json.loads(final_summary_path.read_text(encoding="utf-8-sig"))
    expected_checkpoint_hash = final_summary.get("best_checkpoint_sha256")
    if sha256(checkpoint) != expected_checkpoint_hash:
        raise ValueError("Frozen M3 checkpoint SHA-256 mismatch")
    if sha256(config) != final_summary.get("config_sha256"):
        raise ValueError("Frozen M3 config SHA-256 mismatch")
    if final_summary.get("preprocess_protocol") != EXPECTED_PREPROCESS:
        raise ValueError("Frozen M3 summary uses the wrong preprocessing protocol")
    protocol_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    return {
        "protocol_verification": str(verification_path),
        "protocol_manifest": str(manifest_path),
        "data_protocol_id": protocol_manifest.get("protocol_id"),
        "data_protocol_fingerprint": protocol_manifest.get(
            "verification_fingerprint_sha256"
        ),
        "final_summary": str(final_summary_path),
        "final_summary_sha256": sha256(final_summary_path),
        "expected_checkpoint_sha256": expected_checkpoint_hash,
    }


def load_train_metadata(metadata: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(metadata)
    train = [row for row in rows if row.get("split") == "train"]
    if len(train) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"Frozen target train row count mismatch: {len(train)} != {EXPECTED_TRAIN_ROWS}"
        )
    counts = Counter(row.get("language") for row in train)
    if dict(counts) != EXPECTED_LANGUAGE_ROWS:
        raise ValueError(
            f"Frozen target train language counts mismatch: {dict(counts)}"
        )
    identifiers = [str(row.get("id") or row.get("candidate_id") or "") for row in train]
    if not all(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Target train sample IDs are missing or duplicated")
    if any(row.get("language") not in LANGUAGES for row in train):
        raise ValueError("Unexpected target train language")
    return train


def metadata_ratio(row: dict[str, Any]) -> float:
    try:
        bbox = json.loads(row.get("bbox") or "{}")
        width = float(bbox.get("width") or 0)
        height = float(bbox.get("height") or 0)
        if width > 0 and height > 0:
            return min(40.0, max(1.0, width / height))
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    try:
        return min(40.0, max(1.0, float(row.get("bbox_aspect") or 1.0)))
    except ValueError:
        return 1.0


def bucket_key(row: dict[str, Any]) -> int:
    return max(1, min(40, int(math.ceil(metadata_ratio(row)))))


def configure_recognizer(root: Path, config_path: Path, checkpoint: Path):
    openocr_root = (root / "third_party/OpenOCR").resolve()
    sys.path.insert(0, str(openocr_root))
    sys.path.insert(0, str(openocr_root / "tools"))
    from tools.engine.config import Config
    from tools.infer_rec import OpenRecognizer

    cfg = Config(str(config_path)).cfg
    if cfg["Global"].get("preprocess_protocol") != EXPECTED_PREPROCESS:
        raise ValueError("Config is not P1_MSR_V3")
    if cfg["Global"].get("method_variant") != EXPECTED_METHOD:
        raise ValueError("Config is not the frozen M3 method")
    if abs(float(cfg["Loss"].get("consistency_weight")) - EXPECTED_ALPHA) > 1e-12:
        raise ValueError("Config is not frozen M3 alpha=0.15")
    cfg["Global"]["character_dict_path"] = str(
        root / "04_model_training/character_dict_hz_ug_kk_v1/character_dict.txt"
    )
    cfg["Global"]["checkpoints"] = str(checkpoint)
    cfg["Global"]["pretrained_model"] = None
    cfg["Global"]["use_amp"] = False
    cfg["Global"]["infer_img"] = None
    recognizer = OpenRecognizer(
        config=cfg, mode="server", backend="torch", use_gpu="true", numId=0
    )
    decoder = getattr(recognizer.model, "decoder", None)
    if decoder is None or not hasattr(decoder, "ctc_decoder"):
        raise ValueError("M3 checkpoint does not expose the frozen RCTC branch")
    decoder.infer_gtc = False
    from openrec.postprocess import build_post_process

    recognizer.post_process_class = build_post_process(
        {
            "name": "CTCLabelDecode",
            "character_dict_path": cfg["Global"]["character_dict_path"],
            "use_space_char": cfg["Global"].get("use_space_char", False),
        },
        cfg["Global"],
    )
    recognizer.torch.backends.cudnn.benchmark = True
    return recognizer, cfg


def run_recognizer_with_oom_retry(recognizer, images: list[Any]) -> tuple[list[dict], int]:
    torch = recognizer.torch
    try:
        return recognizer(img_numpy_list=images, batch_num=len(images)), 0
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(images) == 1:
            raise
        split = len(images) // 2
        left, left_retries = run_recognizer_with_oom_retry(recognizer, images[:split])
        right, right_retries = run_recognizer_with_oom_retry(recognizer, images[split:])
        return left + right, left_retries + right_retries + 1


def derive_record(
    row: dict[str, Any], quality: dict[str, Any], prediction: dict[str, Any], dictionary: set[str]
) -> dict[str, Any]:
    sample_id = str(row.get("id") or row.get("candidate_id"))
    language = str(row["language"])
    gt = str(row.get("logical_text") or row.get("text") or "")
    pred_visual = str(prediction.get("text") or "")
    pred_logical = logical_prediction(pred_visual, language)
    distance = edit_distance(gt, pred_logical)
    risks = text_risk_flags(gt, dictionary) + list(quality["image_risk_flags"])
    if row.get("review_status") != "pass":
        risks.append("metadata_review_not_passed")
    if distance > 0 and normalize_for_risk(gt) == normalize_for_risk(pred_logical):
        risks.append("normalization_or_width_only_difference")
    confidence = float(prediction.get("score") or 0.0)
    if not pred_logical or confidence < 0.15:
        risks.append("possible_label_image_mismatch_or_unreadable")
    risks = sorted(set(risks))
    eligible = distance in {1, 2} and not risks
    return {
        "protocol_id": PROTOCOL_ID,
        "sample_id": sample_id,
        "language": language,
        "image_path": str(row["image"]).replace("\\", "/"),
        "gt_text": gt,
        "prediction_visual_u2": pred_visual,
        "prediction": pred_logical,
        "confidence": confidence,
        "ed": distance,
        "ed_bucket": str(distance) if distance < 3 else ">=3",
        "width": quality["width"],
        "height": quality["height"],
        "aspect_ratio": quality["aspect_ratio"],
        "contrast_span": quality["contrast_span"],
        "blur_variance": quality["blur_variance"],
        "edge_contact": quality["edge_contact"],
        "risk_flags": risks,
        "eligible_for_review": eligible,
        "manual_label_image_check_required": distance in {1, 2},
        "weight_before_manual_gate": 1.0,
        "test_evaluated": False,
    }


def select_stratified_review(
    records: list[dict[str, Any]], per_stratum: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("eligible_for_review") and int(record["ed"]) in {1, 2}:
            strata[(str(record["language"]), int(record["ed"]))].append(record)
    selected = []
    population = {}
    for language in LANGUAGES:
        for distance in (1, 2):
            key = (language, distance)
            candidates = sorted(strata[key], key=lambda row: row["sample_id"])
            stratum_name = f"{language}_ed{distance}"
            population[stratum_name] = len(candidates)
            rng = random.Random(f"{seed}:{stratum_name}")
            chosen = (
                candidates
                if len(candidates) <= per_stratum
                else rng.sample(candidates, per_stratum)
            )
            selected.extend(sorted(chosen, key=lambda row: row["sample_id"]))
    review = []
    for record in selected:
        row = {field: record.get(field, "") for field in REVIEW_FIELDS}
        row["risk_flags"] = "|".join(record.get("risk_flags") or [])
        row["manual_decision"] = ""
        row["manual_notes"] = ""
        review.append(row)
    return review, population


def mining_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_language = {}
    for language in LANGUAGES:
        rows = [record for record in records if record["language"] == language]
        by_language[language] = {
            "rows": len(rows),
            "ed": dict(Counter(record["ed_bucket"] for record in rows)),
            "eligible_ed1": sum(
                record["eligible_for_review"] and record["ed"] == 1 for record in rows
            ),
            "eligible_ed2": sum(
                record["eligible_for_review"] and record["ed"] == 2 for record in rows
            ),
            "risk_flags": dict(
                Counter(flag for record in rows for flag in record["risk_flags"])
            ),
        }
    return {
        "rows": len(records),
        "ed": dict(Counter(record["ed_bucket"] for record in records)),
        "eligible_ed1": sum(
            record["eligible_for_review"] and record["ed"] == 1 for record in records
        ),
        "eligible_ed2": sum(
            record["eligible_for_review"] and record["ed"] == 2 for record in records
        ),
        "risk_flags": dict(
            Counter(flag for record in records for flag in record["risk_flags"])
        ),
        "languages": by_language,
    }


def mine(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    output = args.output.resolve()
    target_root = root / "01_data_preparation/real_line_dataset_eval_reviewed"
    metadata = target_root / "train_reviewed/metadata.jsonl"
    image_root = target_root
    config = root / "04_model_training/configs/svtrv2_s_m3_dual_order_s50_to_target.yml"
    checkpoint = (
        root
        / "04_model_training/runs/svtrv2_s_m3_dual_order_s50_to_target"
        / "best_clean_dev_macro_cer.pth"
    )
    dictionary_path = (
        root / "04_model_training/character_dict_hz_ug_kk_v1/character_dict.txt"
    )
    import cv2  # noqa: F401
    import numpy  # noqa: F401
    import PIL  # noqa: F401

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError(
            "This audit is frozen to physical GPU1. Run with CUDA_VISIBLE_DEVICES=1."
        )
    if args.replace and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    partial = output / "target_train_predictions.jsonl.partial"
    complete = output / "target_train_predictions.jsonl"
    provenance_path = output / "mining_provenance.json"
    frozen = validate_frozen_inputs(root, config, checkpoint, metadata, dictionary_path)
    provenance = {
        "status": "in_progress",
        "protocol_id": PROTOCOL_ID,
        "root": str(root),
        "input_split": "target_domain_train_only",
        "test_evaluated": False,
        "physical_gpu": 1,
        "visible_device_inside_process": 0,
        "amp": False,
        "method": "M3 / SOAR-SVTR",
        "alpha": EXPECTED_ALPHA,
        "config": str(config),
        "config_sha256": sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "metadata": str(metadata),
        "metadata_sha256": sha256(metadata),
        "dictionary": str(dictionary_path),
        "dictionary_sha256": sha256(dictionary_path),
        "base_batch_size": args.base_batch_size,
        "target_gpu_memory_gib": args.target_gpu_memory_gib,
        "maximum_base_batch_size": args.maximum_base_batch_size,
        "io_workers": args.io_workers,
        "risk_thresholds": {
            "low_contrast_percentile_span": 20.0,
            "severe_blur_laplacian_variance": 12.0,
            "edge_contact_fraction": 0.75,
            "very_low_confidence": 0.15,
            "maximum_aspect_ratio": 40.0,
        },
        "frozen_inputs": frozen,
    }
    if complete.is_file() and provenance_path.is_file() and not args.replace:
        previous = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
        immutable = (
            "protocol_id",
            "config_sha256",
            "checkpoint_sha256",
            "metadata_sha256",
            "dictionary_sha256",
        )
        if any(previous.get(key) != provenance.get(key) for key in immutable):
            raise ValueError("Existing completed mining output has different provenance")
        if previous.get("status") == "mining_complete_review_pending":
            print(
                json.dumps(
                    {
                        "status": "TARGET_TRAIN_MINING_ALREADY_COMPLETE",
                        "output": str(complete),
                        "sha256": sha256(complete),
                        "test_evaluated": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
    if provenance_path.exists() and partial.exists() and not args.replace:
        previous = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
        immutable = (
            "protocol_id",
            "config_sha256",
            "checkpoint_sha256",
            "metadata_sha256",
            "dictionary_sha256",
        )
        if any(previous.get(key) != provenance.get(key) for key in immutable):
            raise ValueError("Cannot resume: mining provenance changed")
    write_json(provenance_path, provenance)
    train = load_train_metadata(metadata)
    dictionary = load_dictionary(dictionary_path)
    completed: set[str] = set()
    if partial.exists() and not args.replace:
        completed = {row["sample_id"] for row in read_resumable_jsonl(partial)}
    pending = [
        row
        for row in train
        if str(row.get("id") or row.get("candidate_id")) not in completed
    ]
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pending:
        groups[bucket_key(row)].append(row)
    recognizer, _ = configure_recognizer(root, config, checkpoint)
    torch = recognizer.torch
    torch.cuda.reset_peak_memory_stats()
    target_bytes = args.target_gpu_memory_gib * 1024**3
    adaptive_base = args.base_batch_size
    oom_retries = 0
    processed = len(completed)
    started = time.time()
    mode = "a" if partial.exists() and completed else "w"
    with partial.open(mode, encoding="utf-8") as output_handle, ThreadPoolExecutor(
        max_workers=args.io_workers
    ) as pool:
        for ratio in sorted(groups):
            rows = groups[ratio]
            cursor = 0
            while cursor < len(rows):
                ratio_factor = max(1.0, ratio / 4.0)
                batch_size = max(1, int(adaptive_base / ratio_factor))
                batch_rows = rows[cursor : cursor + batch_size]
                qualities = list(
                    pool.map(lambda row: image_quality(image_root / row["image"]), batch_rows)
                )
                valid_indices = [
                    index for index, quality in enumerate(qualities) if quality["image"] is not None
                ]
                predictions: dict[int, dict[str, Any]] = {}
                if valid_indices:
                    images = [qualities[index]["image"] for index in valid_indices]
                    inferred, retries = run_recognizer_with_oom_retry(recognizer, images)
                    oom_retries += retries
                    predictions.update(zip(valid_indices, inferred))
                for index, (row, quality) in enumerate(zip(batch_rows, qualities)):
                    prediction = predictions.get(index, {"text": "", "score": 0.0})
                    record = derive_record(row, quality, prediction, dictionary)
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                cursor += len(batch_rows)
                processed += len(batch_rows)
                peak = int(torch.cuda.max_memory_allocated())
                if peak < target_bytes * 0.82 and adaptive_base < args.maximum_base_batch_size:
                    adaptive_base = min(
                        args.maximum_base_batch_size, max(adaptive_base + 1, int(adaptive_base * 1.2))
                    )
                elapsed = max(time.time() - started, 1e-6)
                print(
                    json.dumps(
                        {
                            "status": "TRAIN_MINING_PROGRESS",
                            "processed": processed,
                            "total": len(train),
                            "ratio_bucket": ratio,
                            "batch": len(batch_rows),
                            "adaptive_base_batch": adaptive_base,
                            "peak_gpu_memory_gib": round(peak / 1024**3, 3),
                            "oom_retries": oom_retries,
                            "samples_per_second": round((processed - len(completed)) / elapsed, 2),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    records = read_resumable_jsonl(partial)
    if len(records) != len(train) or len({row["sample_id"] for row in records}) != len(train):
        raise RuntimeError("Mining output is incomplete or contains duplicate sample IDs")
    partial.replace(complete)
    review_rows, population = select_stratified_review(
        records, args.review_per_stratum, args.review_seed
    )
    review_path = output / "train_hard_candidate_review.csv"
    write_csv(review_path, review_rows, REVIEW_FIELDS)
    selection = {
        "protocol_id": PROTOCOL_ID,
        "review_seed": args.review_seed,
        "review_per_stratum": args.review_per_stratum,
        "population_by_stratum": population,
        "review_rows": len(review_rows),
        "membership_fields": list(MEMBERSHIP_FIELDS),
        "membership_sha256": stable_fingerprint(review_rows, MEMBERSHIP_FIELDS),
        "decision_vocabulary": sorted(ALLOWED_DECISIONS),
        "genuine_ocr_error_gate": 0.70,
        "test_evaluated": False,
    }
    write_json(output / "review_selection_manifest.json", selection)
    provenance.update(
        {
            "status": "mining_complete_review_pending",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "rows": len(records),
            "predictions_sha256": sha256(complete),
            "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
            "oom_retries": oom_retries,
            "final_adaptive_base_batch": adaptive_base,
            "mining_summary": mining_summary(records),
            "review_selection": selection,
            "formal_hem_manifest_created": False,
            "next_action": "complete_train_hard_candidate_review_csv_then_finalize",
        }
    )
    write_json(provenance_path, provenance)
    print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def assign_sample_weight(
    record: dict[str, Any], reviewed_non_genuine: set[str]
) -> tuple[float, str]:
    sample_id = str(record["sample_id"])
    if sample_id in reviewed_non_genuine:
        return 1.0, "manual_review_excluded"
    if parse_bool(record.get("eligible_for_review")) and int(record["ed"]) == 1:
        return 2.0, "clean_ed1"
    if parse_bool(record.get("eligible_for_review")) and int(record["ed"]) == 2:
        return 1.5, "clean_ed2"
    return 1.0, "ordinary"


def finalize(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    output = args.output.resolve()
    predictions_path = output / "target_train_predictions.jsonl"
    review_path = args.review_csv.resolve()
    selection_path = output / "review_selection_manifest.json"
    for path in (predictions_path, review_path, selection_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    records = read_jsonl(predictions_path)
    _, review = read_csv(review_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    if stable_fingerprint(review, MEMBERSHIP_FIELDS) != selection["membership_sha256"]:
        raise ValueError("Review membership or immutable candidate fields changed")
    decisions = [row.get("manual_decision", "").strip() for row in review]
    invalid = sorted({decision for decision in decisions if decision not in ALLOWED_DECISIONS})
    if invalid:
        raise ValueError(
            "Review is incomplete or contains invalid decisions: " + repr(invalid)
        )
    counts = Counter(decisions)
    genuine_ratio = counts["genuine_ocr_error"] / len(review) if review else 0.0
    reviewed_non_genuine = {
        row["sample_id"]
        for row in review
        if row["manual_decision"] != "genuine_ocr_error"
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "status": (
            "TRAIN_HEM_AUTHORIZED" if genuine_ratio >= 0.70 else "TRAIN_HEM_NOT_AUTHORIZED"
        ),
        "review_csv": str(review_path),
        "review_csv_sha256": sha256(review_path),
        "review_rows": len(review),
        "decision_counts": dict(counts),
        "genuine_ocr_error_ratio": genuine_ratio,
        "required_ratio": 0.70,
        "formal_hem_manifest_created": False,
        "test_evaluated": False,
    }
    if genuine_ratio < 0.70:
        report["next_action"] = "do_not_run_hem"
        write_json(output / "train_review_gate_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    manifest_path = output / "formal_hem_manifest.jsonl"
    if manifest_path.exists() and not args.replace_manifest:
        raise FileExistsError(
            f"Formal HEM manifest already exists and is immutable: {manifest_path}"
        )
    weighted = []
    for record in records:
        sample_id = record["sample_id"]
        weight, reason = assign_sample_weight(record, reviewed_non_genuine)
        weighted.append(
            {
                "sample_id": sample_id,
                "image_path": record["image_path"],
                "language": record["language"],
                "gt_text": record["gt_text"],
                "prediction": record["prediction"],
                "confidence": record["confidence"],
                "ed": record["ed"],
                "risk_flags": record["risk_flags"],
                "sample_weight": weight,
                "weight_reason": reason,
            }
        )
    if len(weighted) != EXPECTED_TRAIN_ROWS:
        raise ValueError("Formal HEM manifest would not preserve every Train row")
    train_metadata = (
        root
        / "01_data_preparation/real_line_dataset_eval_reviewed"
        / "train_reviewed/metadata.jsonl"
    )
    frozen_train = load_train_metadata(train_metadata)
    ordered_ids = [
        str(row.get("id") or row.get("candidate_id")) for row in frozen_train
    ]
    weighted_by_id = {str(row["sample_id"]): row for row in weighted}
    if len(weighted_by_id) != EXPECTED_TRAIN_ROWS or set(weighted_by_id) != set(ordered_ids):
        raise ValueError("HEM weighted membership differs from frozen Train")
    # Inference is aspect-ratio grouped. Rebind by sample ID and serialize in
    # frozen Train order because LMDB file indices follow that exact order.
    weighted = [weighted_by_id[sample_id] for sample_id in ordered_ids]
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in weighted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(manifest_path)
    report.update(
        {
            "formal_hem_manifest_created": True,
            "formal_hem_manifest": str(manifest_path),
            "formal_hem_manifest_sha256": sha256(manifest_path),
            "manifest_rows": len(weighted),
            "weight_counts": dict(Counter(str(row["sample_weight"]) for row in weighted)),
            "all_train_rows_preserved": True,
            "manifest_order": "frozen_target_train_lmdb_file_order",
            "frozen_train_modified": False,
            "next_action": "preflight_b1_and_soar_hem_training_without_test",
        }
    )
    write_json(output / "train_review_gate_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    mine_parser = subparsers.add_parser("mine")
    mine_parser.add_argument("--root", type=Path, required=True)
    mine_parser.add_argument("--output", type=Path, required=True)
    mine_parser.add_argument("--base-batch-size", type=int, default=512)
    mine_parser.add_argument("--maximum-base-batch-size", type=int, default=2048)
    mine_parser.add_argument("--target-gpu-memory-gib", type=float, default=20.0)
    mine_parser.add_argument("--io-workers", type=int, default=8)
    mine_parser.add_argument("--review-per-stratum", type=int, default=30)
    mine_parser.add_argument("--review-seed", type=int, default=20260911)
    mine_parser.add_argument("--replace", action="store_true")
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--root", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    finalize_parser.add_argument("--review-csv", type=Path, required=True)
    finalize_parser.add_argument("--replace-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "mine":
        mine(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
