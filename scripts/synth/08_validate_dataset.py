#!/usr/bin/env python3
"""Validate a generated synthetic line dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageStat


KK_SPECIFIC = set("ӘәҒғҚқҢңӨөҰұҮүҺһІі")


def default_paths() -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    root = script_path.parents[2]
    return {
        "dataset_dir": root / "03_synthetic_generation" / "synthetic_smoke_v1",
        "character_dict": root / "04_model_training" / "character_dict_hz_ug_kk_v1" / "character_dict.txt",
    }


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=defaults["dataset_dir"])
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--character-dict", type=Path, default=defaults["character_dict"])
    parser.add_argument("--min-width", type=int, default=8)
    parser.add_argument("--min-height", type=int, default=24)
    parser.add_argument("--max-aspect", type=float, default=40.0)
    parser.add_argument("--blank-std-threshold", type=float, default=1.2)
    parser.add_argument("--copy-failures", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dict(path: Path) -> set[str]:
    return {line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.rstrip("\n")}


def validate_row(row: dict, dataset_dir: Path, charset: set[str], args: argparse.Namespace) -> tuple[list[str], dict]:
    reasons = []
    metrics = {}
    text = row.get("ctc_text") or row.get("logical_text") or ""
    if not text:
        reasons.append("empty_label")
    missing = sorted({ch for ch in text if not ch.isspace() and ch not in charset}, key=ord)
    if missing:
        reasons.append("unknown_chars:" + "".join(missing))
    image_rel = row.get("image") or ""
    image_path = dataset_dir / image_rel
    if not image_rel or not image_path.exists():
        reasons.append("missing_image")
        return reasons, metrics
    try:
        img = Image.open(image_path).convert("RGB")
        img.verify()
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        reasons.append(f"read_error:{exc}")
        return reasons, metrics
    w, h = img.size
    metrics.update({"width": w, "height": h, "aspect": w / max(1, h)})
    if w < args.min_width:
        reasons.append("too_narrow")
    if h < args.min_height:
        reasons.append("too_short")
    if metrics["aspect"] > args.max_aspect:
        reasons.append("extreme_aspect")
    stat = ImageStat.Stat(img.convert("L"))
    mean = float(stat.mean[0])
    std = float(stat.stddev[0])
    metrics.update({"luminance_mean": mean, "luminance_std": std})
    if std < args.blank_std_threshold:
        reasons.append("near_blank")
    return reasons, metrics


def copy_failure(row: dict, dataset_dir: Path, quarantine: Path, reasons: list[str]) -> None:
    image_rel = row.get("image") or ""
    if image_rel:
        src = dataset_dir / image_rel
        if src.exists():
            dst = quarantine / "images" / image_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    meta = dict(row)
    meta["failure_reasons"] = reasons
    with (quarantine / "failed_metadata.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata or (args.dataset_dir / "metadata.jsonl")
    rows = read_jsonl(metadata_path)
    charset = load_dict(args.character_dict)
    failures = []
    valid_rows = []
    hashes = {}
    duplicate_hashes = []
    text_counts = Counter()
    kk_with_specific = 0
    ug_mixed = 0
    quarantine = args.dataset_dir / "data" / "quarantine"
    if args.copy_failures and quarantine.exists():
        shutil.rmtree(quarantine)
    for row in rows:
        reasons, metrics = validate_row(row, args.dataset_dir, charset, args)
        row_with_metrics = dict(row)
        row_with_metrics.update({f"validate_{k}": v for k, v in metrics.items()})
        image_path = args.dataset_dir / row.get("image", "")
        if not reasons and image_path.exists():
            sha1 = file_sha1(image_path)
            if sha1 in hashes:
                reasons.append("duplicate_image_hash")
                duplicate_hashes.append({"id": row.get("id"), "duplicate_of": hashes[sha1], "sha1": sha1})
            else:
                hashes[sha1] = row.get("id")
        text = row.get("ctc_text") or row.get("logical_text") or ""
        text_counts[text] += 1
        if row.get("language") == "kk" and any(ch in KK_SPECIFIC for ch in text):
            kk_with_specific += 1
        if row.get("language") == "ug" and any(ch.isascii() and ch.isalnum() for ch in text):
            ug_mixed += 1
        if reasons:
            fail = {"id": row.get("id"), "language": row.get("language"), "image": row.get("image"), "reasons": reasons}
            failures.append(fail)
            if args.copy_failures:
                copy_failure(row, args.dataset_dir, quarantine, reasons)
        else:
            valid_rows.append(row_with_metrics)

    duplicate_texts = sum(1 for count in text_counts.values() if count > 1)
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "metadata": str(metadata_path),
        "samples": len(rows),
        "valid": len(valid_rows),
        "failures": len(failures),
        "failure_rate": len(failures) / len(rows) if rows else 0.0,
        "per_lang": dict(Counter(row.get("language") for row in rows)),
        "valid_per_lang": dict(Counter(row.get("language") for row in valid_rows)),
        "difficulty_counts": dict(Counter(row.get("difficulty") for row in rows)),
        "unknown_char_failures": sum(1 for f in failures if any(str(r).startswith("unknown_chars:") for r in f["reasons"])),
        "duplicate_image_hashes": len(duplicate_hashes),
        "duplicate_text_unique": duplicate_texts,
        "duplicate_text_ratio": duplicate_texts / max(1, len(text_counts)),
        "kk_specific_samples": kk_with_specific,
        "kk_specific_ratio": kk_with_specific / max(1, sum(1 for r in rows if r.get("language") == "kk")),
        "ug_ascii_mixed_samples": ug_mixed,
        "ug_ascii_mixed_ratio": ug_mixed / max(1, sum(1 for r in rows if r.get("language") == "ug")),
        "outputs": {
            "validation_summary": str(args.dataset_dir / "validation_summary.json"),
            "validation_failures": str(args.dataset_dir / "validation_failures.jsonl"),
            "duplicate_hashes": str(args.dataset_dir / "duplicate_hashes.jsonl"),
            "quarantine": str(quarantine) if args.copy_failures else "",
        },
    }
    (args.dataset_dir / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.dataset_dir / "validation_failures.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in failures) + ("\n" if failures else ""),
        encoding="utf-8",
    )
    (args.dataset_dir / "duplicate_hashes.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in duplicate_hashes) + ("\n" if duplicate_hashes else ""),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
