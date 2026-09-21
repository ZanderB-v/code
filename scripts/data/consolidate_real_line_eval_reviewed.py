#!/usr/bin/env python3
"""Consolidate reviewed real meme line data into one pass-only dataset root."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path


CSV_ENCODING = "utf-8-sig"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pass-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_id(value: str) -> str:
    value = value.strip()
    value = value.replace(":", "_").replace("/", "_").replace("\\", "_")
    value = re.sub(r"[^0-9A-Za-z_\-.]+", "_", value)
    return value.strip("._") or "sample"


def clean_label(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"[\r\n\u2028\u2029\x85]+", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_labels(path: Path, rows: list[dict[str, object]], image_key: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(f"{row[image_key]}\t{row['text']}\n")


def unique_name(used: set[str], preferred: str) -> str:
    stem = Path(preferred).stem
    suffix = Path(preferred).suffix
    candidate = preferred
    idx = 2
    while candidate in used:
        candidate = f"{stem}_{idx:04d}{suffix}"
        idx += 1
    used.add(candidate)
    return candidate


def copy_image(src: Path, dst: Path, dry_run: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def normalize_review_row(
    row: dict[str, str],
    split: str,
    split_dir_name: str,
    split_crop_rel: str,
    root_crop_rel: str,
    source_csv: Path,
    source_crop: Path,
    overlay_rel: str = "",
) -> dict[str, object]:
    text_raw = row.get("logical_text") or row.get("text") or ""
    text = clean_label(text_raw)
    candidate_id = row.get("candidate_id") or row.get("id") or Path(split_crop_rel).stem
    return {
        "candidate_id": candidate_id,
        "id": clean_id(candidate_id),
        "language": row.get("language", ""),
        "split": split,
        "source_id": row.get("source_id", ""),
        "source_image": row.get("source_image", ""),
        "image_path_resolved": row.get("image_path_resolved", ""),
        "crop_path_rel": split_crop_rel,
        "image": root_crop_rel,
        "overlay_path_rel": overlay_rel,
        "text": text,
        "logical_text": text_raw,
        "text_length": len(text),
        "bbox": row.get("bbox", ""),
        "bbox_aspect": row.get("bbox_aspect", ""),
        "review_status": "pass",
        "review_note": row.get("review_note", ""),
        "source_csv": str(source_csv),
        "source_crop": str(source_crop),
        "split_dir": split_dir_name,
    }


def build_train_rows(train_pass_dir: Path, build_root: Path, dry_run: bool) -> list[dict[str, object]]:
    source_csv = train_pass_dir / "train_metadata.csv"
    rows = read_csv(source_csv)
    out_rows: list[dict[str, object]] = []
    used_by_lang: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        language = row.get("language", "")
        src = train_pass_dir / row["image"]
        name = unique_name(used_by_lang[language], Path(row["image"]).name)
        split_crop_rel = Path("crops") / language / name
        root_crop_rel = Path("train_reviewed") / split_crop_rel
        dst = build_root / root_crop_rel
        copy_image(src, dst, dry_run)
        out_rows.append(
            normalize_review_row(
                row=row,
                split="train",
                split_dir_name="train_reviewed",
                split_crop_rel=split_crop_rel.as_posix(),
                root_crop_rel=root_crop_rel.as_posix(),
                source_csv=source_csv,
                source_crop=src,
            )
        )
    return out_rows


def build_eval_rows(
    original_root: Path, build_root: Path, split: str, dry_run: bool
) -> list[dict[str, object]]:
    split_dir_name = f"{split}_reviewed"
    source_dir = original_root / split_dir_name
    source_csv = source_dir / f"{split}.csv"
    rows = [row for row in read_csv(source_csv) if row.get("review_status", "").lower() == "pass"]
    out_rows: list[dict[str, object]] = []
    used_by_lang: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        language = row.get("language", "")
        src = source_dir / row["crop_path_rel"]
        src_suffix = src.suffix if src.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
        preferred = clean_id(row.get("candidate_id", "")) + src_suffix
        name = unique_name(used_by_lang[language], preferred)
        split_crop_rel = Path("crops") / language / name
        root_crop_rel = Path(split_dir_name) / split_crop_rel
        copy_image(src, build_root / root_crop_rel, dry_run)

        overlay_rel = ""
        old_overlay_rel = row.get("overlay_path_rel", "").strip()
        if old_overlay_rel:
            old_overlay = source_dir / old_overlay_rel
            if old_overlay.exists():
                overlay_name = unique_name(
                    used_by_lang[f"{language}_overlay"], clean_id(row.get("candidate_id", "")) + old_overlay.suffix
                )
                new_overlay_rel = Path("overlays") / language / overlay_name
                copy_image(old_overlay, build_root / split_dir_name / new_overlay_rel, dry_run)
                overlay_rel = new_overlay_rel.as_posix()

        out_rows.append(
            normalize_review_row(
                row=row,
                split=split,
                split_dir_name=split_dir_name,
                split_crop_rel=split_crop_rel.as_posix(),
                root_crop_rel=root_crop_rel.as_posix(),
                source_csv=source_csv,
                source_crop=src,
                overlay_rel=overlay_rel,
            )
        )
    return out_rows


def write_split_outputs(build_root: Path, split: str, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    split_dir = build_root / f"{split}_reviewed"
    split_dir.mkdir(parents=True, exist_ok=True)
    write_csv(split_dir / f"{split}.csv", rows, fieldnames)
    write_csv(split_dir / "metadata.csv", rows, fieldnames)
    write_jsonl(split_dir / "metadata.jsonl", rows)
    write_labels(split_dir / "labels.txt", rows, "crop_path_rel")


def main() -> None:
    args = parse_args()
    if not args.eval_root.exists():
        raise FileNotFoundError(args.eval_root)
    if not args.train_pass_dir.exists():
        raise FileNotFoundError(args.train_pass_dir)
    if not args.overwrite and not args.dry_run:
        raise SystemExit("Pass --overwrite to replace eval-root with the consolidated pass-only dataset.")

    parent = args.eval_root.parent
    build_root = parent / f"{args.eval_root.name}.__pass_only_build_tmp"
    if build_root.exists() and not args.dry_run:
        shutil.rmtree(build_root)
    if not args.dry_run:
        build_root.mkdir(parents=True)

    splits = {
        "train": build_train_rows(args.train_pass_dir, build_root, args.dry_run),
        "dev": build_eval_rows(args.eval_root, build_root, "dev", args.dry_run),
        "test": build_eval_rows(args.eval_root, build_root, "test", args.dry_run),
    }
    all_rows = [row for split in ("train", "dev", "test") for row in splits[split]]

    fieldnames = [
        "candidate_id",
        "id",
        "language",
        "split",
        "source_id",
        "source_image",
        "image_path_resolved",
        "crop_path_rel",
        "image",
        "overlay_path_rel",
        "text",
        "logical_text",
        "text_length",
        "bbox",
        "bbox_aspect",
        "review_status",
        "review_note",
        "source_csv",
        "source_crop",
        "split_dir",
    ]

    summary = {
        "eval_root": str(args.eval_root),
        "train_pass_dir": str(args.train_pass_dir),
        "rows": {split: len(rows) for split, rows in splits.items()},
        "by_language": {
            split: dict(Counter(row["language"] for row in rows)) for split, rows in splits.items()
        },
        "total": len(all_rows),
        "dry_run": args.dry_run,
    }

    if not args.dry_run:
        for split, rows in splits.items():
            write_split_outputs(build_root, split, rows, fieldnames)
        write_csv(build_root / "metadata.csv", all_rows, fieldnames)
        write_jsonl(build_root / "metadata.jsonl", all_rows)
        write_labels(build_root / "labels.txt", all_rows, "image")
        with (build_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        backup = parent / f"{args.eval_root.name}.backup_before_pass_only_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(args.eval_root), str(backup))
        shutil.move(str(build_root), str(args.eval_root))
        summary["backup"] = str(backup)
        if not args.keep_backup:
            shutil.rmtree(backup)
            summary["backup_removed"] = True
        else:
            summary["backup_removed"] = False
        with (args.eval_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
