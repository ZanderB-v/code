#!/usr/bin/env python3
"""Build a pass-only real line OCR dataset from reviewed CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


CSV_ENCODING = "utf-8-sig"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-review-dir", type=Path, required=True)
    parser.add_argument("--dev-review-dir", type=Path, required=True)
    parser.add_argument("--test-review-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_id(value: str) -> str:
    value = value.strip()
    value = value.replace(":", "_").replace("/", "_").replace("\\", "_")
    value = re.sub(r"[^0-9A-Za-z_\-.]+", "_", value)
    return value.strip("._") or "sample"


def one_line_label(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"[\r\n]+", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        return list(csv.DictReader(f))


def resolve_crop_path(csv_path: Path, row: dict[str, str], split: str) -> Path:
    crop_rel = row.get("crop_path_rel", "").strip()
    if not crop_rel:
        raise FileNotFoundError("empty crop_path_rel")

    crop_path = Path(crop_rel)
    if crop_path.is_absolute() and crop_path.exists():
        return crop_path

    candidates = [csv_path.parent / crop_rel]
    if split == "train":
        candidates.append(csv_path.parent / "train_review" / crop_rel)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(str(candidates[0]))


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


def write_labels(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(f"{row['image']}\t{row['text']}\n")


def iter_split_csvs(train_dir: Path, dev_dir: Path, test_dir: Path):
    for csv_path in sorted(train_dir.glob("train_*.csv"), key=lambda p: p.name):
        yield "train", csv_path
    yield "dev", dev_dir / "dev.csv"
    yield "test", test_dir / "test.csv"


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists; pass --overwrite to replace: {args.output_dir}")
        if not args.dry_run:
            shutil.rmtree(args.output_dir)

    split_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    all_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "input": {
            "train_review_dir": str(args.train_review_dir),
            "dev_review_dir": str(args.dev_review_dir),
            "test_review_dir": str(args.test_review_dir),
        },
        "output_dir": str(args.output_dir),
        "status_counts": defaultdict(Counter),
        "kept_counts": defaultdict(Counter),
        "missing_crops": [],
        "duplicate_output_ids": 0,
        "dry_run": args.dry_run,
    }
    used_ids: set[str] = set()

    for split, csv_path in iter_split_csvs(
        args.train_review_dir, args.dev_review_dir, args.test_review_dir
    ):
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        for row in read_csv(csv_path):
            status = row.get("review_status", "").strip().lower()
            language = row.get("language", "").strip()
            summary["status_counts"][split][status] += 1
            if status != "pass":
                continue

            try:
                src_image = resolve_crop_path(csv_path, row, split)
            except FileNotFoundError as exc:
                summary["missing_crops"].append(
                    {
                        "split": split,
                        "csv": str(csv_path),
                        "candidate_id": row.get("candidate_id", ""),
                        "crop_path_rel": row.get("crop_path_rel", ""),
                        "error": str(exc),
                    }
                )
                continue

            base_id = clean_id(row.get("candidate_id", ""))
            output_id = base_id
            if output_id in used_ids:
                summary["duplicate_output_ids"] += 1
                output_id = f"{base_id}_{len(used_ids) + 1:06d}"
            used_ids.add(output_id)

            suffix = src_image.suffix.lower() if src_image.suffix else ".jpg"
            rel_image = Path("images") / split / language / f"{output_id}{suffix}"
            dst_image = args.output_dir / rel_image
            if not args.dry_run:
                dst_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_image, dst_image)

            text_raw = row.get("text", "")
            text = one_line_label(text_raw)
            out_row: dict[str, object] = {
                "id": output_id,
                "image": rel_image.as_posix(),
                "text": text,
                "logical_text": text_raw,
                "language": language,
                "split": split,
                "source_id": row.get("source_id", ""),
                "source_image": row.get("source_image", ""),
                "bbox": row.get("bbox", ""),
                "bbox_aspect": row.get("bbox_aspect", ""),
                "text_length": row.get("text_length", ""),
                "review_status": status,
                "review_note": row.get("review_note", ""),
                "source_csv": str(csv_path),
                "source_crop": str(src_image),
            }
            split_rows[split].append(out_row)
            all_rows.append(out_row)
            summary["kept_counts"][split][language] += 1

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "id",
            "image",
            "text",
            "logical_text",
            "language",
            "split",
            "source_id",
            "source_image",
            "bbox",
            "bbox_aspect",
            "text_length",
            "review_status",
            "review_note",
            "source_csv",
            "source_crop",
        ]
        write_csv(args.output_dir / "metadata.csv", all_rows, fieldnames)
        write_jsonl(args.output_dir / "metadata.jsonl", all_rows)
        write_labels(args.output_dir / "labels.txt", all_rows)

        for split, rows in split_rows.items():
            write_csv(args.output_dir / f"{split}_metadata.csv", rows, fieldnames)
            write_jsonl(args.output_dir / f"{split}_metadata.jsonl", rows)
            write_labels(args.output_dir / f"{split}_labels.txt", rows)

    serializable_summary = {
        **summary,
        "status_counts": {
            split: dict(counter) for split, counter in summary["status_counts"].items()
        },
        "kept_counts": {
            split: dict(counter) for split, counter in summary["kept_counts"].items()
        },
        "kept_total": len(all_rows),
        "missing_crop_count": len(summary["missing_crops"]),
    }
    if not args.dry_run:
        with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(serializable_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(serializable_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
