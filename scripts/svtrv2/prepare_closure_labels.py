#!/usr/bin/env python3
"""Prepare label files for the small SVTRv2 synthetic-data closure experiment."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

LANGS = ("zh", "ug", "kk")


def default_paths() -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    root = script_path.parents[2]
    return {
        "synthetic_dir": root / "03_synthetic_generation" / "synthetic_closure_synth5k_v1",
        "real_dir": root / "01_data_preparation" / "real_line_dataset_eval_reviewed",
    }


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-dir", type=Path, default=defaults["synthetic_dir"])
    parser.add_argument("--real-dir", type=Path, default=defaults["real_dir"])
    parser.add_argument("--synthetic-val-per-lang", type=int, default=500)
    parser.add_argument("--real-samples-per-lang", type=int, default=150)
    parser.add_argument("--max-label-len", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_label_file(path: Path, rows: list[dict], image_key: str = "image", text_key: str = "ctc_text") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{row[image_key]}\t{row[text_key]}" for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def label_length(text: str) -> int:
    return len((text or "").replace(" ", ""))


def split_synthetic(args: argparse.Namespace, rng: random.Random) -> dict:
    metadata_path = args.synthetic_dir / "metadata.jsonl"
    rows = read_jsonl(metadata_path)
    rows = [
        row for row in rows
        if row.get("language") in LANGS
        and row.get("image")
        and row.get("ctc_text")
        and label_length(row["ctc_text"]) <= args.max_label_len
        and (args.synthetic_dir / row["image"]).exists()
    ]
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_lang[row["language"]].append(row)

    train_rows = []
    val_rows = []
    for lang in LANGS:
        lang_rows = list(by_lang.get(lang, []))
        rng.shuffle(lang_rows)
        n_val = min(args.synthetic_val_per_lang, max(1, len(lang_rows) // 10))
        val_rows.extend(lang_rows[:n_val])
        train_rows.extend(lang_rows[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    labels_dir = args.synthetic_dir / "labels_closure"
    write_label_file(labels_dir / "train_synth_all.txt", train_rows)
    write_label_file(labels_dir / "val_synth_all.txt", val_rows)
    for lang in LANGS:
        write_label_file(labels_dir / f"train_synth_{lang}.txt", [r for r in train_rows if r["language"] == lang])
        write_label_file(labels_dir / f"val_synth_{lang}.txt", [r for r in val_rows if r["language"] == lang])

    return {
        "synthetic_metadata": str(metadata_path),
        "synthetic_samples": len(rows),
        "synthetic_train": len(train_rows),
        "synthetic_val": len(val_rows),
        "synthetic_train_per_lang": dict(Counter(row["language"] for row in train_rows)),
        "synthetic_val_per_lang": dict(Counter(row["language"] for row in val_rows)),
        "synthetic_labels_dir": str(labels_dir),
    }


def real_text(row: dict) -> str:
    return row.get("logical_text") or row.get("text") or ""


def make_real_eval(args: argparse.Namespace, rng: random.Random) -> dict:
    metadata_path = args.real_dir / "metadata.csv"
    rows = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = real_text(row)
            image_rel = row.get("image") or ""
            if row.get("language") not in LANGS:
                continue
            if row.get("review_status", "pass") != "pass":
                continue
            if not text or label_length(text) > args.max_label_len:
                continue
            if not image_rel or not (args.real_dir / image_rel).exists():
                continue
            rows.append(
                {
                    "language": row["language"],
                    "image": image_rel,
                    "ctc_text": text,
                    "split": row.get("split", ""),
                    "id": row.get("id", ""),
                }
            )

    by_lang: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_lang[row["language"]].append(row)

    eval_rows = []
    for lang in LANGS:
        candidates = list(by_lang.get(lang, []))
        rng.shuffle(candidates)
        eval_rows.extend(candidates[:args.real_samples_per_lang])
    rng.shuffle(eval_rows)

    labels_dir = args.real_dir / "labels_closure"
    write_label_file(labels_dir / "real_pilot_all.txt", eval_rows)
    for lang in LANGS:
        write_label_file(labels_dir / f"real_pilot_{lang}.txt", [r for r in eval_rows if r["language"] == lang])

    return {
        "real_metadata": str(metadata_path),
        "real_available_after_filter": len(rows),
        "real_pilot": len(eval_rows),
        "real_pilot_per_lang": dict(Counter(row["language"] for row in eval_rows)),
        "real_labels_dir": str(labels_dir),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    summary = {
        "synthetic_dir": str(args.synthetic_dir),
        "real_dir": str(args.real_dir),
        "max_label_len": args.max_label_len,
        "seed": args.seed,
    }
    summary.update(split_synthetic(args, rng))
    summary.update(make_real_eval(args, rng))
    out = args.synthetic_dir / "closure_label_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
