#!/usr/bin/env python3
"""Prepare Uyghur label files for RTL direction experiments.

U1 keeps Uyghur labels in Unicode logical order and relies on the model to
reverse the CTC time sequence.

U2 converts Uyghur labels to visual left-to-right order for CTC alignment.
The conversion uses python-bidi when available; otherwise it falls back to a
simple character reverse and records that fallback in the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


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
    parser.add_argument("--synthetic-val-count", type=int, default=500)
    parser.add_argument("--real-pilot-count", type=int, default=150)
    parser.add_argument("--max-label-len", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--labels-subdir", default="labels_ug_direction")
    parser.add_argument(
        "--require-python-bidi",
        action="store_true",
        help="Fail instead of falling back to simple reverse when python-bidi is unavailable.",
    )
    return parser.parse_args()


def get_bidi_display():
    try:
        from bidi.algorithm import get_display  # type: ignore

        return get_display, "python-bidi"
    except Exception:
        return None, "simple_reverse_fallback"


GET_DISPLAY, VISUAL_ORDER_METHOD = get_bidi_display()


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def label_length(text: str) -> int:
    return len(normalize_text(text).replace(" ", ""))


def ug_logical_to_visual(text: str) -> str:
    text = normalize_text(text)
    if GET_DISPLAY is not None:
        return GET_DISPLAY(text, base_dir="R")
    return text[::-1]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_label_file(path: Path, rows: list[dict], text_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{row['image']}\t{row[text_key]}" for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def make_synthetic_labels(args: argparse.Namespace, rng: random.Random) -> dict:
    metadata_path = args.synthetic_dir / "metadata.jsonl"
    rows = []
    for row in read_jsonl(metadata_path):
        if row.get("language") != "ug":
            continue
        image = row.get("image") or ""
        logical = normalize_text(row.get("logical_text") or row.get("ctc_text") or "")
        if not image or not logical:
            continue
        if label_length(logical) > args.max_label_len:
            continue
        if not (args.synthetic_dir / image).exists():
            continue
        rows.append(
            {
                "image": image,
                "logical_text": logical,
                "visual_text": ug_logical_to_visual(logical),
                "id": row.get("id", ""),
                "source": row.get("source", ""),
            }
        )

    rng.shuffle(rows)
    n_val = min(args.synthetic_val_count, max(1, len(rows) // 10))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    labels_dir = args.synthetic_dir / args.labels_subdir
    write_label_file(labels_dir / "u1_train_synth_ug_logical.txt", train_rows, "logical_text")
    write_label_file(labels_dir / "u1_val_synth_ug_logical.txt", val_rows, "logical_text")
    write_label_file(labels_dir / "u2_train_synth_ug_visual.txt", train_rows, "visual_text")
    write_label_file(labels_dir / "u2_val_synth_ug_visual.txt", val_rows, "visual_text")

    return {
        "synthetic_metadata": str(metadata_path),
        "synthetic_ug_after_filter": len(rows),
        "synthetic_train": len(train_rows),
        "synthetic_val": len(val_rows),
        "synthetic_labels_dir": str(labels_dir),
    }


def real_text(row: dict) -> str:
    return normalize_text(row.get("logical_text") or row.get("text") or "")


def make_real_labels(args: argparse.Namespace, rng: random.Random) -> dict:
    metadata_path = args.real_dir / "metadata.csv"
    rows = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("language") != "ug":
                continue
            if row.get("review_status", "pass") != "pass":
                continue
            image = row.get("image") or ""
            logical = real_text(row)
            if not image or not logical:
                continue
            if label_length(logical) > args.max_label_len:
                continue
            if not (args.real_dir / image).exists():
                continue
            rows.append(
                {
                    "image": image,
                    "logical_text": logical,
                    "visual_text": ug_logical_to_visual(logical),
                    "split": row.get("split", ""),
                    "id": row.get("id", ""),
                }
            )

    by_split: dict[str, list[dict]] = {}
    for split in ("train", "dev", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        rng.shuffle(split_rows)
        by_split[split] = split_rows

    pilot_pool = list(by_split.get("train") or rows)
    rng.shuffle(pilot_pool)
    pilot_rows = pilot_pool[: args.real_pilot_count]

    labels_dir = args.real_dir / args.labels_subdir
    write_label_file(labels_dir / "real_pilot_ug_logical.txt", pilot_rows, "logical_text")
    write_label_file(labels_dir / "real_pilot_ug_visual.txt", pilot_rows, "visual_text")
    for split, split_rows in by_split.items():
        write_label_file(labels_dir / f"{split}_ug_logical.txt", split_rows, "logical_text")
        write_label_file(labels_dir / f"{split}_ug_visual.txt", split_rows, "visual_text")

    return {
        "real_metadata": str(metadata_path),
        "real_ug_after_filter": len(rows),
        "real_ug_after_filter_by_split": dict(Counter(row.get("split", "") for row in rows)),
        "real_pilot": len(pilot_rows),
        "real_labels_dir": str(labels_dir),
    }


def main() -> None:
    args = parse_args()
    if args.require_python_bidi and VISUAL_ORDER_METHOD != "python-bidi":
        raise RuntimeError(
            "python-bidi is required for this run. Install it with: "
            "python -m pip install python-bidi"
        )
    rng = random.Random(args.seed)
    summary = {
        "task": "ug_direction_labels",
        "max_label_len": args.max_label_len,
        "seed": args.seed,
        "labels_subdir": args.labels_subdir,
        "visual_order_method": VISUAL_ORDER_METHOD,
        "note": "U1 uses logical labels with decoder reverse_sequence=True; U2 uses visual-order labels with the normal decoder.",
    }
    summary.update(make_synthetic_labels(args, rng))
    summary.update(make_real_labels(args, rng))

    out = args.synthetic_dir / "ug_direction_label_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
