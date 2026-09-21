#!/usr/bin/env python3
"""Create visual statistics/contact sheets for a synthetic dataset."""

from __future__ import annotations

import argparse
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


KK_SPECIFIC = set("\u04d8\u04d9\u0492\u0493\u049a\u049b\u04a2\u04a3\u04e8\u04e9\u04b0\u04b1\u04ae\u04af\u04ba\u04bb\u0406\u0456")


def default_paths() -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    root = script_path.parents[2]
    return {"dataset_dir": root / "03_synthetic_generation" / "synthetic_smoke_v1"}


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=defaults["dataset_dir"])
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--samples-per-sheet", type=int, default=24)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pick(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


def make_sheet(dataset_dir: Path, rows: list[dict], out_path: Path, columns: int = 4) -> None:
    thumb_w, thumb_h = 300, 120
    gap_h = 8
    rows_n = max(1, (len(rows) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows_n * (thumb_h + gap_h)), (245, 243, 238))
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        x = (i % columns) * thumb_w
        y = (i // columns) * (thumb_h + gap_h)
        color = {"zh": (48, 103, 214), "ug": (30, 132, 73), "kk": (160, 92, 26)}.get(row.get("language"), (80, 80, 80))
        draw.rectangle((x + 3, y + 3, x + thumb_w - 3, y + thumb_h + 3), outline=color, width=3)
        try:
            img = Image.open(dataset_dir / row["image"]).convert("RGB")
            img.thumbnail((thumb_w - 12, thumb_h - 12), Image.Resampling.LANCZOS)
            ox = x + (thumb_w - img.width) // 2
            oy = y + 6 + (thumb_h - img.height) // 2
            sheet.paste(img, (ox, oy))
        except Exception:
            draw.rectangle((x + 6, y + 6, x + thumb_w - 6, y + thumb_h - 6), outline=(200, 0, 0), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def section_meta(rows: list[dict]) -> str:
    lines = []
    for idx, row in enumerate(rows):
        lines.append(
            html.escape(
                f"{idx + 1:02d} {row.get('language','')} {row.get('id','')} {row.get('difficulty','')} "
                f"{row.get('background_type','')} {row.get('font_name','')} len={row.get('text_length','')} "
                f"text={row.get('logical_text','')}"
            )
        )
    return "\n".join(lines)


def write_html(output_dir: Path, sections: list[tuple[str, Path, list[dict]]]) -> None:
    items = []
    for title, path, rows in sections:
        items.append(
            f"<section><h2>{html.escape(title)}</h2><img src='{html.escape(path.name)}'><pre>{section_meta(rows)}</pre></section>"
        )
    page = """<!doctype html>
<html><head><meta charset="utf-8"><title>Synthetic Dataset Statistics</title>
<style>
body{margin:0;background:#f5f3ee;font-family:Arial,sans-serif;color:#20242a}
header{position:sticky;top:0;background:#fffdf7;border-bottom:1px solid #ddd5c7;padding:12px 18px}
main{padding:16px}
section{margin:0 0 24px;background:white;border:1px solid #d8d8d0;border-radius:6px;padding:12px}
img{max-width:100%;height:auto;border:1px solid #ded9cc;background:#ebe7dc}
pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0 0;font-size:13px;line-height:1.45}
h1,h2{margin:0 0 10px}
</style></head><body><header><h1>Synthetic Dataset Statistics</h1></header><main>
"""
    page += "\n".join(items)
    page += "\n</main></body></html>\n"
    (output_dir / "statistics.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata or (args.dataset_dir / "metadata.jsonl")
    output_dir = args.output_dir or (args.dataset_dir / "statistics")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(metadata_path)
    rng = random.Random(args.seed)
    sections: list[tuple[str, Path, list[dict]]] = []

    for lang in ("zh", "ug", "kk"):
        subset = [r for r in rows if r.get("language") == lang]
        chosen = pick(subset, args.samples_per_sheet, rng)
        out = output_dir / f"random_{lang}.jpg"
        make_sheet(args.dataset_dir, chosen, out)
        sections.append((f"Random {lang}", out, chosen))

    for field, prefix in [("font_name", "by_font"), ("background_type", "by_background"), ("difficulty", "by_difficulty")]:
        buckets = defaultdict(list)
        for row in rows:
            buckets[row.get(field, "unknown")].append(row)
        sample = []
        per_bucket = max(1, args.samples_per_sheet // max(1, len(buckets)))
        for key in sorted(buckets):
            sample.extend(pick(buckets[key], per_bucket, rng))
        chosen = pick(sample, args.samples_per_sheet, rng)
        out = output_dir / f"{prefix}.jpg"
        make_sheet(args.dataset_dir, chosen, out)
        sections.append((prefix, out, chosen))

    ug_mixed = [r for r in rows if r.get("language") == "ug" and any(ch.isascii() and ch.isalnum() for ch in r.get("logical_text", ""))]
    chosen = pick(ug_mixed, args.samples_per_sheet, rng)
    out = output_dir / "ug_mixed_ascii.jpg"
    make_sheet(args.dataset_dir, chosen, out)
    sections.append(("Uyghur mixed ASCII", out, chosen))

    kk_specific = [r for r in rows if r.get("language") == "kk" and any(ch in KK_SPECIFIC for ch in r.get("logical_text", ""))]
    chosen = pick(kk_specific, args.samples_per_sheet, rng)
    out = output_dir / "kk_specific_chars.jpg"
    make_sheet(args.dataset_dir, chosen, out)
    sections.append(("Kazakh specific chars", out, chosen))

    long_rows = sorted(rows, key=lambda r: int(r.get("text_length") or len(r.get("logical_text", ""))), reverse=True)
    chosen = long_rows[: args.samples_per_sheet]
    out = output_dir / "long_text.jpg"
    make_sheet(args.dataset_dir, chosen, out)
    sections.append(("Long text", out, chosen))

    write_html(output_dir, sections)
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "metadata": str(metadata_path),
        "samples": len(rows),
        "per_lang": dict(Counter(r.get("language") for r in rows)),
        "difficulty": dict(Counter(r.get("difficulty") for r in rows)),
        "background_type": dict(Counter(r.get("background_type") for r in rows)),
        "font_top20": Counter(r.get("font_name") for r in rows).most_common(20),
        "ug_mixed_ascii": len(ug_mixed),
        "kk_specific_chars": len(kk_specific),
        "outputs": {"statistics_html": str(output_dir / "statistics.html"), "output_dir": str(output_dir)},
    }
    (output_dir / "statistics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
