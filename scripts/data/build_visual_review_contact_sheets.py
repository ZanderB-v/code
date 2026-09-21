#!/usr/bin/env python3
"""Build readable contact sheets for visual auditing of train-review pages.

The script never changes review CSV files. It ranks rows by overlap with other
line boxes from the same source image, then writes a manifest and contact sheets
that can be inspected before decisions are merged back into the annotation UI.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_bbox(value):
    box = json.loads(value) if isinstance(value, str) else value
    x = float(box["x"])
    y = float(box["y"])
    width = float(box["width"])
    height = float(box["height"])
    return x, y, x + width, y + height


def intersection(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width, height, width * height


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rank_page_rows(all_rows, page_rows):
    groups = defaultdict(list)
    for row in all_rows:
        groups[(row["language"], row["source_id"])].append(row)

    ranked = []
    for page_index, row in enumerate(page_rows, start=1):
        box = parse_bbox(row["bbox"])
        box_area = max(area(box), 1.0)
        best_score = 0.0
        best_height = 0.0
        neighbors = []
        for other in groups[(row["language"], row["source_id"])]:
            if other["candidate_id"] == row["candidate_id"]:
                continue
            other_box = parse_bbox(other["bbox"])
            width, height, overlap_area = intersection(box, other_box)
            if overlap_area <= 0:
                continue
            denominator = max(min(box_area, area(other_box)), 1.0)
            score = overlap_area / denominator
            neighbors.append(
                {
                    "candidate_id": other["candidate_id"],
                    "overlap_width": round(width, 2),
                    "overlap_height": round(height, 2),
                    "overlap_fraction": round(score, 6),
                }
            )
            best_score = max(best_score, score)
            best_height = max(best_height, height)

        item = dict(row)
        item["page_row"] = page_index
        item["max_overlap_fraction"] = round(best_score, 6)
        item["max_overlap_height"] = round(best_height, 2)
        item["overlap_neighbors"] = json.dumps(neighbors, ensure_ascii=False)
        ranked.append(item)

    ranked.sort(
        key=lambda row: (
            float(row["max_overlap_fraction"]),
            float(row["max_overlap_height"]),
        ),
        reverse=True,
    )
    return ranked


def load_font(size):
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def resolve_crop(train_review_dir, row):
    return (train_review_dir / row["crop_path_rel"]).resolve()


def build_sheet(rows, train_review_dir, output_path, sheet_number):
    columns = 2
    rows_per_column = 5
    cell_width = 1200
    cell_height = 300
    header_height = 42
    sheet = Image.new("RGB", (columns * cell_width, rows_per_column * cell_height), "#f5f3ed")
    draw = ImageDraw.Draw(sheet)
    font = load_font(22)
    small_font = load_font(18)

    for index, row in enumerate(rows):
        column = index % columns
        line = index // columns
        left = column * cell_width
        top = line * cell_height
        draw.rectangle(
            (left + 2, top + 2, left + cell_width - 2, top + cell_height - 2),
            outline="#848b91",
            width=2,
        )
        title = (
            f"#{row['page_row']}  {row['candidate_id']}  "
            f"overlap={float(row['max_overlap_fraction']):.3f}  "
            f"h={float(row['max_overlap_height']):.0f}"
        )
        draw.text((left + 10, top + 8), title, fill="#111111", font=font)

        crop_path = resolve_crop(train_review_dir, row)
        image_top = top + header_height
        image_box = (left + 10, image_top, left + cell_width - 10, top + cell_height - 10)
        if not crop_path.exists():
            draw.text((left + 10, image_top + 20), f"MISSING: {crop_path}", fill="#b00020", font=small_font)
            continue
        try:
            with Image.open(crop_path) as source:
                crop = ImageOps.exif_transpose(source).convert("RGB")
                fitted = ImageOps.contain(
                    crop,
                    (image_box[2] - image_box[0], image_box[3] - image_box[1]),
                    method=Image.Resampling.LANCZOS,
                )
            paste_x = image_box[0] + (image_box[2] - image_box[0] - fitted.width) // 2
            paste_y = image_box[1] + (image_box[3] - image_box[1] - fitted.height) // 2
            sheet.paste(fitted, (paste_x, paste_y))
        except Exception as exc:
            draw.text((left + 10, image_top + 20), f"ERROR: {exc}", fill="#b00020", font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return {
        "sheet": sheet_number,
        "file": output_path.name,
        "samples": [row["candidate_id"] for row in rows],
    }


def write_manifest(rows, path):
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--samples-per-sheet", type=int, default=10)
    args = parser.parse_args()

    train_review_dir = args.dataset_root / "train_review"
    all_rows = read_rows(train_review_dir / "train_candidates_all.csv")
    page_path = train_review_dir / f"train_candidates_{args.page:04d}.csv"
    page_rows = read_rows(page_path)
    ranked = rank_page_rows(all_rows, page_rows)
    selected = ranked[: args.max_samples] if args.max_samples > 0 else ranked

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"page_{args.page:04d}_visual_candidates.csv"
    write_manifest(selected, manifest_path)

    sheets = []
    size = max(1, args.samples_per_sheet)
    for offset in range(0, len(selected), size):
        sheet_number = offset // size + 1
        output_path = args.output_dir / f"page_{args.page:04d}_sheet_{sheet_number:04d}.jpg"
        sheets.append(
            build_sheet(
                selected[offset : offset + size],
                train_review_dir,
                output_path,
                sheet_number,
            )
        )

    summary = {
        "page": args.page,
        "page_rows": len(page_rows),
        "selected_rows": len(selected),
        "overlap_rows_on_page": sum(float(row["max_overlap_fraction"]) > 0 for row in ranked),
        "manifest": str(manifest_path),
        "sheets": sheets,
    }
    (args.output_dir / f"page_{args.page:04d}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
