import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError as exc:
    raise SystemExit("Missing Pillow. Install it with: pip install Pillow") from exc


def parse_bbox(value):
    box = json.loads(value)
    if isinstance(box, dict):
        x = float(box.get("x", 0))
        y = float(box.get("y", 0))
        w = float(box.get("width", 0))
        h = float(box.get("height", 0))
        return x, y, x + w, y + h
    if isinstance(box, list) and len(box) == 4:
        x1, y1, x2, y2 = [float(v) for v in box]
        return x1, y1, x2, y2
    raise ValueError(f"Unsupported bbox: {value}")


def language_padding(language, padding):
    if language == "zh":
        return padding, padding
    if language == "ug":
        return max(padding, 32), max(padding, 16)
    if language == "kk":
        return max(padding, 28), max(padding, 14)
    return padding, padding


def clamp_bbox(box, width, height, pad_x, pad_y):
    x1, y1, x2, y2 = box
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(width, int(round(x2 + pad_x)))
    y2 = min(height, int(round(y2 + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def safe_name(value):
    return str(value).replace(":", "_").replace("/", "_").replace("\\", "_")


def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_image_path(value):
    path = Path(value)
    if path.exists():
        return path
    parts = list(path.parts)
    if "Meme_Dataset_Project" in parts:
        idx = parts.index("Meme_Dataset_Project")
        root = Path(*parts[: idx + 1])
        candidate = root / "images" / "meme" / path.name
        if candidate.exists():
            return candidate
    return path


def resize_for_overlay(image, max_side):
    width, height = image.size
    if max(width, height) <= max_side:
        return image.copy(), 1.0
    scale = max_side / max(width, height)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(size), scale


def crop_rows(rows, output_dir, padding, overlay_max_side):
    crops_dir = output_dir / "crops"
    overlays_dir = output_dir / "overlays"
    crops_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    review_rows = []
    status_counts = Counter()

    for row in rows:
        language = row["language"]
        candidate_id = row["candidate_id"]
        crop_path = crops_dir / language / f"{safe_name(candidate_id)}.jpg"
        overlay_path = overlays_dir / language / f"{safe_name(candidate_id)}_overlay.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)

        review = dict(row)
        review.update(
            {
                "crop_path": str(crop_path),
                "crop_path_rel": str(crop_path.relative_to(output_dir)).replace("\\", "/"),
                "overlay_path": str(overlay_path),
                "overlay_path_rel": str(overlay_path.relative_to(output_dir)).replace("\\", "/"),
                "crop_size": "",
                "source_image_size": "",
                "crop_status": "pending",
                "crop_error": "",
            }
        )

        try:
            image_path = resolve_image_path(row["image_path_resolved"])
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                bbox = parse_bbox(row["bbox"])
                pad_x, pad_y = language_padding(language, padding)
                clamped = clamp_bbox(bbox, width, height, pad_x, pad_y)
                if clamped is None:
                    raise ValueError(f"Invalid bbox after clamping: {row['bbox']}")

                crop = image.crop(clamped)
                crop.save(crop_path, quality=95)

                overlay, scale = resize_for_overlay(image, overlay_max_side)
                draw = ImageDraw.Draw(overlay)
                scaled = tuple(int(round(v * scale)) for v in clamped)
                line_width = max(2, int(round(4 * scale)))
                draw.rectangle(scaled, outline=(255, 0, 0), width=line_width)
                overlay.save(overlay_path, quality=92)

                review["crop_size"] = f"{crop.width}x{crop.height}"
                review["source_image_size"] = f"{width}x{height}"
                review["crop_status"] = "ok"
        except Exception as exc:
            review["crop_status"] = "error"
            review["crop_error"] = str(exc)

        status_counts[review["crop_status"]] += 1
        review_rows.append(review)

    return review_rows, status_counts


def write_review_html(path, rows):
    table_rows = []
    for row in rows:
        text_dir = "rtl" if row["language"] == "ug" else "ltr"
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['language'])}</td>"
            f"<td>{html.escape(row['candidate_id'])}</td>"
            f"<td>{html.escape(row['source_id'])}</td>"
            f"<td dir='{text_dir}'>{html.escape(row['text'])}</td>"
            f"<td><img class='crop' src='{html.escape(row['crop_path_rel'])}'></td>"
            f"<td><img class='overlay' src='{html.escape(row['overlay_path_rel'])}'></td>"
            f"<td>{html.escape(row['crop_status'])}</td>"
            f"<td>{html.escape(row['crop_error'])}</td>"
            f"<td>{html.escape(row.get('review_status', 'pending'))}</td>"
            f"<td>{html.escape(row.get('review_note', ''))}</td>"
            "</tr>"
        )

    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>SVTRv2 pilot line crop review</title>
  <style>
    body {{ font-family: Arial, "Noto Sans", sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; position: sticky; top: 0; }}
    .crop {{ max-width: 420px; max-height: 130px; background: #eee; }}
    .overlay {{ max-width: 260px; max-height: 220px; background: #eee; }}
  </style>
</head>
<body>
  <h1>SVTRv2 pilot line crop review</h1>
  <p>Review policy: pass = complete single line and confirmed label; recrop = source is usable but bbox needs adjustment; drop = truncated, neighboring full line, inseparable multi-line text, or uncertain label.</p>
  <table>
    <thead>
      <tr>
        <th>lang</th>
        <th>candidate_id</th>
        <th>source_id</th>
        <th>text</th>
        <th>crop</th>
        <th>overlay</th>
        <th>crop_status</th>
        <th>crop_error</th>
        <th>review_status</th>
        <th>review_note</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def default_paths():
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    input_path = (
        svtr_root
        / "01_data_preparation"
        / "real_lines"
        / "pilot_manifest"
        / "train_pilot_line_candidates.csv"
    )
    output_dir = svtr_root / "01_data_preparation" / "real_lines" / "pilot_review"
    return input_path, output_dir


def main():
    default_input, default_output = default_paths()
    parser = argparse.ArgumentParser(description="Crop pilot line candidates and build an HTML review page.")
    parser.add_argument("--manifest", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--overlay-max-side", type=int, default=900)
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_rows, status_counts = crop_rows(rows, args.output_dir, args.padding, args.overlay_max_side)

    review_csv = args.output_dir / "pilot_line_crop_review.csv"
    review_html = args.output_dir / "review.html"
    write_csv(review_csv, review_rows)
    write_review_html(review_html, review_rows)

    summary = {
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "padding": args.padding,
        "input_rows": len(rows),
        "crop_status_counts": dict(status_counts),
        "outputs": {
            "review_csv": str(review_csv),
            "review_html": str(review_html),
            "crops_dir": str(args.output_dir / "crops"),
            "overlays_dir": str(args.output_dir / "overlays"),
        },
    }
    summary_path = args.output_dir / "pilot_line_crop_review_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


