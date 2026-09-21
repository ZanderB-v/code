import argparse
import csv
import json
import random
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image

from build_pilot_annotation_html import build_html, read_rows


LANGS = ("zh", "ug", "kk")
MARKERS = (
    "Misogyny_Dataset_Project/",
    "Meme_Dataset_Project/",
    "final_multilingual_meme_ocr_dataset/",
)


def safe_id(value):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_") or "sample"


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text or ""))
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def resolve_path(value, workspace_root):
    if not value:
        return None
    raw = str(value)
    raw = raw.replace("/home/data_home/wudayu", "/data_home/wudayu")
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path
    candidate = workspace_root / raw
    if candidate.exists():
        return candidate

    normalized = raw.replace("\\", "/")
    if ":/" in normalized:
        normalized = normalized.split(":/", 1)[1]
    for marker in MARKERS:
        if marker in normalized:
            candidate = workspace_root / normalized[normalized.index(marker):]
            if candidate.exists():
                return candidate
    return None


def rendered_image_path(record, workspace_root):
    for key in ("rendered_image_abs", "rendered_image_workspace", "rendered_image"):
        path = resolve_path(record.get(key), workspace_root)
        if path is not None:
            return path
    return None


def source_image_path(record, workspace_root):
    for key in ("source_image_abs", "source_image_workspace", "source_image"):
        path = resolve_path(record.get(key), workspace_root)
        if path is not None:
            return path
    return None


def image_stem(record):
    for key in ("source_image_workspace", "source_image_abs", "source_image", "image"):
        value = record.get(key)
        if value:
            return Path(str(value)).stem
    return safe_id(record.get("sample_id") or "unknown")


def source_family_id(record):
    dataset = record.get("source_dataset") or record.get("dataset_source") or "unknown_dataset"
    return f"{dataset}:{image_stem(record)}"


def box_xyxy(box):
    if isinstance(box, dict):
        x = float(box.get("x", 0))
        y = float(box.get("y", 0))
        w = float(box.get("width", 0))
        h = float(box.get("height", 0))
        return x, y, x + w, y + h
    if isinstance(box, (list, tuple)) and len(box) == 4 and not isinstance(box[0], (list, tuple)):
        x, y, w, h = [float(v) for v in box]
        return x, y, x + w, y + h
    if isinstance(box, (list, tuple)) and len(box) >= 4 and isinstance(box[0], (list, tuple)):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        return min(xs), min(ys), max(xs), max(ys)
    return 0.0, 0.0, 0.0, 0.0


def clamp_box(box, width, height, pad_x, pad_y):
    x1, y1, x2, y2 = box
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(width, int(round(x2 + pad_x)))
    y2 = min(height, int(round(y2 + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def padding_for_language(language, pad_x, pad_y):
    if language == "ug":
        return max(pad_x, 24), max(pad_y, 12)
    if language == "kk":
        return max(pad_x, 20), max(pad_y, 10)
    return pad_x, pad_y


def line_text(line, language):
    if language == "zh":
        return normalize_text(line.get("zh_text") or line.get("source_text") or line.get("text"))
    return normalize_text(
        line.get("render_text_normalized")
        or line.get(f"{language}_text")
        or line.get("text")
    )


def line_bbox(line, language, bbox_field="render_layout_bbox"):
    if language == "zh":
        return line.get("bbox") or line.get("polygon")
    if bbox_field != "auto":
        return line.get(bbox_field) or line.get("render_layout_bbox") or line.get("render_bbox") or line.get("render_source_bbox") or line.get("bbox")
    return line.get("render_layout_bbox") or line.get("render_bbox") or line.get("render_source_bbox") or line.get("bbox")


def load_records(path):
    return json.loads(path.read_text(encoding="utf-8"))


def text_len_ok(text, language, args):
    length = len(text)
    if length < args.min_text_len:
        return False
    if language == "zh":
        return length <= args.max_text_len_zh
    return length <= args.max_text_len_ugkk


def quality_bucket(width, height, aspect, text, language, args):
    if width < args.min_width or height < args.min_height:
        return "drop_size"
    if aspect < args.min_aspect or aspect > args.max_aspect:
        return "hard_aspect"
    if not text_len_ok(text, language, args):
        return "drop_text_length"
    return "auto"


def collect_candidates(records_by_lang, workspace_root, languages, args):
    candidates = []
    seen = set()
    zh_source_records = records_by_lang.get("ug") or records_by_lang.get("kk") or []

    if "zh" in languages:
        for record in zh_source_records:
            family = source_family_id(record)
            image_path = source_image_path(record, workspace_root)
            if image_path is None:
                continue
            for line in record.get("texts") or []:
                text = line_text(line, "zh")
                bbox = line_bbox(line, "zh", args.bbox_field)
                if not text or not bbox:
                    continue
                line_index = int(line.get("line_index", 0))
                cid = f"{safe_id(family)}_zh_L{line_index + 1:03d}"
                if cid in seen:
                    continue
                seen.add(cid)
                candidates.append(make_candidate(record, line, "zh", cid, family, image_path, text, bbox, line_index, "natural_chinese_meme_line"))

    for language in ("ug", "kk"):
        if language not in languages:
            continue
        for record in records_by_lang.get(language, []):
            family = source_family_id(record)
            image_path = rendered_image_path(record, workspace_root)
            if image_path is None:
                continue
            for line in record.get("texts") or []:
                text = line_text(line, language)
                bbox = line_bbox(line, language, args.bbox_field)
                if not text or not bbox:
                    continue
                line_index = int(line.get("line_index", 0))
                cid = f"{safe_id(family)}_{language}_L{line_index + 1:03d}"
                if cid in seen:
                    continue
                seen.add(cid)
                candidates.append(make_candidate(record, line, language, cid, family, image_path, text, bbox, line_index, f"in_domain_rendered_{language}_meme_line"))
    return candidates


def make_candidate(record, line, language, cid, family, image_path, text, bbox, line_index, data_type):
    return {
        "candidate_id": cid,
        "language": language,
        "split": record.get("final_split") or record.get("split") or "train",
        "source_family_id": family,
        "source_id": family,
        "source_image": record.get("source_image_workspace") or record.get("source_image_abs") or record.get("source_image") or "",
        "image_path": str(image_path),
        "text": text,
        "text_length": len(text),
        "line_index": line_index,
        "bbox": bbox,
        "bbox_xyxy": box_xyxy(bbox),
        "data_type": data_type,
        "sample_id": record.get("sample_id") or "",
        "render_font_size": line.get("render_fitted_font_size", ""),
        "render_fill": line.get("render_fill", ""),
        "render_stroke": line.get("render_stroke", ""),
        "render_shadow": line.get("render_shadow", ""),
        "render_scale_x": line.get("render_scale_x", ""),
    }


def maybe_sample(candidates, max_per_lang, seed):
    if max_per_lang <= 0:
        return candidates
    rng = random.Random(seed)
    selected = []
    for language in LANGS:
        items = [c for c in candidates if c["language"] == language]
        rng.shuffle(items)
        selected.extend(items[:max_per_lang])
    selected.sort(key=lambda item: (item["language"], item["source_family_id"], item["line_index"]))
    return selected


def write_outputs(rows, output_dir):
    metadata_path = output_dir / "metadata.jsonl"
    labels_path = output_dir / "labels.txt"
    csv_path = output_dir / "metadata.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    labels_path.write_text("\n".join(f"{row['image']}\t{row['text']}" for row in rows if row.get("subset") == "train_auto") + "\n", encoding="utf-8")

    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def crop_all(candidates, output_dir, args):
    rows = []
    counts = Counter()
    by_image = {}
    for item in candidates:
        by_image.setdefault(item["image_path"], []).append(item)

    for image_idx, (image_path_str, items) in enumerate(sorted(by_image.items()), 1):
        image_path = Path(image_path_str)
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as exc:
            for item in items:
                rows.append(error_row(item, "drop_image_open", repr(exc)))
                counts["drop_image_open"] += 1
            continue

        for item in items:
            pad_x, pad_y = padding_for_language(item["language"], args.pad_x, args.pad_y)
            crop_box = clamp_box(item["bbox_xyxy"], image.width, image.height, pad_x, pad_y)
            if crop_box is None:
                rows.append(error_row(item, "drop_invalid_bbox", compact_json(item["bbox"])))
                counts["drop_invalid_bbox"] += 1
                continue

            x1, y1, x2, y2 = crop_box
            width = x2 - x1
            height = y2 - y1
            aspect = width / max(1, height)
            bucket = quality_bucket(width, height, aspect, item["text"], item["language"], args)
            if bucket.startswith("drop"):
                rows.append(error_row(item, bucket, f"{width}x{height}, len={item['text_length']}"))
                counts[bucket] += 1
                continue

            subset = "train_auto" if bucket == "auto" else "train_hard"
            rel_image = Path(subset) / "images" / item["language"] / f"{item['candidate_id']}.jpg"
            out_path = output_dir / rel_image
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists() or not args.skip_existing:
                image.crop(crop_box).save(out_path, quality=args.jpeg_quality)

            row = {
                "id": item["candidate_id"],
                "image": str(rel_image).replace("\\", "/"),
                "text": item["text"],
                "language": item["language"],
                "split": "train",
                "subset": subset,
                "source_family_id": item["source_family_id"],
                "source_id": item["source_id"],
                "source_image": item["source_image"],
                "full_image": str(image_path),
                "data_type": item["data_type"],
                "line_index": item["line_index"],
                "text_length": item["text_length"],
                "bbox": compact_json({"x": x1, "y": y1, "width": width, "height": height}),
                "bbox_aspect": round(aspect, 4),
                "bbox_source": "source_bbox" if item["language"] == "zh" else "render_layout_bbox",
                "quality_bucket": bucket,
                "crop_status": "ok",
                "render_font_size": item.get("render_font_size", ""),
                "render_fill": item.get("render_fill", ""),
                "render_stroke": item.get("render_stroke", ""),
                "render_shadow": item.get("render_shadow", ""),
                "render_scale_x": item.get("render_scale_x", ""),
            }
            rows.append(row)
            counts[f"{subset}_{item['language']}"] += 1

        if args.progress_every and image_idx % args.progress_every == 0:
            print(json.dumps({"image_progress": image_idx, "images": len(by_image), "rows": len(rows)}, ensure_ascii=False), flush=True)
    return rows, counts


def error_row(item, status, error):
    return {
        "id": item["candidate_id"],
        "image": "",
        "text": item["text"],
        "language": item["language"],
        "split": "train",
        "subset": "drop",
        "source_family_id": item["source_family_id"],
        "source_id": item["source_id"],
        "source_image": item["source_image"],
        "full_image": item["image_path"],
        "data_type": item["data_type"],
        "line_index": item["line_index"],
        "text_length": item["text_length"],
        "bbox": "",
        "bbox_aspect": "",
        "bbox_source": "source_bbox" if item["language"] == "zh" else "render_layout_bbox",
        "quality_bucket": status,
        "crop_status": status,
        "crop_error": error,
    }




def write_review_html(rows, output_dir, limit=300):
    shown = [r for r in rows if r.get("crop_status") == "ok"][:limit]
    cards = []
    for row in shown:
        direction = "rtl" if row.get("language") == "ug" else "ltr"
        image = row.get("image", "")
        text = row.get("text", "")
        cards.append(
            "<article class='card'>"
            f"<div class='image'><img src='{escape_html(image)}'></div>"
            "<div class='meta'>"
            f"<div><b>{escape_html(row.get('language',''))}</b> | {escape_html(row.get('id',''))}</div>"
            f"<div dir='{direction}' class='label'>{escape_html(text)}</div>"
            f"<pre>subset={escape_html(row.get('subset',''))}\nbbox_source={escape_html(row.get('bbox_source',''))}\nbbox={escape_html(row.get('bbox',''))}\naspect={escape_html(row.get('bbox_aspect',''))}\ntext_length={escape_html(row.get('text_length',''))}\nsource_family_id={escape_html(row.get('source_family_id',''))}</pre>"
            "</div></article>"
        )
    html = """<!doctype html><html><head><meta charset='utf-8'><title>Train Auto Render Box Check</title>
<style>
body{margin:20px;background:#f7f5ef;color:#17201a;font-family:Arial,'Microsoft YaHei',sans-serif}.card{display:grid;grid-template-columns:minmax(360px,1fr) minmax(320px,.8fr);gap:14px;background:#fff;border:1px solid #d8ddd7;border-radius:6px;margin:0 0 14px;padding:12px}.image{display:flex;align-items:center;justify-content:center;background:#ece9df;min-height:90px;border:1px solid #ded8cc}.image img{max-width:100%;max-height:150px}.label{font-size:22px;line-height:1.35;border:1px solid #d8ddd7;background:#fbfaf6;padding:8px;margin:8px 0}pre{white-space:pre-wrap;font-size:12px;color:#4e5a54}
</style></head><body><h1>Train Auto Render Box Check</h1><p>UG/KK crops use render_layout_bbox by default.</p>""" + "".join(cards) + "</body></html>"
    (output_dir / "review.html").write_text(html, encoding="utf-8")


def escape_html(value):
    return str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")


def train_review_row(row):
    image = row.get("image", "")
    crop_path = f"../{image}" if image else ""
    return {
        "candidate_id": row.get("id", ""),
        "language": row.get("language", ""),
        "split": "train",
        "source_id": row.get("source_id", ""),
        "source_image": row.get("source_image", ""),
        "image_path_resolved": row.get("full_image", ""),
        "crop_path_rel": crop_path,
        "overlay_path_rel": "",
        "text": row.get("text", ""),
        "text_length": row.get("text_length", ""),
        "bbox": row.get("bbox", ""),
        "bbox_aspect": row.get("bbox_aspect", ""),
        "review_status": "pass",
        "review_note": "",
    }


def write_train_review_pages(rows, output_dir, chunk_size):
    if chunk_size <= 0:
        return {}
    review_dir = output_dir / "train_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    ok_rows = [
        train_review_row(row)
        for row in rows
        if row.get("crop_status") == "ok" and row.get("image")
    ]
    ok_rows.sort(key=lambda row: (row["language"], row["source_id"], row["candidate_id"]))
    fields = list(ok_rows[0].keys()) if ok_rows else [
        "candidate_id", "language", "split", "source_id", "source_image",
        "image_path_resolved", "crop_path_rel", "overlay_path_rel", "text",
        "text_length", "bbox", "bbox_aspect", "review_status", "review_note",
    ]

    all_csv = review_dir / "train_candidates_all.csv"
    with all_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ok_rows)

    pages = []
    for chunk_idx, start in enumerate(range(0, len(ok_rows), chunk_size), 1):
        chunk = ok_rows[start:start + chunk_size]
        stem = f"train_candidates_{chunk_idx:04d}"
        chunk_csv = review_dir / f"{stem}.csv"
        chunk_html = review_dir / f"annotate_{chunk_idx:04d}.html"
        with chunk_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(chunk)
        chunk_html.write_text(build_html(read_rows(chunk_csv)), encoding="utf-8")
        pages.append({
            "index": chunk_idx,
            "csv": chunk_csv.name,
            "html": chunk_html.name,
            "rows": len(chunk),
            "start": start + 1,
            "end": start + len(chunk),
        })

    links = []
    for page in pages:
        links.append(
            "<li>"
            f"<a href='{page['html']}'>{page['html']}</a> "
            f"rows {page['start']}-{page['end']} ({page['rows']})"
            "</li>"
        )
    index_html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Train Review Pages</title>"
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:24px;line-height:1.6}"
        "a{color:#1f5f8b}</style></head><body>"
        "<h1>Train Review Pages</h1>"
        "<p>Open one page at a time. Export each reviewed CSV from the page after annotation.</p>"
        f"<p>Total review rows: {len(ok_rows)}. Chunk size: {chunk_size}.</p>"
        "<ol>" + "".join(links) + "</ol>"
        "</body></html>"
    )
    (review_dir / "index.html").write_text(index_html, encoding="utf-8")
    manifest = {
        "review_dir": str(review_dir),
        "all_csv": str(all_csv),
        "chunk_size": chunk_size,
        "rows": len(ok_rows),
        "pages": pages,
        "index_html": str(review_dir / "index.html"),
    }
    (review_dir / "train_review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
def summarize(rows, counts, args, output_dir):
    ok_rows = [r for r in rows if r.get("crop_status") == "ok"]
    train_auto = [r for r in ok_rows if r.get("subset") == "train_auto"]
    train_hard = [r for r in ok_rows if r.get("subset") == "train_hard"]
    summary = {
        "output_dir": str(output_dir),
        "total_rows": len(rows),
        "ok_rows": len(ok_rows),
        "train_auto_rows": len(train_auto),
        "train_hard_rows": len(train_hard),
        "counts": dict(counts),
        "train_auto_by_language": dict(Counter(r["language"] for r in train_auto)),
        "train_hard_by_language": dict(Counter(r["language"] for r in train_hard)),
        "drop_by_status": dict(Counter(r["crop_status"] for r in rows if r.get("subset") == "drop")),
        "filters": {
            "min_width": args.min_width,
            "min_height": args.min_height,
            "min_aspect": args.min_aspect,
            "max_aspect": args.max_aspect,
            "min_text_len": args.min_text_len,
            "max_text_len_zh": args.max_text_len_zh,
            "max_text_len_ugkk": args.max_text_len_ugkk,
            "bbox_field": args.bbox_field,
            "pad_x": args.pad_x,
            "pad_y": args.pad_y,
        },
        "notes": [
            "UG/KK train crops use the selected rendering-pipeline box field, default render_layout_bbox, not PP-OCR detections.",
            "Chinese train crops use the original Chinese OCR bbox on the source image.",
            "labels.txt contains only train_auto rows. train_hard rows are kept in metadata for optional robustness experiments.",
        ],
    }
    return summary


def default_paths():
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    workspace_root = script_path.parents[5]
    return {
        "workspace_root": workspace_root,
        "index_ug": workspace_root / "final_multilingual_meme_ocr_dataset" / "train_ug.json",
        "index_kk": workspace_root / "final_multilingual_meme_ocr_dataset" / "train_kk.json",
        "output_dir": svtr_root / "01_data_preparation" / "real_line_dataset",
    }


def main():
    defaults = default_paths()
    parser = argparse.ArgumentParser(description="Build train_auto real-line crops from known source/render boxes, without PP-OCR.")
    parser.add_argument("--workspace-root", type=Path, default=defaults["workspace_root"])
    parser.add_argument("--index-ug", type=Path, default=defaults["index_ug"])
    parser.add_argument("--index-kk", type=Path, default=defaults["index_kk"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--languages", nargs="+", default=list(LANGS), choices=list(LANGS))
    parser.add_argument("--max-per-lang", type=int, default=0, help="Debug/sample cap. 0 means all.")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--bbox-field", default="render_layout_bbox", choices=["render_layout_bbox", "render_bbox", "render_source_bbox", "auto"])
    parser.add_argument("--pad-x", type=int, default=10)
    parser.add_argument("--pad-y", type=int, default=6)
    parser.add_argument("--min-width", type=int, default=24)
    parser.add_argument("--min-height", type=int, default=18)
    parser.add_argument("--min-aspect", type=float, default=0.35)
    parser.add_argument("--max-aspect", type=float, default=18.0)
    parser.add_argument("--min-text-len", type=int, default=2)
    parser.add_argument("--max-text-len-zh", type=int, default=35)
    parser.add_argument("--max-text-len-ugkk", type=int, default=90)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--review-chunk-size", type=int, default=3000, help="Rows per train_review annotate page. 0 disables review HTML.")
    args = parser.parse_args()

    records_by_lang = {
        "ug": load_records(args.index_ug),
        "kk": load_records(args.index_kk),
    }
    candidates = collect_candidates(records_by_lang, args.workspace_root, set(args.languages), args)
    candidates = maybe_sample(candidates, args.max_per_lang, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, counts = crop_all(candidates, args.output_dir, args)
    write_outputs(rows, args.output_dir)
    write_review_html(rows, args.output_dir)
    review_manifest = write_train_review_pages(rows, args.output_dir, args.review_chunk_size)
    summary = summarize(rows, counts, args, args.output_dir)
    summary["train_review"] = review_manifest
    (args.output_dir / "train_auto_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()









