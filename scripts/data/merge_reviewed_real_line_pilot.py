import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


DEFAULT_REVIEWED = [
    "pilot_annotation_450.reviewed.csv",
    "ppocr_ugkk_supplement/ppocr_ugkk_candidates_ok.reviewed.csv",
]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_crop_path(real_lines_dir, row, source_csv):
    rel = row.get("crop_path_rel") or row.get("image") or ""
    if not rel:
        return None
    candidates = [source_csv.parent / rel, real_lines_dir / rel]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def add_image_stats(real_lines_dir, row, source_csv):
    p = resolve_crop_path(real_lines_dir, row, source_csv)
    row = dict(row)
    row["resolved_crop_path"] = str(p) if p else ""
    row["image_width"] = ""
    row["image_height"] = ""
    row["aspect_ratio"] = row.get("bbox_aspect", "")
    if p and p.exists() and Image is not None:
        with Image.open(p) as img:
            w, h = img.size
        row["image_width"] = w
        row["image_height"] = h
        row["aspect_ratio"] = round(w / max(1, h), 4)
    return row


def numeric(value):
    try:
        return float(value)
    except Exception:
        return None


def bucket_length(length):
    if length <= 15:
        return "short"
    if length <= 35:
        return "medium"
    return "long"


def summarize(rows):
    summary = {
        "total_pass": len(rows),
        "pass_by_language": dict(Counter(r.get("language", "") for r in rows)),
        "length_buckets_by_language": {},
        "image_stats_by_language": {},
    }
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r.get("language", "")].append(r)
    for lang, items in sorted(by_lang.items()):
        lengths = [int(float(r.get("text_length") or len(r.get("text", "")))) for r in items]
        aspects = [numeric(r.get("aspect_ratio")) for r in items]
        widths = [numeric(r.get("image_width")) for r in items]
        heights = [numeric(r.get("image_height")) for r in items]
        aspects = [x for x in aspects if x is not None]
        widths = [x for x in widths if x is not None]
        heights = [x for x in heights if x is not None]
        summary["length_buckets_by_language"][lang] = dict(Counter(bucket_length(x) for x in lengths))
        summary["image_stats_by_language"][lang] = {
            "samples": len(items),
            "text_length_min": min(lengths) if lengths else None,
            "text_length_mean": sum(lengths) / len(lengths) if lengths else None,
            "text_length_max": max(lengths) if lengths else None,
            "width_mean": sum(widths) / len(widths) if widths else None,
            "height_mean": sum(heights) / len(heights) if heights else None,
            "aspect_ratio_mean": sum(aspects) / len(aspects) if aspects else None,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Merge reviewed real-line pilot CSVs and summarize pass samples.")
    parser.add_argument("--real-lines-dir", type=Path, default=Path("/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition/01_data_preparation/real_lines"))
    parser.add_argument("--reviewed", nargs="*", default=DEFAULT_REVIEWED)
    parser.add_argument("--output-prefix", default="pilot_pass_merged")
    args = parser.parse_args()

    all_rows = []
    source_status = {}
    seen = set()
    for rel in args.reviewed:
        path = Path(rel)
        if not path.is_absolute():
            path = args.real_lines_dir / rel
        if not path.exists():
            source_status[str(path)] = "missing"
            continue
        rows = read_csv(path)
        source_status[str(path)] = {"rows": len(rows), "status_counts": dict(Counter(r.get("review_status", "") for r in rows))}
        for r in rows:
            if r.get("review_status") != "pass":
                continue
            cid = r.get("candidate_id") or f"{r.get('source_id')}:{r.get('language')}:{r.get('text')}"
            if cid in seen:
                continue
            seen.add(cid)
            r = add_image_stats(args.real_lines_dir, r, path)
            r["review_source_csv"] = str(path)
            all_rows.append(r)

    out_csv = args.real_lines_dir / f"{args.output_prefix}.csv"
    out_json = args.real_lines_dir / f"{args.output_prefix}_summary.json"
    fields = sorted(set().union(*(r.keys() for r in all_rows))) if all_rows else []
    preferred = [
        "candidate_id", "language", "split", "source_id", "text", "text_length",
        "crop_path_rel", "resolved_crop_path", "image_width", "image_height", "aspect_ratio",
        "review_status", "review_note", "review_source_csv",
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    write_csv(out_csv, all_rows, fields)
    summary = summarize(all_rows)
    summary["sources"] = source_status
    summary["outputs"] = {"pass_csv": str(out_csv), "summary_json": str(out_json)}
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
