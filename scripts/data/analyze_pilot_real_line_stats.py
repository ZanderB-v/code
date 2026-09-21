import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_bbox(value):
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def to_float(value):
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def describe(values, digits=4):
    values = [v for v in values if v is not None]
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "std": None,
        }
    return {
        "count": len(values),
        "min": round(min(values), digits),
        "p25": round(percentile(values, 0.25), digits),
        "median": round(percentile(values, 0.50), digits),
        "mean": round(sum(values) / len(values), digits),
        "p75": round(percentile(values, 0.75), digits),
        "p90": round(percentile(values, 0.90), digits),
        "p95": round(percentile(values, 0.95), digits),
        "max": round(max(values), digits),
        "std": round(statistics.pstdev(values), digits) if len(values) > 1 else 0,
    }


def length_bucket(length):
    if length <= 15:
        return "short"
    if length <= 35:
        return "medium"
    return "long"


def resolve_crop_path(real_lines_dir, row):
    rel = row.get("crop_path_rel") or row.get("image") or ""
    rel = rel.replace("\\", "/")
    candidates = []

    merge_source = row.get("merge_source_csv") or row.get("review_source_csv") or ""
    if merge_source:
        candidates.append(Path(merge_source).parent / rel)

    candidates.extend(
        [
            real_lines_dir / rel,
            real_lines_dir / "pilot_review" / rel,
            real_lines_dir / "ppocr_ugkk_supplement" / rel,
        ]
    )

    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else None


def image_size(path):
    if Image is None or path is None or not path.exists():
        return None
    with Image.open(path) as img:
        return img.size


def enrich_row(real_lines_dir, row):
    row = dict(row)
    text = row.get("text") or ""
    length = int(to_float(row.get("text_length")) or len(text))
    bbox = parse_bbox(row.get("bbox", ""))
    crop_path = resolve_crop_path(real_lines_dir, row)
    size = image_size(crop_path)

    if size:
        width, height = size
        size_source = "image"
    else:
        width = int(to_float(bbox.get("width")) or 0)
        height = int(to_float(bbox.get("height")) or 0)
        size_source = "bbox_fallback"

    aspect = width / height if width and height else to_float(row.get("bbox_aspect"))
    row.update(
        {
            "resolved_crop_path": str(crop_path) if crop_path else "",
            "size_source": size_source,
            "image_width": width,
            "image_height": height,
            "aspect_ratio": round(aspect, 4) if aspect else "",
            "computed_text_length": length,
            "length_bucket": length_bucket(length),
        }
    )
    return row


def summarize(rows, long_threshold):
    summary = {
        "samples": len(rows),
        "counts_by_language": dict(Counter(r["language"] for r in rows)),
        "size_source_counts": dict(Counter(r["size_source"] for r in rows)),
        "length_bucket_thresholds": {"short_max": 15, "medium_max": 35, "long_min": 36},
        "long_text_threshold": long_threshold,
        "by_language": {},
        "all": {},
    }

    groups = {"all": rows}
    for lang in sorted(set(r["language"] for r in rows)):
        groups[lang] = [r for r in rows if r["language"] == lang]

    for key, items in groups.items():
        lengths = [r["computed_text_length"] for r in items]
        widths = [to_float(r["image_width"]) for r in items]
        heights = [to_float(r["image_height"]) for r in items]
        aspects = [to_float(r["aspect_ratio"]) for r in items]
        buckets = Counter(r["length_bucket"] for r in items)
        long_count = sum(1 for x in lengths if x >= long_threshold)
        payload = {
            "samples": len(items),
            "length": describe(lengths),
            "width": describe(widths),
            "height": describe(heights),
            "aspect_ratio": describe(aspects),
            "length_buckets": dict(buckets),
            "length_bucket_ratio": {k: round(v / len(items), 4) for k, v in sorted(buckets.items())} if items else {},
            "long_text_count": long_count,
            "long_text_ratio": round(long_count / len(items), 4) if items else None,
        }
        if key == "all":
            summary["all"] = payload
        else:
            summary["by_language"][key] = payload
    return summary


def recommend(summary):
    all_stats = summary["all"]
    lang_stats = summary["by_language"]
    aspect_p95 = all_stats["aspect_ratio"]["p95"] or 12
    aspect_max = all_stats["aspect_ratio"]["max"] or aspect_p95
    height_p75 = all_stats["height"]["p75"] or 64
    height_p95 = all_stats["height"]["p95"] or 96

    long_ratios = {
        lang: stats["long_text_ratio"]
        for lang, stats in lang_stats.items()
        if stats["long_text_ratio"] is not None
    }

    return {
        "synthetic_canvas": {
            "render_height_px": [64, 96],
            "training_height_px": [32, 48],
            "reason": (
                "Use 64/96 px render height to cover the pilot line-height upper range, "
                "then resize to 32/48 px during recognition training."
            ),
        },
        "dynamic_width": {
            "keep_dynamic_width": True,
            "target_aspect_ratio_range": [
                round(max(1.0, all_stats["aspect_ratio"]["p25"] or 1.0), 2),
                round(aspect_p95, 2),
            ],
            "soft_max_aspect_ratio": round(aspect_p95, 2),
            "hard_max_aspect_ratio": round(min(max(aspect_max, aspect_p95), aspect_p95 * 1.25), 2),
            "svtrv2_note": "Do not force fixed width at generation time; cap or resample extreme-wide lines before training.",
        },
        "text_length_sampling": {
            "bucket_definition": {
                "short": "length <= 15",
                "medium": "16 <= length <= 35",
                "long": "length >= 36",
            },
            "suggested_ratio": all_stats["length_bucket_ratio"],
            "long_text_ratio_by_language": long_ratios,
        },
        "language_direction": {
            "zh": {"dir": "ltr", "lang": "zh"},
            "ug": {"dir": "rtl", "lang": "ug", "note": "Keep Unicode logical order; do not reverse strings."},
            "kk": {"dir": "ltr", "lang": "kk"},
        },
        "crop_margin": {
            "horizontal_px": [6, 24],
            "vertical_px": [4, 16],
            "note": "Sample small margins like the real crops; allow slight edge noise but avoid complete adjacent lines.",
        },
        "style_balance_start": {
            "high_contrast": 0.60,
            "medium_contrast": 0.30,
            "low_contrast": 0.10,
            "single_stroke": 0.60,
            "shadow": 0.40,
            "double_stroke": 0.15,
            "glow": 0.10,
            "no_effect": 0.20,
            "slight_rotation": 0.20,
            "tilt_or_perspective": 0.10,
        },
        "observed_height_reference": {
            "pilot_height_p75": height_p75,
            "pilot_height_p95": height_p95,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze balanced pilot real-line OCR crops.")
    parser.add_argument(
        "--real-lines-dir",
        type=Path,
        default=Path("experiments/multilingual_meme_ocr/svtrv2_line_recognition/01_data_preparation/real_lines"),
    )
    parser.add_argument("--input", default="pilot_pass_balanced_393.csv")
    parser.add_argument("--output-prefix", default="pilot_real_line_stats")
    parser.add_argument("--long-threshold", type=int, default=36)
    args = parser.parse_args()

    real_lines_dir = args.real_lines_dir
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = real_lines_dir / input_path

    rows = [enrich_row(real_lines_dir, row) for row in read_csv(input_path)]
    summary = summarize(rows, args.long_threshold)
    summary["input"] = str(input_path)
    summary["outputs"] = {
        "enriched_csv": str(real_lines_dir / f"{args.output_prefix}_enriched.csv"),
        "summary_json": str(real_lines_dir / f"{args.output_prefix}.json"),
        "by_language_csv": str(real_lines_dir / f"{args.output_prefix}_by_language.csv"),
        "synthetic_params_json": str(real_lines_dir / "synthetic_params_from_pilot.json"),
    }
    params = recommend(summary)

    enriched_csv = Path(summary["outputs"]["enriched_csv"])
    summary_json = Path(summary["outputs"]["summary_json"])
    by_lang_csv = Path(summary["outputs"]["by_language_csv"])
    params_json = Path(summary["outputs"]["synthetic_params_json"])

    fields = list(rows[0].keys()) if rows else []
    write_csv(enriched_csv, rows, fields)

    flat_rows = []
    for lang, stats in summary["by_language"].items():
        row = {"language": lang, "samples": stats["samples"]}
        for section in ["length", "width", "height", "aspect_ratio"]:
            for k, v in stats[section].items():
                row[f"{section}_{k}"] = v
        for bucket in ["short", "medium", "long"]:
            row[f"bucket_{bucket}"] = stats["length_buckets"].get(bucket, 0)
            row[f"bucket_{bucket}_ratio"] = stats["length_bucket_ratio"].get(bucket, 0)
        row["long_text_count"] = stats["long_text_count"]
        row["long_text_ratio"] = stats["long_text_ratio"]
        flat_rows.append(row)
    by_lang_fields = list(flat_rows[0].keys()) if flat_rows else []
    write_csv(by_lang_csv, flat_rows, by_lang_fields)

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    params_json.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "synthetic_params": params}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
