import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError as exc:
    raise SystemExit("Missing Pillow. Install it with: pip install pillow") from exc


LANGS = ("ug", "kk")
MARKERS = (
    "Misogyny_Dataset_Project/",
    "Meme_Dataset_Project/",
    "final_multilingual_meme_ocr_dataset/",
)


def safe_id(value):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_") or "sample"


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def image_stem(record):
    for key in ("source_image_workspace", "source_image_abs", "source_image", "image"):
        value = record.get(key)
        if value:
            return Path(str(value)).stem
    return safe_id(record.get("sample_id") or "unknown")


def derive_source_id(record):
    dataset = record.get("source_dataset") or record.get("dataset_source") or "unknown_dataset"
    return f"{dataset}:{image_stem(record)}"


def record_split(record):
    return record.get("final_split") or record.get("split") or "unknown"


def resolve_path(value, workspace_root):
    if not value:
        return None
    p = Path(str(value))
    wrong_prefix = "/home/data_home/wudayu"
    if str(p).startswith(wrong_prefix):
        q = Path(str(p).replace(wrong_prefix, "/data_home/wudayu", 1))
        if q.exists():
            return q
    if p.is_absolute() and p.exists():
        return p
    q = workspace_root / str(value)
    if q.exists():
        return q
    normalized = str(value).replace("\\", "/")
    if ":/" in normalized:
        normalized = normalized.split(":/", 1)[1]
    for marker in MARKERS:
        if marker in normalized:
            q = workspace_root / normalized[normalized.index(marker):]
            if q.exists():
                return q
    name = Path(str(value)).name
    q = workspace_root / "Meme_Dataset_Project" / "images" / "meme" / name
    if q.exists():
        return q
    return None


def rendered_image_path(record, workspace_root):
    for key in ("rendered_image_abs", "rendered_image_workspace", "rendered_image"):
        p = resolve_path(record.get(key), workspace_root)
        if p is not None:
            return p
    return None


def line_text(line, language):
    return (
        line.get("render_text_normalized")
        or line.get(f"{language}_text")
        or line.get("text")
        or ""
    )


def line_bbox(line):
    return line.get("render_bbox") or line.get("render_layout_bbox") or line.get("bbox")


def box_xyxy(box):
    if isinstance(box, dict):
        x = float(box.get("x", 0))
        y = float(box.get("y", 0))
        w = float(box.get("width", 0))
        h = float(box.get("height", 0))
        return x, y, x + w, y + h
    if isinstance(box, (list, tuple)) and len(box) == 4 and not isinstance(box[0], (list, tuple)):
        x1, y1, x2, y2 = [float(v) for v in box]
        return x1, y1, x2, y2
    if isinstance(box, (list, tuple)) and len(box) >= 4 and isinstance(box[0], (list, tuple)):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        return min(xs), min(ys), max(xs), max(ys)
    return 0.0, 0.0, 0.0, 0.0


def polygon_bbox(poly):
    return box_xyxy(poly)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


def center_distance(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2
    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2
    return math.hypot(acx - bcx, acy - bcy)


def diagonal(box):
    x1, y1, x2, y2 = box
    return math.hypot(max(1.0, x2 - x1), max(1.0, y2 - y1))


def language_padding(language, pad_x, pad_y):
    if language == "ug":
        return max(pad_x, 26), max(pad_y, 14)
    if language == "kk":
        return max(pad_x, 24), max(pad_y, 12)
    return pad_x, pad_y


def clamp_box(box, width, height, pad_x, pad_y):
    x1, y1, x2, y2 = box
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(width, int(round(x2 + pad_x)))
    y2 = min(height, int(round(y2 + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def load_reviewed_ids(path):
    ids = set()
    if not path or not path.exists():
        return ids
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = row.get("candidate_id")
            if cid:
                ids.add(cid)
    return ids


def load_records(paths):
    records = []
    for path in paths:
        if path is None:
            continue
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    return records


def collect_line_candidates(records, workspace_root, languages, split, reviewed_ids):
    candidates = []
    for record in records:
        language = record.get("language") or record.get("render_lang")
        if language not in languages or record_split(record) != split:
            continue
        image_path = rendered_image_path(record, workspace_root)
        if image_path is None:
            continue
        source_id = derive_source_id(record)
        for line in record.get("texts") or []:
            text = line_text(line, language).strip()
            bbox = line_bbox(line)
            if not text or not bbox:
                continue
            line_index = int(line.get("line_index", 0))
            candidate_id = f"{safe_id(source_id)}_{language}_L{line_index + 1:03d}"
            if candidate_id in reviewed_ids:
                continue
            candidates.append({
                "candidate_id": candidate_id,
                "language": language,
                "split": record_split(record),
                "source_id": source_id,
                "source_image": record.get("source_image_workspace") or record.get("source_image_abs") or record.get("source_image") or "",
                "image_path": str(image_path),
                "text": text,
                "text_length": len(text),
                "line_index": line_index,
                "gt_render_bbox": bbox,
                "gt_box_xyxy": box_xyxy(bbox),
                "sample_id": record.get("sample_id") or "",
            })
    return candidates


def sample_candidates(candidates, target_per_lang, seed):
    rng = random.Random(seed)
    by_lang = defaultdict(list)
    for item in candidates:
        by_lang[item["language"]].append(item)
    selected = []
    for language in LANGS:
        items = by_lang.get(language, [])
        rng.shuffle(items)
        selected.extend(items[:target_per_lang])
    selected.sort(key=lambda x: (x["language"], x["source_id"], x["line_index"]))
    return selected, {lang: len(by_lang.get(lang, [])) for lang in LANGS}


def build_detector(args):
    from paddleocr import PaddleOCR
    kwargs_new = {
        "lang": args.paddle_lang,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    try:
        return PaddleOCR(**kwargs_new)
    except TypeError:
        return PaddleOCR(lang=args.paddle_lang, use_angle_cls=False, show_log=False)


def normalize_old_ocr_result(result):
    boxes = []

    def is_point(value):
        return isinstance(value, (list, tuple)) and len(value) >= 2 and all(isinstance(value[i], (int, float)) for i in (0, 1))

    def is_polygon(value):
        return isinstance(value, (list, tuple)) and len(value) >= 4 and all(is_point(p) for p in value[:4])

    def add_polygon(poly, score=""):
        boxes.append({"polygon": [[float(p[0]), float(p[1])] for p in poly[:4]], "score": score})

    def walk(value):
        if value is None:
            return
        if is_polygon(value):
            add_polygon(value)
            return
        if isinstance(value, (list, tuple)) and value:
            if is_polygon(value[0]):
                score = ""
                if len(value) > 1 and isinstance(value[1], (int, float)):
                    score = value[1]
                add_polygon(value[0], score)
                return
            for child in value:
                walk(child)

    walk(result)
    return boxes


def normalize_predict_result(result):
    boxes = []
    for res in result or []:
        data = getattr(res, "json", None) or res
        if isinstance(data, dict) and "res" in data:
            data = data["res"]
        candidates = data.get("dt_polys") or data.get("rec_polys") or data.get("dt_boxes") or [] if isinstance(data, dict) else []
        scores = data.get("dt_scores") or data.get("rec_scores") or [] if isinstance(data, dict) else []
        for idx, poly in enumerate(candidates):
            score = scores[idx] if idx < len(scores) else ""
            if poly:
                boxes.append({"polygon": [[float(p[0]), float(p[1])] for p in poly], "score": score})
    return boxes


def detect_boxes(detector, image_path):
    if hasattr(detector, "ocr"):
        try:
            return normalize_old_ocr_result(detector.ocr(str(image_path), det=True, rec=False, cls=False))
        except Exception:
            pass
    if hasattr(detector, "predict"):
        return normalize_predict_result(detector.predict(str(image_path)))
    raise RuntimeError("Unsupported PaddleOCR detector object: no ocr() or predict().")


def choose_match(gt_box, detections, min_iou, max_center_ratio):
    best = None
    best_key = None
    for det in detections:
        det_box = polygon_bbox(det["polygon"])
        score_iou = iou(gt_box, det_box)
        dist = center_distance(gt_box, det_box)
        ratio = dist / max(1.0, diagonal(gt_box))
        key = (score_iou, -ratio)
        if best is None or key > best_key:
            best = {**det, "det_box": det_box, "match_iou": score_iou, "center_distance": dist, "center_ratio": ratio}
            best_key = key
    if best and (best["match_iou"] >= min_iou or best["center_ratio"] <= max_center_ratio):
        return best
    return None


def resize_for_overlay(image, max_side):
    w, h = image.size
    if max(w, h) <= max_side:
        return image.copy(), 1.0
    scale = max_side / max(w, h)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return image.resize(size), scale


def draw_overlay(image, gt_box, det_box, out_path, max_side):
    overlay, scale = resize_for_overlay(image, max_side)
    draw = ImageDraw.Draw(overlay)
    def scaled(box):
        return tuple(int(round(v * scale)) for v in box)
    draw.rectangle(scaled(gt_box), outline=(255, 180, 0), width=max(2, int(3 * scale)))
    draw.rectangle(scaled(det_box), outline=(0, 220, 80), width=max(2, int(3 * scale)))
    overlay.save(out_path, quality=92)


def crop_and_write(selected, detector, output_dir, args):
    crops_dir = output_dir / "crops"
    overlays_dir = output_dir / "overlays"
    crops_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    by_image = defaultdict(list)
    for item in selected:
        by_image[item["image_path"]].append(item)

    rows = []
    status_counts = Counter()
    for image_idx, (image_path_str, items) in enumerate(sorted(by_image.items()), 1):
        image_path = Path(image_path_str)
        try:
            detections = detect_boxes(detector, image_path)
        except Exception as exc:
            for item in items:
                row = make_error_row(item, output_dir, "detect_error", repr(exc))
                rows.append(row)
                status_counts[row["crop_status"]] += 1
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            for item in items:
                match = choose_match(item["gt_box_xyxy"], detections, args.min_iou, args.max_center_ratio)
                if match is None:
                    row = make_error_row(item, output_dir, "no_matched_detection", f"detections={len(detections)}")
                    rows.append(row)
                    status_counts[row["crop_status"]] += 1
                    continue
                pad_x, pad_y = language_padding(item["language"], args.pad_x, args.pad_y)
                crop_box = clamp_box(match["det_box"], width, height, pad_x, pad_y)
                if crop_box is None:
                    row = make_error_row(item, output_dir, "invalid_crop_box", compact_json(match["det_box"]))
                    rows.append(row)
                    status_counts[row["crop_status"]] += 1
                    continue

                safe = safe_id(item["candidate_id"])
                lang = item["language"]
                crop_path = crops_dir / lang / f"{safe}.jpg"
                overlay_path = overlays_dir / lang / f"{safe}_overlay.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                crop = image.crop(crop_box)
                crop.save(crop_path, quality=95)
                draw_overlay(image, item["gt_box_xyxy"], match["det_box"], overlay_path, args.overlay_max_side)

                x1, y1, x2, y2 = crop_box
                row = {
                    "candidate_id": item["candidate_id"],
                    "language": lang,
                    "split": item["split"],
                    "source_id": item["source_id"],
                    "source_image": item["source_image"],
                    "image_path_resolved": str(image_path),
                    "crop_path_rel": str(crop_path.relative_to(output_dir)).replace("\\", "/"),
                    "overlay_path_rel": str(overlay_path.relative_to(output_dir)).replace("\\", "/"),
                    "text": item["text"],
                    "text_length": item["text_length"],
                    "bbox": compact_json({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}),
                    "bbox_aspect": round((x2 - x1) / max(1, y2 - y1), 4),
                    "review_status": "pending",
                    "review_note": "",
                    "gt_render_bbox": compact_json(item["gt_render_bbox"]),
                    "det_bbox": compact_json(match["det_box"]),
                    "det_polygon": compact_json(match["polygon"]),
                    "match_iou": round(match["match_iou"], 4),
                    "center_ratio": round(match["center_ratio"], 4),
                    "det_score": match.get("score", ""),
                    "crop_status": "ok",
                    "crop_error": "",
                    "crop_size": f"{crop.width}x{crop.height}",
                }
                rows.append(row)
                status_counts["ok"] += 1
        print(json.dumps({"image_progress": image_idx, "images": len(by_image), "path": image_path_str, "detections": len(detections)}, ensure_ascii=False), flush=True)
    return rows, status_counts


def make_error_row(item, output_dir, status, error):
    return {
        "candidate_id": item["candidate_id"],
        "language": item["language"],
        "split": item["split"],
        "source_id": item["source_id"],
        "source_image": item["source_image"],
        "image_path_resolved": item["image_path"],
        "crop_path_rel": "",
        "overlay_path_rel": "",
        "text": item["text"],
        "text_length": item["text_length"],
        "bbox": "",
        "bbox_aspect": "",
        "review_status": "drop",
        "review_note": status,
        "gt_render_bbox": compact_json(item["gt_render_bbox"]),
        "det_bbox": "",
        "det_polygon": "",
        "match_iou": "",
        "center_ratio": "",
        "det_score": "",
        "crop_status": status,
        "crop_error": error,
        "crop_size": "",
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_paths():
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    workspace_root = script_path.parents[5]
    return {
        "workspace_root": workspace_root,
        "index_ug": workspace_root / "final_multilingual_meme_ocr_dataset" / "train_ug.json",
        "index_kk": workspace_root / "final_multilingual_meme_ocr_dataset" / "train_kk.json",
        "reviewed_csv": svtr_root / "01_data_preparation" / "real_lines" / "pilot_annotation_450.reviewed.csv",
        "output_dir": svtr_root / "01_data_preparation" / "real_lines" / "ppocr_ugkk_supplement",
    }


def main():
    defaults = default_paths()
    parser = argparse.ArgumentParser(description="Use PP-OCR detection on rendered UG/KK images to build real-line crop candidates.")
    parser.add_argument("--workspace-root", type=Path, default=defaults["workspace_root"])
    parser.add_argument("--index-ug", type=Path, default=defaults["index_ug"])
    parser.add_argument("--index-kk", type=Path, default=defaults["index_kk"])
    parser.add_argument("--reviewed-csv", type=Path, default=defaults["reviewed_csv"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--target-per-lang", type=int, default=300)
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--paddle-lang", default="ch")
    parser.add_argument("--min-iou", type=float, default=0.05)
    parser.add_argument("--max-center-ratio", type=float, default=0.85)
    parser.add_argument("--pad-x", type=int, default=12)
    parser.add_argument("--pad-y", type=int, default=8)
    parser.add_argument("--overlay-max-side", type=int, default=900)
    args = parser.parse_args()

    reviewed_ids = load_reviewed_ids(args.reviewed_csv)
    records = load_records([args.index_ug, args.index_kk])
    candidates = collect_line_candidates(records, args.workspace_root, LANGS, args.split, reviewed_ids)
    selected, candidate_counts = sample_candidates(candidates, args.target_per_lang, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_csv = args.output_dir / "selected_candidates_before_detection.csv"
    write_csv(selected_csv, [{k: (compact_json(v) if isinstance(v, (dict, list, tuple)) else v) for k, v in item.items() if k != "gt_box_xyxy"} for item in selected])

    detector = build_detector(args)
    rows, status_counts = crop_and_write(selected, detector, args.output_dir, args)
    candidate_csv = args.output_dir / "ppocr_ugkk_candidates.csv"
    ok_candidate_csv = args.output_dir / "ppocr_ugkk_candidates_ok.csv"
    write_csv(candidate_csv, rows)
    write_csv(ok_candidate_csv, [row for row in rows if row.get("crop_status") == "ok"])

    summary = {
        "workspace_root": str(args.workspace_root),
        "reviewed_csv": str(args.reviewed_csv),
        "output_dir": str(args.output_dir),
        "target_per_lang": args.target_per_lang,
        "seed": args.seed,
        "candidate_counts_before_sampling": candidate_counts,
        "selected_counts": dict(Counter(item["language"] for item in selected)),
        "crop_status_counts": dict(status_counts),
        "outputs": {
            "selected_csv": str(selected_csv),
            "candidate_csv": str(candidate_csv),
            "ok_candidate_csv": str(ok_candidate_csv),
        },
        "notes": [
            "Yellow overlay box is the original render_bbox; green overlay box is the PP-OCR detected box.",
            "Ground-truth text is still taken from the dataset index, not from PP-OCR recognition.",
        ],
    }
    (args.output_dir / "ppocr_ugkk_candidates_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



