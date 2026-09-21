import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from build_pilot_annotation_html import build_html, read_rows
from build_ppocr_ugkk_line_candidates import (
    box_xyxy,
    build_detector,
    choose_match,
    clamp_box,
    compact_json,
    derive_source_id,
    detect_boxes,
    draw_overlay,
    image_stem,
    language_padding,
    line_bbox,
    line_text,
    rendered_image_path,
    resolve_path,
    safe_id,
)


LANGS = ("zh", "ug", "kk")

def safe_file_id(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_") or "sample"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_records(paths):
    records = []
    for path in paths:
        if path and path.exists():
            records.extend(json.loads(path.read_text(encoding="utf-8")))
    return records


def record_split(record):
    return record.get("final_split") or record.get("split") or "unknown"


def source_family_id(record):
    return derive_source_id(record)


def source_image_path(record, workspace_root):
    for key in ("source_image_abs", "source_image_workspace", "source_image"):
        path = resolve_path(record.get(key), workspace_root)
        if path is not None:
            return path
    return None


def zh_text(line):
    return (
        line.get("zh_text")
        or line.get("source_text")
        or line.get("text")
        or ""
    ).strip()


def zh_bbox(line):
    return line.get("bbox") or line.get("polygon") or line.get("render_bbox")


def candidate_id(family_id, language, line_index):
    return f"{safe_file_id(family_id)}_{language}_L{line_index + 1:03d}"


def collect_candidates(records_by_lang, workspace_root, split, languages):
    candidates = []
    seen = set()

    # Use UG records as the aligned source for Chinese, because they retain the
    # original Chinese OCR boxes and source-image path.
    zh_source_records = records_by_lang.get("ug") or records_by_lang.get("kk") or []

    if "zh" in languages:
        for record in zh_source_records:
            if record_split(record) != split:
                continue
            image_path = source_image_path(record, workspace_root)
            if image_path is None:
                continue
            family = source_family_id(record)
            for line in record.get("texts") or []:
                text = zh_text(line)
                bbox = zh_bbox(line)
                if not text or not bbox:
                    continue
                line_index = int(line.get("line_index", 0))
                cid = candidate_id(family, "zh", line_index)
                if cid in seen:
                    continue
                seen.add(cid)
                candidates.append(
                    {
                        "candidate_id": cid,
                        "language": "zh",
                        "split": split,
                        "source_family_id": family,
                        "source_id": family,
                        "source_image": record.get("source_image_workspace") or record.get("source_image_abs") or record.get("source_image") or "",
                        "image_path": str(image_path),
                        "text": text,
                        "text_length": len(text),
                        "line_index": line_index,
                        "gt_bbox": bbox,
                        "gt_box_xyxy": box_xyxy(bbox),
                        "sample_id": record.get("sample_id") or "",
                        "data_type": "natural_chinese_meme_line",
                    }
                )

    for language in ("ug", "kk"):
        if language not in languages:
            continue
        for record in records_by_lang.get(language, []):
            if record_split(record) != split:
                continue
            image_path = rendered_image_path(record, workspace_root)
            if image_path is None:
                continue
            family = source_family_id(record)
            for line in record.get("texts") or []:
                text = line_text(line, language).strip()
                bbox = line_bbox(line)
                if not text or not bbox:
                    continue
                line_index = int(line.get("line_index", 0))
                cid = candidate_id(family, language, line_index)
                if cid in seen:
                    continue
                seen.add(cid)
                candidates.append(
                    {
                        "candidate_id": cid,
                        "language": language,
                        "split": split,
                        "source_family_id": family,
                        "source_id": family,
                        "source_image": record.get("source_image_workspace") or record.get("source_image_abs") or record.get("source_image") or "",
                        "image_path": str(image_path),
                        "text": text,
                        "text_length": len(text),
                        "line_index": line_index,
                        "gt_bbox": bbox,
                        "gt_box_xyxy": box_xyxy(bbox),
                        "sample_id": record.get("sample_id") or "",
                        "data_type": f"in_domain_rendered_{language}_meme_line",
                    }
                )
    return candidates


def select_source_families(candidates, split, source_family_count, seed):
    families = sorted({item["source_family_id"] for item in candidates})
    if split == "dev" or source_family_count <= 0 or source_family_count >= len(families):
        return set(families)
    rng = random.Random(seed)
    chosen = list(families)
    rng.shuffle(chosen)
    return set(chosen[:source_family_count])


def cap_rows_per_language(candidates, max_rows_per_lang, seed):
    if max_rows_per_lang <= 0:
        return candidates
    rng = random.Random(seed)
    by_lang = defaultdict(list)
    for item in candidates:
        by_lang[item["language"]].append(item)
    selected = []
    for language in LANGS:
        items = by_lang.get(language, [])
        if len(items) > max_rows_per_lang:
            rng.shuffle(items)
            items = items[:max_rows_per_lang]
        selected.extend(items)
    return selected


def crop_with_box(image, box, language, pad_x, pad_y):
    px, py = language_padding(language, pad_x, pad_y)
    return clamp_box(box, image.width, image.height, px, py)


def make_error_row(item, status, error):
    return {
        "candidate_id": item["candidate_id"],
        "language": item["language"],
        "split": item["split"],
        "source_family_id": item["source_family_id"],
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
        "data_type": item["data_type"],
        "line_index": item["line_index"],
        "gt_bbox": compact_json(item["gt_bbox"]),
        "det_bbox": "",
        "match_iou": "",
        "center_ratio": "",
        "crop_status": status,
        "crop_error": error,
        "crop_size": "",
    }


def crop_candidates(candidates, output_dir, detector, args):
    crops_dir = output_dir / "crops"
    overlays_dir = output_dir / "overlays"
    crops_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    by_image = defaultdict(list)
    for item in candidates:
        by_image[item["image_path"]].append(item)

    rows = []
    status_counts = Counter()
    detection_cache = {}
    total_images = len(by_image)

    for image_idx, (image_path_str, items) in enumerate(sorted(by_image.items()), 1):
        image_path = Path(image_path_str)
        try:
            with Image.open(image_path) as src:
                image = src.convert("RGB")
        except Exception as exc:
            for item in items:
                row = make_error_row(item, "image_open_error", repr(exc))
                rows.append(row)
                status_counts[row["crop_status"]] += 1
            continue

        needs_detection = (
            args.detector == "ppocr"
            and detector is not None
            and any(item["language"] in ("ug", "kk") for item in items)
        )
        detections = []
        if needs_detection:
            try:
                detections = detection_cache.get(image_path_str)
                if detections is None:
                    detections = detect_boxes(detector, image_path)
                    detection_cache[image_path_str] = detections
            except Exception as exc:
                for item in items:
                    if item["language"] in ("ug", "kk"):
                        row = make_error_row(item, "detect_error", repr(exc))
                        rows.append(row)
                        status_counts[row["crop_status"]] += 1
                detections = []

        for item in items:
            if args.detector == "ppocr" and item["language"] in ("ug", "kk") and detector is not None:
                match = choose_match(item["gt_box_xyxy"], detections, args.min_iou, args.max_center_ratio)
                if match is None:
                    row = make_error_row(item, "no_matched_detection", f"detections={len(detections)}")
                    rows.append(row)
                    status_counts[row["crop_status"]] += 1
                    continue
                det_box = match["det_box"]
                match_iou = round(match["match_iou"], 4)
                center_ratio = round(match["center_ratio"], 4)
            else:
                det_box = item["gt_box_xyxy"]
                match_iou = ""
                center_ratio = ""

            crop_box = crop_with_box(image, det_box, item["language"], args.pad_x, args.pad_y)
            if crop_box is None:
                row = make_error_row(item, "invalid_crop_box", compact_json(det_box))
                rows.append(row)
                status_counts[row["crop_status"]] += 1
                continue

            safe = safe_file_id(item["candidate_id"])
            language = item["language"]
            crop_path = crops_dir / language / f"{safe}.jpg"
            overlay_path = overlays_dir / language / f"{safe}_overlay.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)

            crop = image.crop(crop_box)
            crop.save(crop_path, quality=95)
            draw_overlay(image, item["gt_box_xyxy"], det_box, overlay_path, args.overlay_max_side)

            x1, y1, x2, y2 = crop_box
            row = {
                "candidate_id": item["candidate_id"],
                "language": language,
                "split": item["split"],
                "source_family_id": item["source_family_id"],
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
                "data_type": item["data_type"],
                "line_index": item["line_index"],
                "gt_bbox": compact_json(item["gt_bbox"]),
                "det_bbox": compact_json(det_box),
                "match_iou": match_iou,
                "center_ratio": center_ratio,
                "crop_status": "ok",
                "crop_error": "",
                "crop_size": f"{crop.width}x{crop.height}",
            }
            rows.append(row)
            status_counts["ok"] += 1

        print(
            json.dumps(
                {
                    "image_progress": image_idx,
                    "images": total_images,
                    "path": image_path_str,
                    "rows": len(items),
                    "detections": len(detections),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    return rows, status_counts


def build_annotation_html(input_csv, output_html):
    rows = read_rows(input_csv)
    output_html.write_text(build_html(rows), encoding="utf-8")


def default_paths():
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    workspace_root = script_path.parents[5]
    return {
        "workspace_root": workspace_root,
        "dataset_root": svtr_root / "01_data_preparation" / "real_line_dataset",
    }


def main():
    defaults = default_paths()
    parser = argparse.ArgumentParser(description="Build reviewed dev/test line-crop candidates and annotation pages.")
    parser.add_argument("--workspace-root", type=Path, default=defaults["workspace_root"])
    parser.add_argument("--dataset-root", type=Path, default=defaults["dataset_root"])
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--output-subset", choices=["dev_reviewed", "test_reviewed"], default="")
    parser.add_argument("--languages", nargs="+", default=list(LANGS), choices=list(LANGS))
    parser.add_argument("--source-family-count", type=int, default=0, help="Use all families when <=0. Recommended 400-600 for test.")
    parser.add_argument("--max-rows-per-lang", type=int, default=0, help="Optional row cap after source-family sampling.")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--detector", choices=["bbox", "ppocr"], default="bbox")
    parser.add_argument("--paddle-lang", default="ch")
    parser.add_argument("--min-iou", type=float, default=0.05)
    parser.add_argument("--max-center-ratio", type=float, default=0.85)
    parser.add_argument("--pad-x", type=int, default=12)
    parser.add_argument("--pad-y", type=int, default=8)
    parser.add_argument("--overlay-max-side", type=int, default=900)
    args = parser.parse_args()

    subset = args.output_subset or ("dev_reviewed" if args.split == "dev" else "test_reviewed")
    output_dir = args.dataset_root / subset
    output_dir.mkdir(parents=True, exist_ok=True)

    index_root = args.workspace_root / "final_multilingual_meme_ocr_dataset"
    records_by_lang = {
        "ug": load_records([index_root / f"{args.split}_ug.json"]),
        "kk": load_records([index_root / f"{args.split}_kk.json"]),
    }
    candidates = collect_candidates(records_by_lang, args.workspace_root, args.split, set(args.languages))
    candidate_counts = Counter(item["language"] for item in candidates)

    chosen_families = select_source_families(candidates, args.split, args.source_family_count, args.seed)
    selected = [item for item in candidates if item["source_family_id"] in chosen_families]
    selected = cap_rows_per_language(selected, args.max_rows_per_lang, args.seed)
    selected.sort(key=lambda x: (x["language"], x["source_family_id"], x["line_index"]))

    selected_csv = output_dir / "selected_candidates_before_crop.csv"
    write_csv(
        selected_csv,
        [
            {
                k: compact_json(v) if isinstance(v, (dict, list, tuple)) else v
                for k, v in item.items()
                if k != "gt_box_xyxy"
            }
            for item in selected
        ],
    )

    detector = build_detector(args) if args.detector == "ppocr" else None
    rows, status_counts = crop_candidates(selected, output_dir, detector, args)
    candidates_csv = output_dir / "candidates.csv"
    ok_csv = output_dir / "candidates_ok.csv"
    write_csv(candidates_csv, rows)
    write_csv(ok_csv, [row for row in rows if row.get("crop_status") == "ok"])

    annotate_html = output_dir / "annotate.html"
    build_annotation_html(ok_csv, annotate_html)

    summary = {
        "workspace_root": str(args.workspace_root),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "subset": subset,
        "languages": args.languages,
        "detector": args.detector,
        "seed": args.seed,
        "source_family_count_requested": args.source_family_count,
        "source_family_count_selected": len(chosen_families),
        "candidate_counts_before_sampling": dict(candidate_counts),
        "selected_counts": dict(Counter(item["language"] for item in selected)),
        "crop_status_counts": dict(status_counts),
        "outputs": {
            "selected_csv": str(selected_csv),
            "candidate_csv": str(candidates_csv),
            "ok_candidate_csv": str(ok_csv),
            "annotate_html": str(annotate_html),
        },
        "notes": [
            "Chinese crops use original OCR boxes on the source image.",
            "UG/KK crops use PP-OCR detected boxes when --detector ppocr is selected; otherwise they use rendered GT boxes.",
            "Yellow overlay box is the dataset GT/reference box; green overlay box is the actual crop detector box.",
            "Ground-truth text is taken from the dataset index, not from PP-OCR recognition.",
        ],
    }
    summary_path = output_dir / "candidate_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


