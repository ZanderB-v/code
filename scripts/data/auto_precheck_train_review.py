import argparse
import csv
import difflib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_pilot_annotation_html import build_html, read_rows


PAGE_RE = re.compile(r"^train_candidates_(\d{4})\.csv$")
LINE_RE = re.compile(r"_L(\d+)$")
AUTO_NOTE_PREFIX = "auto_precheck:"


def default_workspace_root():
    return Path(__file__).resolve().parents[5]


def default_dataset_root():
    return Path(__file__).resolve().parents[2] / "01_data_preparation" / "real_line_dataset_train_aligned_v2"


def normalize_text(value):
    text = unicodedata.normalize("NFC", str(value or ""))
    text = text.replace("\ufeff", "").replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()


def image_stem(record):
    for key in ("source_image_workspace", "source_image_abs", "source_image", "image"):
        value = record.get(key)
        if value:
            return Path(str(value)).stem
    return str(record.get("sample_id") or "unknown")


def source_family_id(record):
    dataset = record.get("source_dataset") or record.get("dataset_source") or "unknown_dataset"
    return f"{dataset}:{image_stem(record)}"


def parse_line_index(candidate_id):
    match = LINE_RE.search(str(candidate_id or ""))
    return int(match.group(1)) - 1 if match else None


def box_xyxy(box):
    if not isinstance(box, dict):
        return None
    try:
        x = float(box.get("x", 0))
        y = float(box.get("y", 0))
        width = float(box.get("width", 0))
        height = float(box.get("height", 0))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0:
        return None
    return x, y, x + width, y + height


def box_size(box):
    coords = box_xyxy(box)
    if coords is None:
        return 0.0, 0.0
    return coords[2] - coords[0], coords[3] - coords[1]


def overlap_ratios(first, second):
    a = box_xyxy(first)
    b = box_xyxy(second)
    if a is None or b is None:
        return 0.0, 0.0, 0.0
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    if iw <= 0 or ih <= 0:
        return 0.0, 0.0, 0.0
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    horizontal = iw / max(1.0, min(aw, bw))
    vertical = ih / max(1.0, min(ah, bh))
    area = (iw * ih) / max(1.0, min(aw * ah, bw * bh))
    return horizontal, vertical, area


def raw_line_text(line, language):
    if language == "zh":
        return normalize_text(line.get("zh_text") or line.get("source_text") or line.get("text"))
    return normalize_text(line.get(f"{language}_text") or line.get("text"))


def read_needed_sources(review_dir, start_page, end_page):
    pages = []
    needed = defaultdict(set)
    for path in sorted(review_dir.glob("train_candidates_*.csv")):
        match = PAGE_RE.match(path.name)
        if not match:
            continue
        page = int(match.group(1))
        if page < start_page or (end_page is not None and page > end_page):
            continue
        rows = read_rows(path)
        pages.append((page, path, rows))
        for row in rows:
            language = row.get("language", "")
            if language in {"ug", "kk", "zh"}:
                needed[language].add(row.get("source_id", ""))
    return pages, needed


def load_index_records(path, language, needed_sources):
    if not path.exists():
        raise FileNotFoundError(f"Index not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    selected = {}
    for record in records:
        family = source_family_id(record)
        if family in needed_sources:
            selected[family] = record
    return selected


def inspect_crop(review_dir, row, verify_images=False):
    rel = str(row.get("crop_path_rel") or "").strip()
    if not rel:
        return "drop", "missing_crop_path"
    path = (review_dir / Path(rel)).resolve()
    if not path.exists():
        return "drop", "missing_crop_file"
    if not verify_images:
        return None, None
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        return "drop", "corrupt_crop_file"
    if width < 16 or height < 12:
        return "drop", f"crop_too_small_{width}x{height}"
    if width / max(1, height) > 40:
        return "recrop", f"extreme_aspect_{width / height:.2f}"
    return None, None


def language_letter_ratio(text, language):
    letters = [char for char in text if unicodedata.category(char).startswith("L")]
    if not letters:
        return None
    if language == "ug":
        expected = sum("ARABIC" in unicodedata.name(char, "") for char in letters)
    elif language == "kk":
        expected = sum("CYRILLIC" in unicodedata.name(char, "") for char in letters)
    elif language == "zh":
        expected = sum("CJK UNIFIED IDEOGRAPH" in unicodedata.name(char, "") for char in letters)
    else:
        return None
    return expected / len(letters)


def has_wrong_script_majority(text, language):
    letters = [char for char in text if unicodedata.category(char).startswith("L")]
    if not letters:
        return False
    names = [unicodedata.name(char, "") for char in letters]
    if language == "ug":
        wrong = sum("CYRILLIC" in name or "CJK UNIFIED IDEOGRAPH" in name for name in names)
    elif language == "kk":
        wrong = sum("ARABIC" in name or "CJK UNIFIED IDEOGRAPH" in name for name in names)
    elif language == "zh":
        wrong = sum("ARABIC" in name or "CYRILLIC" in name for name in names)
    else:
        return False
    return wrong / len(letters) >= 0.5


def comparison_core(text):
    normalized = unicodedata.normalize("NFKC", normalize_text(text)).casefold()
    return "".join(char for char in normalized if char.isalnum())


def estimated_text_width_ratio(text, font_size, layout_width):
    if font_size <= 0 or layout_width <= 0:
        return 0.0
    units = 0.0
    for char in text:
        if char.isspace():
            units += 0.28
        elif unicodedata.category(char).startswith("P"):
            units += 0.32
        elif char.isdigit():
            units += 0.55
        else:
            units += 0.56
    return units * font_size / layout_width


def assess_geometry(record, target_line, language):
    reasons = []
    severe = []
    layout = target_line.get("render_layout_bbox") or target_line.get("render_bbox") or target_line.get("bbox")
    render = target_line.get("render_bbox") or target_line.get("render_source_bbox") or target_line.get("bbox")
    layout_width, layout_height = box_size(layout)
    _, render_height = box_size(render)
    try:
        font_size = float(target_line.get("render_fitted_font_size") or 0)
    except (TypeError, ValueError):
        font_size = 0.0

    height_to_font = layout_height / font_size if font_size > 0 else 0.0
    layout_to_render_height = layout_height / render_height if render_height > 0 else 0.0
    text = raw_line_text(target_line, language)
    width_ratio = estimated_text_width_ratio(text, font_size, layout_width)
    multiline = (
        width_ratio >= 1.35
        and height_to_font >= 1.8
        and layout_to_render_height >= 1.4
        and layout_height >= 45
    )
    severe_multiline = multiline and width_ratio >= 1.65 and height_to_font >= 2.2
    if multiline:
        reasons.append(
            f"multiline_or_wrap_wr={width_ratio:.2f}_hf={height_to_font:.2f}_hr={layout_to_render_height:.2f}"
        )
    if severe_multiline:
        reasons.append("strong_multiline_wrap")

    max_layout_overlap = (0.0, 0.0, 0.0)
    max_render_overlap = (0.0, 0.0, 0.0)
    target_index = int(target_line.get("line_index", -1))
    for sibling in record.get("texts") or []:
        if int(sibling.get("line_index", -2)) == target_index:
            continue
        sibling_layout = sibling.get("render_layout_bbox") or sibling.get("render_bbox") or sibling.get("bbox")
        sibling_render = sibling.get("render_bbox") or sibling.get("render_source_bbox") or sibling.get("bbox")
        layout_overlap = overlap_ratios(layout, sibling_layout)
        render_overlap = overlap_ratios(render, sibling_render)
        if layout_overlap[2] > max_layout_overlap[2]:
            max_layout_overlap = layout_overlap
        if render_overlap[2] > max_render_overlap[2]:
            max_render_overlap = render_overlap

    layout_neighbor = max_layout_overlap[0] >= 0.45 and max_layout_overlap[1] >= 0.25
    strong_layout_neighbor = max_layout_overlap[0] >= 0.60 and max_layout_overlap[1] >= 0.35
    strong_render_neighbor = max_render_overlap[0] >= 0.50 and max_render_overlap[1] >= 0.35
    if layout_neighbor:
        reasons.append(
            "neighbor_layout_overlap_"
            f"x={max_layout_overlap[0]:.2f}_y={max_layout_overlap[1]:.2f}_a={max_layout_overlap[2]:.2f}"
        )
    if strong_layout_neighbor and strong_render_neighbor:
        severe.append("strong_neighbor_text_overlap")

    if language in {"ug", "kk"} and layout_width <= 0:
        severe.append("invalid_render_layout_bbox")
    return reasons, severe


def assess_row(review_dir, row, records_by_language, verify_images=False):
    language = row.get("language", "")
    text = normalize_text(row.get("text"))
    hard_reasons = []
    review_reasons = []

    image_status, image_reason = inspect_crop(review_dir, row, verify_images=verify_images)
    if image_status == "drop":
        hard_reasons.append(image_reason)
    elif image_status == "recrop":
        review_reasons.append(image_reason)

    if not text:
        hard_reasons.append("empty_label")
    if any(unicodedata.category(char) in {"Cc", "Cs"} and char not in "\t\n\r" for char in text):
        hard_reasons.append("control_character_in_label")

    ratio = language_letter_ratio(text, language)
    if ratio is not None and ratio < 0.2 and len(text) >= 4 and has_wrong_script_majority(text, language):
        review_reasons.append(f"wrong_{language}_script_ratio={ratio:.2f}")

    source_id = row.get("source_id", "")
    record = records_by_language.get(language, {}).get(source_id)
    line_index = parse_line_index(row.get("candidate_id"))
    if record is None or line_index is None:
        review_reasons.append("source_index_not_matched")
    else:
        target_line = next(
            (line for line in (record.get("texts") or []) if int(line.get("line_index", -1)) == line_index),
            None,
        )
        if target_line is None:
            hard_reasons.append("line_index_not_found")
        else:
            expected = raw_line_text(target_line, language)
            expected_core = comparison_core(expected)
            actual_core = comparison_core(text)
            if expected_core and actual_core:
                similarity = difflib.SequenceMatcher(None, expected_core, actual_core).ratio()
                if similarity < 0.75:
                    hard_reasons.append(f"label_core_differs_from_render_index={similarity:.2f}")
            if language in {"ug", "kk"}:
                geometry_reasons, severe_geometry = assess_geometry(record, target_line, language)
                review_reasons.extend(geometry_reasons)
                hard_reasons.extend(severe_geometry)

    if hard_reasons:
        return "drop", hard_reasons + review_reasons
    if review_reasons:
        return "recrop", review_reasons
    return "pass", []


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    workspace_root = default_workspace_root()
    parser = argparse.ArgumentParser(
        description="Conservatively precheck paginated train line-review data using renderer metadata."
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--index-ug", type=Path, default=workspace_root / "final_multilingual_meme_ocr_dataset" / "train_ug.json")
    parser.add_argument("--index-kk", type=Path, default=workspace_root / "final_multilingual_meme_ocr_dataset" / "train_kk.json")
    parser.add_argument("--start-page", type=int, default=6)
    parser.add_argument("--end-page", type=int, default=None)
    parser.add_argument("--suffix", default="auto_prechecked")
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    review_dir = args.dataset_root / "train_review"
    if not review_dir.exists():
        raise FileNotFoundError(f"Review directory not found: {review_dir}")
    pages, needed = read_needed_sources(review_dir, args.start_page, args.end_page)
    if not pages:
        raise ValueError("No matching train review pages found.")

    records_by_language = {"ug": {}, "kk": {}, "zh": {}}
    if needed["ug"] or needed["zh"]:
        ug_needed = needed["ug"] | needed["zh"]
        ug_records = load_index_records(args.index_ug, "ug", ug_needed)
        records_by_language["ug"] = ug_records
        records_by_language["zh"] = ug_records
    if needed["kk"]:
        records_by_language["kk"] = load_index_records(args.index_kk, "kk", needed["kk"])

    summary = {
        "dataset_root": str(args.dataset_root),
        "review_dir": str(review_dir),
        "start_page": args.start_page,
        "end_page": args.end_page,
        "dry_run": args.dry_run,
        "policy": {
            "drop": "Only deterministic corruption, strong label/index mismatch, or strong neighboring-text overlap.",
            "recrop": "Potential wrap/neighbor overlap/script/aspect risks requiring fast manual confirmation.",
            "pass": "No automatic high-confidence risk detected; not equivalent to human verification.",
        },
        "pages": [],
        "counts": Counter(),
        "reasons": Counter(),
    }
    links = []
    for page, source_csv, rows in pages:
        page_counts = Counter()
        page_reasons = Counter()
        updated = []
        for row in rows:
            status, reasons = assess_row(
                review_dir,
                row,
                records_by_language,
                verify_images=args.verify_images,
            )
            clean_note = str(row.get("review_note") or "")
            if clean_note.startswith(AUTO_NOTE_PREFIX):
                clean_note = ""
            auto_note = f"{AUTO_NOTE_PREFIX} {'; '.join(reasons)}" if reasons else ""
            row["review_status"] = status
            row["review_note"] = auto_note or clean_note
            updated.append(row)
            page_counts[status] += 1
            for reason in reasons:
                page_reasons[reason.split("=", 1)[0]] += 1

        stem = source_csv.stem
        output_csv = review_dir / f"{stem}.{args.suffix}.csv"
        output_html = review_dir / f"annotate_{page:04d}.{args.suffix}.html"
        fields = list(updated[0].keys()) if updated else []
        if not args.dry_run:
            write_csv(output_csv, updated, fields)
            output_html.write_text(build_html(updated), encoding="utf-8")
        summary["counts"].update(page_counts)
        summary["reasons"].update(page_reasons)
        summary["pages"].append(
            {
                "page": page,
                "rows": len(updated),
                "counts": dict(page_counts),
                "reasons": dict(page_reasons),
                "csv": str(output_csv),
                "html": str(output_html),
            }
        )
        links.append(
            f"<li><a href='{output_html.name}'>{output_html.name}</a> "
            f"pass={page_counts['pass']} recrop={page_counts['recrop']} drop={page_counts['drop']}</li>"
        )
        print(json.dumps(summary["pages"][-1], ensure_ascii=False))

    summary["counts"] = dict(summary["counts"])
    summary["reasons"] = dict(summary["reasons"])
    summary_path = review_dir / f"auto_precheck_pages_{args.start_page:04d}_to_{pages[-1][0]:04d}_summary.json"
    index_path = review_dir / f"auto_precheck_pages_{args.start_page:04d}_to_{pages[-1][0]:04d}.html"
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        index_path.write_text(
            "<!doctype html><html><head><meta charset='utf-8'><title>Auto Prechecked Train Review</title>"
            "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:24px;line-height:1.7}a{color:#1f5f8b}</style>"
            "</head><body><h1>Auto Prechecked Train Review</h1><ol>"
            + "".join(links)
            + "</ol></body></html>",
            encoding="utf-8",
        )
    summary["summary_json"] = str(summary_path)
    summary["index_html"] = str(index_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
