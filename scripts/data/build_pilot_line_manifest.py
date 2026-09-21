import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path


LANGS = ("zh", "ug", "kk")
LANG_ORDER = {"zh": 0, "ug": 1, "kk": 2}


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def safe_text(value):
    return "" if value is None else str(value)


def safe_id(value):
    return re.sub(r"[^0-9A-Za-z_.:-]+", "_", safe_text(value)).strip("_")


def image_stem(record):
    for key in ("source_image_workspace", "source_image_abs", "source_image", "image"):
        value = record.get(key)
        if value:
            return Path(str(value)).stem
    return safe_id(record.get("sample_id") or "unknown_image")


def derive_source_id(record):
    dataset = record.get("source_dataset") or record.get("dataset_source") or "unknown_dataset"
    return f"{dataset}:{image_stem(record)}"


def record_split(record):
    return record.get("final_split") or record.get("split") or "unknown"


def resolve_workspace_path(value, workspace_root):
    if not value:
        return ""
    path = Path(str(value))
    if path.is_absolute() and path.exists():
        return str(path)
    candidate = workspace_root / str(value)
    if candidate.exists():
        return str(candidate)
    normalized = str(value).replace("\\", "/")
    if ":/" in normalized:
        tail = normalized.split(":/", 1)[1]
        for marker in (
            "Misogyny_Dataset_Project/",
            "Meme_Dataset_Project/",
            "final_multilingual_meme_ocr_dataset/",
        ):
            if marker in tail:
                candidate = workspace_root / tail[tail.index(marker) :]
                if candidate.exists():
                    return str(candidate)
    meme_image = workspace_root / "Meme_Dataset_Project" / "images" / "meme" / Path(str(value)).name
    if meme_image.exists():
        return str(meme_image)
    return str(value)


def language_text(line, language):
    if language == "zh":
        return line.get("zh_text") or line.get("source_text") or ""
    return line.get("render_text_normalized") or line.get(f"{language}_text") or line.get("text") or ""


def language_image(record, language):
    if language == "zh":
        return record.get("source_image_workspace") or record.get("source_image_abs") or record.get("source_image") or ""
    return record.get("rendered_image_workspace") or record.get("rendered_image_abs") or record.get("rendered_image") or ""


def language_bbox(line, language):
    if language == "zh":
        return line.get("bbox")
    return line.get("render_bbox") or line.get("render_layout_bbox") or line.get("bbox")


def bbox_as_xywh(box):
    if isinstance(box, dict):
        x = float(box.get("x", 0))
        y = float(box.get("y", 0))
        w = float(box.get("width", 0))
        h = float(box.get("height", 0))
        return x, y, w, h
    if isinstance(box, list) and len(box) == 4:
        x1, y1, x2, y2 = [float(v) for v in box]
        return x1, y1, x2 - x1, y2 - y1
    return 0.0, 0.0, 0.0, 0.0


def make_line_row(record, line, language, workspace_root):
    source_id = derive_source_id(record)
    line_index = int(line.get("line_index", 0))
    box = language_bbox(line, language)
    x, y, w, h = bbox_as_xywh(box)
    text = language_text(line, language)
    split = record_split(record)
    source_dataset = record.get("source_dataset") or record.get("dataset_source") or ""
    image_path = language_image(record, language)

    return {
        "candidate_id": f"{safe_id(source_id)}_{language}_L{line_index + 1:03d}",
        "source_id": source_id,
        "source_dataset": source_dataset,
        "split": split,
        "language": language,
        "line_index": line_index,
        "image_path": image_path,
        "image_path_resolved": resolve_workspace_path(image_path, workspace_root),
        "bbox": compact_json(box),
        "bbox_x": round(x, 3),
        "bbox_y": round(y, 3),
        "bbox_w": round(w, 3),
        "bbox_h": round(h, 3),
        "bbox_aspect": round(w / h, 4) if h else "",
        "text": text,
        "text_length": len(text),
        "source_image": record.get("source_image_workspace") or record.get("source_image_abs") or "",
        "rendered_image": record.get("rendered_image_workspace") or record.get("rendered_image_abs") or "",
        "sample_id": record.get("sample_id") or "",
        "review_status": "pending",
        "review_note": "",
    }


def load_records(path):
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise TypeError(f"Expected JSON list in {path}")
    return records


def collect_rows(records, workspace_root, split):
    grouped = defaultdict(list)
    for record in records:
        if record_split(record) == split:
            grouped[derive_source_id(record)].append(record)

    rows = []
    for source_id in sorted(grouped):
        by_lang = {}
        for record in grouped[source_id]:
            lang = record.get("language") or record.get("render_lang")
            if lang in ("ug", "kk"):
                by_lang[lang] = record

        base_record = by_lang.get("ug") or by_lang.get("kk")
        if base_record:
            for line in base_record.get("texts") or []:
                rows.append(make_line_row(base_record, line, "zh", workspace_root))

        for language in ("ug", "kk"):
            record = by_lang.get(language)
            if not record:
                continue
            for line in record.get("texts") or []:
                rows.append(make_line_row(record, line, language, workspace_root))

    return rows


def sample_balanced(rows, target_per_language, seed):
    rng = random.Random(seed)
    by_lang = defaultdict(list)
    for row in rows:
        if row["text"].strip() and row["bbox_w"] and row["bbox_h"]:
            by_lang[row["language"]].append(row)

    selected = []
    for language in LANGS:
        items = by_lang.get(language, [])
        if len(items) <= target_per_language:
            chosen = list(items)
        else:
            chosen = rng.sample(items, target_per_language)
        selected.extend(chosen)

    selected.sort(
        key=lambda row: (
            LANG_ORDER.get(row["language"], 99),
            row["source_dataset"],
            row["source_id"],
            int(row["line_index"]),
        )
    )
    return selected, {language: len(by_lang.get(language, [])) for language in LANGS}


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
    input_path = workspace_root / "final_multilingual_meme_ocr_dataset" / "final_all.json"
    output_dir = svtr_root / "01_data_preparation" / "real_lines" / "pilot_manifest"
    return workspace_root, input_path, output_dir


def main():
    default_workspace, default_input, default_output = default_paths()
    parser = argparse.ArgumentParser(
        description="Build a balanced candidate manifest for the first 300-500 real meme line pilot."
    )
    parser.add_argument("--workspace-root", type=Path, default=default_workspace)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--target-per-language", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    records = load_records(args.input)
    all_rows = collect_rows(records, args.workspace_root, args.split)
    selected, candidate_counts = sample_balanced(all_rows, args.target_per_language, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"{args.split}_pilot_line_candidates.csv"
    write_csv(manifest_path, selected)

    summary = {
        "input": str(args.input),
        "workspace_root": str(args.workspace_root),
        "output_dir": str(args.output_dir),
        "split": args.split,
        "target_per_language": args.target_per_language,
        "seed": args.seed,
        "candidate_counts_by_language": candidate_counts,
        "selected_counts_by_language": {
            language: sum(1 for row in selected if row["language"] == language)
            for language in LANGS
        },
        "selected_total": len(selected),
        "output_manifest": str(manifest_path),
        "review_policy": {
            "pass": "complete single line, label confirmed, small surrounding background ok",
            "recrop": "usable source but bbox needs manual adjustment",
            "drop": "truncated, full neighboring line, inseparable multi-line text, or uncertain label",
        },
    }
    summary_path = args.output_dir / f"{args.split}_pilot_line_candidates_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

