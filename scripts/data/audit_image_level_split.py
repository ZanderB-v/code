import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "dev", "test")
LANGS = ("ug", "kk")


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


def source_image_path(record):
    return (
        record.get("source_image_workspace")
        or record.get("source_image_abs")
        or record.get("source_image")
        or record.get("image")
        or ""
    )


def line_count(record):
    return len(record.get("texts") or [])


def load_records(path):
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise TypeError(f"Expected a JSON list in {path}, got {type(records).__name__}")
    return records


def build_manifest(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[derive_source_id(record)].append(record)

    rows = []
    leak_rows = []
    missing_language_rows = []
    line_mismatch_rows = []

    for source_id in sorted(grouped):
        group = grouped[source_id]
        by_lang = {}
        for record in group:
            lang = record.get("language") or record.get("render_lang")
            if lang in LANGS:
                by_lang.setdefault(lang, []).append(record)

        splits = sorted({record_split(record) for record in group})
        source_datasets = sorted(
            {safe_text(record.get("source_dataset") or record.get("dataset_source")) for record in group}
        )
        base = by_lang.get("ug", by_lang.get("kk", group))[0]

        ug_records = by_lang.get("ug", [])
        kk_records = by_lang.get("kk", [])
        ug_record = ug_records[0] if ug_records else None
        kk_record = kk_records[0] if kk_records else None
        ug_lines = line_count(ug_record) if ug_record else 0
        kk_lines = line_count(kk_record) if kk_record else 0

        row = {
            "source_id": source_id,
            "split": splits[0] if len(splits) == 1 else "|".join(splits),
            "source_dataset": "|".join(source_datasets),
            "source_image": source_image_path(base),
            "width": base.get("width", ""),
            "height": base.get("height", ""),
            "ug_sample_id": ug_record.get("sample_id", "") if ug_record else "",
            "kk_sample_id": kk_record.get("sample_id", "") if kk_record else "",
            "ug_record_count": len(ug_records),
            "kk_record_count": len(kk_records),
            "ug_line_count": ug_lines,
            "kk_line_count": kk_lines,
            "language_status": "ok" if len(ug_records) == 1 and len(kk_records) == 1 else "issue",
            "line_alignment_status": "ok" if ug_lines == kk_lines else "issue",
            "split_leak_status": "ok" if len(splits) == 1 else "leak",
            "background_allowed_for_synthesis": "yes" if splits == ["train"] else "no",
            "text_allowed_for_synthesis": "yes" if splits == ["train"] else "no",
        }
        rows.append(row)

        if len(splits) != 1:
            leak_rows.append(row)
        if len(ug_records) != 1 or len(kk_records) != 1:
            missing_language_rows.append(row)
        if ug_lines != kk_lines:
            line_mismatch_rows.append(row)

    return rows, leak_rows, missing_language_rows, line_mismatch_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_split_lists(output_dir, rows):
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        text = "\n".join(row["source_id"] for row in split_rows)
        if text:
            text += "\n"
        (output_dir / f"{split}_source_ids.txt").write_text(text, encoding="utf-8")


def make_summary(input_path, output_dir, records, rows, leak_rows, missing_language_rows, line_mismatch_rows):
    split_counts = Counter(row["split"] for row in rows)
    record_counts = Counter((record_split(record), record.get("language") or record.get("render_lang")) for record in records)
    line_counts = {
        split: {
            lang: sum(int(row[f"{lang}_line_count"]) for row in rows if row["split"] == split)
            for lang in LANGS
        }
        for split in SPLITS
    }

    return {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "image_level_record_count": len(records),
        "source_image_count": len(rows),
        "source_counts_by_split": {split: split_counts.get(split, 0) for split in SPLITS},
        "record_counts_by_split_language": {
            split: {lang: record_counts.get((split, lang), 0) for lang in LANGS}
            for split in SPLITS
        },
        "line_counts_by_split_language": line_counts,
        "split_leak_count": len(leak_rows),
        "missing_or_duplicate_language_count": len(missing_language_rows),
        "line_alignment_mismatch_count": len(line_mismatch_rows),
        "background_policy": {
            "train": "allowed",
            "dev": "forbidden",
            "test": "forbidden",
        },
        "text_synthesis_policy": {
            "train": "allowed",
            "dev": "forbidden",
            "test": "forbidden",
        },
        "status": "pass"
        if not leak_rows and not missing_language_rows and not line_mismatch_rows
        else "fail",
        "outputs": {
            "manifest_csv": str(output_dir / "image_level_split_manifest.csv"),
            "split_leaks_csv": str(output_dir / "image_level_split_leaks.csv"),
            "language_issues_csv": str(output_dir / "image_level_language_issues.csv"),
            "line_mismatches_csv": str(output_dir / "image_level_line_mismatches.csv"),
            "summary_json": str(output_dir / "image_level_split_summary.json"),
        },
    }


def default_paths():
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    workspace_root = script_path.parents[5]
    input_path = workspace_root / "final_multilingual_meme_ocr_dataset" / "final_all.json"
    output_dir = svtr_root / "01_data_preparation" / "source_splits"
    return input_path, output_dir


def main():
    default_input, default_output = default_paths()

    parser = argparse.ArgumentParser(
        description="Audit image-level split integrity before line-level SVTRv2 recognition experiments."
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    args = parser.parse_args()

    records = load_records(args.input)
    rows, leak_rows, missing_language_rows, line_mismatch_rows = build_manifest(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "image_level_split_manifest.csv", rows)
    write_csv(args.output_dir / "image_level_split_leaks.csv", leak_rows)
    write_csv(args.output_dir / "image_level_language_issues.csv", missing_language_rows)
    write_csv(args.output_dir / "image_level_line_mismatches.csv", line_mismatch_rows)
    write_split_lists(args.output_dir, rows)

    summary = make_summary(
        args.input,
        args.output_dir,
        records,
        rows,
        leak_rows,
        missing_language_rows,
        line_mismatch_rows,
    )
    (args.output_dir / "image_level_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
