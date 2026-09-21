#!/usr/bin/env python3
"""Audit Unicode coverage of OCR synthesis fonts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from fontTools.ttLib import TTCollection, TTFont


FONT_SUFFIXES = {".ttf", ".otf", ".ttc", ".otc"}

UG_CORE_CODEPOINTS = [
    0x0626, 0x0627, 0x06D5, 0x0628, 0x067E, 0x062A, 0x062C, 0x0686,
    0x062E, 0x062F, 0x0631, 0x0632, 0x0698, 0x0633, 0x0634, 0x063A,
    0x0641, 0x0642, 0x0643, 0x06AF, 0x06AD, 0x0644, 0x0645, 0x0646,
    0x06BE, 0x0648, 0x06C7, 0x06C6, 0x06C8, 0x06CB, 0x06D0, 0x0649,
    0x064A,
]

KK_SPECIFIC_CODEPOINTS = [
    0x04D8, 0x04D9, 0x0492, 0x0493, 0x049A, 0x049B, 0x04A2, 0x04A3,
    0x04E8, 0x04E9, 0x04B0, 0x04B1, 0x04AE, 0x04AF, 0x04BA, 0x04BB,
    0x0406, 0x0456,
]


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--font-dir", type=Path, action="append", required=True)
    parser.add_argument("--metadata", type=Path, default=defaults["metadata"])
    parser.add_argument("--text-pool-dir", type=Path, default=defaults["text_pool_dir"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--observed-coverage-threshold", type=float, default=0.98)
    return parser.parse_args()


def default_paths() -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    return {
        "metadata": svtr_root / "01_data_preparation" / "real_line_dataset_eval_reviewed" / "metadata.csv",
        "text_pool_dir": svtr_root / "02_corpus_preparation" / "mixed_text_pool_v2",
        "output_dir": svtr_root / "03_synthetic_generation" / "font_library",
    }


def load_metadata_chars(path: Path) -> dict[str, set[int]]:
    chars: dict[str, set[int]] = defaultdict(set)
    if not path.exists():
        return chars
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            language = row.get("language", "")
            text = row.get("text") or row.get("logical_text") or ""
            for ch in text:
                if not ch.isspace():
                    chars[language].add(ord(ch))
    return chars


def load_text_pool_chars(path: Path) -> dict[str, set[int]]:
    chars: dict[str, set[int]] = defaultdict(set)
    if not path.exists():
        return chars
    for language in ("zh", "ug", "kk"):
        file_path = path / f"{language}_mixed_text_pool_v2.jsonl"
        if not file_path.exists():
            continue
        with file_path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("normalized_text") or obj.get("text") or ""
                for ch in text:
                    if not ch.isspace():
                        chars[language].add(ord(ch))
    return chars


def merge_chars(*items: dict[str, set[int]]) -> dict[str, set[int]]:
    merged: dict[str, set[int]] = defaultdict(set)
    for item in items:
        for language, values in item.items():
            merged[language].update(values)
    merged["ug"].update(UG_CORE_CODEPOINTS)
    merged["kk"].update(KK_SPECIFIC_CODEPOINTS)
    return merged


def scan_font_files(font_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for font_dir in font_dirs:
        if not font_dir.exists():
            raise FileNotFoundError(font_dir)
        for path in font_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in FONT_SUFFIXES:
                files.append(path)
    return sorted(set(files))


def font_faces(path: Path):
    if path.suffix.lower() in {".ttc", ".otc"}:
        collection = TTCollection(str(path), lazy=True)
        for idx, font in enumerate(collection.fonts):
            yield idx, font
    else:
        yield 0, TTFont(str(path), lazy=True)


def font_name(font: TTFont) -> str:
    names = font["name"].names if "name" in font else []
    for name_id in (16, 1, 4):
        for record in names:
            if record.nameID == name_id:
                try:
                    return str(record.toUnicode()).strip()
                except Exception:
                    continue
    return ""


def cmap_codepoints(font: TTFont) -> set[int]:
    values: set[int] = set()
    if "cmap" not in font:
        return values
    for table in font["cmap"].tables:
        values.update(table.cmap.keys())
    return values


def coverage(required: set[int], available: set[int]) -> tuple[int, int, float, list[str]]:
    missing = sorted(required - available)
    total = len(required)
    ok = total - len(missing)
    ratio = ok / total if total else 1.0
    return ok, total, ratio, [f"U+{cp:04X}" for cp in missing]


def script_required(language: str, required: set[int]) -> set[int]:
    if language == "zh":
        return {cp for cp in required if 0x4E00 <= cp <= 0x9FFF}
    if language == "ug":
        return {
            cp for cp in required
            if (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F)
        } | set(UG_CORE_CODEPOINTS)
    if language == "kk":
        return {cp for cp in required if 0x0400 <= cp <= 0x04FF} | set(KK_SPECIFIC_CODEPOINTS)
    return set(required)


def pass_for_language(
    language: str,
    observed_ratio: float,
    core_ratio: float,
    script_ratio: float,
    threshold: float,
) -> bool:
    if language in {"ug", "kk"}:
        return core_ratio == 1.0 and script_ratio == 1.0
    return core_ratio == 1.0 and observed_ratio >= threshold


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required = merge_chars(load_metadata_chars(args.metadata), load_text_pool_chars(args.text_pool_dir))
    font_files = scan_font_files(args.font_dir)
    rows: list[dict[str, object]] = []
    manifest: dict[str, list[dict[str, object]]] = {"zh": [], "ug": [], "kk": []}

    for path in font_files:
        for face_index, font in font_faces(path):
            try:
                available = cmap_codepoints(font)
                name = font_name(font) or path.stem
            finally:
                font.close()

            row_base = {
                "font_name": name,
                "font_path": str(path),
                "face_index": face_index,
            }
            lang_pass = {}
            for language in ("zh", "ug", "kk"):
                ok, total, ratio, missing = coverage(required[language], available)
                script_ok, script_total, script_ratio, script_missing = coverage(
                    script_required(language, required[language]), available
                )
                core_required = set()
                if language == "ug":
                    core_required = set(UG_CORE_CODEPOINTS)
                elif language == "kk":
                    core_required = set(KK_SPECIFIC_CODEPOINTS)
                core_ok, core_total, core_ratio, core_missing = coverage(core_required, available)
                passed = pass_for_language(
                    language, ratio, core_ratio, script_ratio, args.observed_coverage_threshold
                )
                lang_pass[language] = passed
                rows.append({
                    **row_base,
                    "language": language,
                    "required_chars": total,
                    "covered_chars": ok,
                    "coverage": f"{ratio:.6f}",
                    "core_required_chars": core_total,
                    "core_covered_chars": core_ok,
                    "core_coverage": f"{core_ratio:.6f}",
                    "script_required_chars": script_total,
                    "script_covered_chars": script_ok,
                    "script_coverage": f"{script_ratio:.6f}",
                    "pass": "yes" if passed else "no",
                    "missing_core": " ".join(core_missing[:100]),
                    "missing_script": " ".join(script_missing[:100]),
                    "missing_observed_count": len(missing),
                    "missing_observed_preview": " ".join(missing[:100]),
                })
            for language, passed in lang_pass.items():
                if passed and path.parent.name == language:
                    manifest[language].append({
                        "font_name": name,
                        "font_path": str(path),
                        "face_index": face_index,
                        "role": "candidate_after_coverage_audit",
                    })

    with (args.output_dir / "font_coverage_report.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "font_name", "font_path", "face_index", "language", "required_chars",
            "covered_chars", "coverage", "core_required_chars", "core_covered_chars",
            "core_coverage", "script_required_chars", "script_covered_chars",
            "script_coverage", "pass", "missing_core", "missing_script", "missing_observed_count",
            "missing_observed_preview",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "font_dirs": [str(path) for path in args.font_dir],
        "metadata": str(args.metadata),
        "text_pool_dir": str(args.text_pool_dir),
        "observed_coverage_threshold": args.observed_coverage_threshold,
        "required_char_counts": {language: len(chars) for language, chars in required.items()},
        "font_files": len(font_files),
        "faces": len({(row["font_path"], row["face_index"]) for row in rows}),
        "pass_counts": Counter(
            f"{row['language']}:{row['pass']}" for row in rows
        ),
        "outputs": {
            "report_csv": str(args.output_dir / "font_coverage_report.csv"),
            "manifest_json": str(args.output_dir / "font_manifest_candidates.json"),
        },
    }
    summary["pass_counts"] = dict(summary["pass_counts"])

    with (args.output_dir / "font_manifest_candidates.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "font_coverage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
