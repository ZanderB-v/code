#!/usr/bin/env python3
"""Clean mixed_text_pool_v1 into a stricter mixed_text_pool_v2."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


LANGUAGES = ("zh", "ug", "kk")

KAZAKH_SPECIFIC_CHARS = set(
    chr(code)
    for code in [
        0x04D8, 0x04D9,
        0x0492, 0x0493,
        0x049A, 0x049B,
        0x04A2, 0x04A3,
        0x04E8, 0x04E9,
        0x04B0, 0x04B1,
        0x04AE, 0x04AF,
        0x04BA, 0x04BB,
        0x0406, 0x0456,
    ]
)

URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HTML_RE = re.compile(r"<[^>]+>|&[A-Za-z]{2,10};|&#\d+;")
LONG_ID_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{16,}(?![A-Za-z0-9])")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s\-()]{8,}\d)(?!\d)")
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=defaults["input_dir"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len-main", type=int, default=60)
    parser.add_argument("--max-len-reserve", type=int, default=90)
    parser.add_argument("--long-reserve-per-lang", type=int, default=500)
    parser.add_argument("--min-ug-arabic-ratio", type=float, default=0.70)
    parser.add_argument("--min-kk-cyrillic-ratio", type=float, default=0.70)
    parser.add_argument("--min-zh-han-ratio", type=float, default=0.50)
    parser.add_argument("--min-kk-specific-ratio", type=float, default=0.30)
    return parser.parse_args()


def default_paths() -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    corpus_root = svtr_root / "02_corpus_preparation"
    return {
        "input_dir": corpus_root / "mixed_text_pool_v1",
        "output_dir": corpus_root / "mixed_text_pool_v2",
    }


def normalize_text(raw_text: str) -> str:
    text = unicodedata.normalize("NFC", str(raw_text or ""))
    text = "".join(ch for ch in text if unicodedata.category(ch) not in {"Cc", "Cf"})
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def dedupe_key(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    kept = []
    for ch in text:
        category = unicodedata.category(ch)
        if category[0] in {"L", "N"}:
            kept.append(ch)
    return "".join(kept)


def char_ratio(text: str, predicate) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if predicate(ch)) / len(text)


def is_arabic_char(ch: str) -> bool:
    return ("\u0600" <= ch <= "\u06ff") or ("\u0750" <= ch <= "\u077f")


def is_cyrillic_char(ch: str) -> bool:
    return "\u0400" <= ch <= "\u04ff"


def is_han_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def length_bucket(length: int) -> str:
    if length <= 10:
        return "very_short_3_10"
    if length <= 20:
        return "short_11_20"
    if length <= 32:
        return "medium_21_32"
    if length <= 48:
        return "long_33_48"
    return "extra_long_49_plus"


def reject_reason(text: str, language: str, args: argparse.Namespace) -> str | None:
    if not text:
        return "empty"
    if len(text) < args.min_len:
        return "too_short"
    if URL_RE.search(text):
        return "url"
    if EMAIL_RE.search(text):
        return "email"
    if HTML_RE.search(text):
        return "html"
    if PHONE_RE.search(text):
        return "phone_like"
    if LONG_ID_RE.search(text):
        return "long_id_like"
    if not any(ch.isalnum() for ch in text):
        return "pure_symbol"

    if language == "zh":
        if char_ratio(text, is_han_char) < args.min_zh_han_ratio:
            return "low_han_ratio"
    elif language == "ug":
        if char_ratio(text, is_arabic_char) < args.min_ug_arabic_ratio:
            return "low_arabic_ratio"
    elif language == "kk":
        if char_ratio(text, is_cyrillic_char) < args.min_kk_cyrillic_ratio:
            return "low_cyrillic_ratio"
    else:
        return "unknown_language"
    return None


def source_path(input_dir: Path, language: str) -> Path:
    return input_dir / f"{language}_mixed_text_pool_50000.jsonl"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def describe_lengths(values: list[int]) -> dict[str, int | None]:
    values = sorted(values)
    if not values:
        return {name: None for name in ["min", "p25", "median", "p75", "p90", "p95", "p99", "max"]}
    return {
        "min": values[0],
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": values[-1],
    }


def percentile(values: list[int], q: float) -> int:
    return values[round((len(values) - 1) * q)]


def summarize_rows(rows: list[dict[str, object]], language: str) -> dict[str, object]:
    lengths = [int(row["text_length"]) for row in rows]
    source_counts = Counter(str(row.get("source_group", "")) for row in rows)
    bucket_counts = Counter(str(row.get("length_bucket", "")) for row in rows)
    summary: dict[str, object] = {
        "samples": len(rows),
        "source_counts": dict(source_counts),
        "source_ratios": {k: v / len(rows) for k, v in source_counts.items()} if rows else {},
        "length": describe_lengths(lengths),
        "length_bucket_counts": dict(bucket_counts),
        "length_bucket_ratios": {k: v / len(rows) for k, v in bucket_counts.items()} if rows else {},
        "duplicates": len(rows) - len({str(row["normalized_text"]) for row in rows}),
    }
    if language == "kk":
        kk_specific = sum(1 for row in rows if row.get("has_kazakh_specific"))
        summary["kazakh_specific_chars"] = {
            "chars": "".join(sorted(KAZAKH_SPECIFIC_CHARS)),
            "count": kk_specific,
            "ratio": kk_specific / len(rows) if rows else 0,
        }
    return summary


def clean_language(language: str, args: argparse.Namespace) -> dict[str, object]:
    input_path = source_path(args.input_dir, language)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    main_rows: list[dict[str, object]] = []
    reserve_rows: list[dict[str, object]] = []
    dropped_rows: list[dict[str, object]] = []
    seen_text_keys: set[str] = set()
    seen_near_keys: set[str] = set()
    stats = Counter()

    with input_path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            stats["input_rows"] += 1
            obj = json.loads(line)
            raw_text = str(obj.get("raw_text") or obj.get("text") or "")
            normalized_text = normalize_text(raw_text)
            reason = reject_reason(normalized_text, language, args)
            if reason:
                stats[f"drop_{reason}"] += 1
                dropped_rows.append(drop_row(obj, raw_text, normalized_text, reason, line_no))
                continue

            text_key = normalized_text
            near_key = dedupe_key(normalized_text)
            if text_key in seen_text_keys:
                stats["drop_duplicate_exact"] += 1
                dropped_rows.append(drop_row(obj, raw_text, normalized_text, "duplicate_exact", line_no))
                continue
            if near_key in seen_near_keys:
                stats["drop_duplicate_near"] += 1
                dropped_rows.append(drop_row(obj, raw_text, normalized_text, "duplicate_near", line_no))
                continue

            length = len(normalized_text)
            if length > args.max_len_reserve:
                stats["drop_too_long_gt_reserve"] += 1
                dropped_rows.append(drop_row(obj, raw_text, normalized_text, "too_long_gt_reserve", line_no))
                continue

            seen_text_keys.add(text_key)
            seen_near_keys.add(near_key)

            out = dict(obj)
            out["raw_text"] = raw_text
            out["normalized_text"] = normalized_text
            out["text"] = normalized_text
            out["text_length"] = length
            out["length_bucket"] = length_bucket(length)
            out["unicode_normalization"] = "NFC"
            out["pool_version"] = "mixed_text_pool_v2"
            if language == "kk":
                out["has_kazakh_specific"] = any(ch in KAZAKH_SPECIFIC_CHARS for ch in normalized_text)

            if length <= args.max_len_main:
                main_rows.append(out)
                stats["kept_main"] += 1
            else:
                stats["kept_long_reserve_candidate"] += 1
                reserve_rows.append(out)

    reserve_rows = reserve_rows[: args.long_reserve_per_lang]
    for idx, row in enumerate(main_rows, 1):
        row["id"] = f"{language}_mixed_v2_{idx:08d}"
    for idx, row in enumerate(reserve_rows, 1):
        row["id"] = f"{language}_mixed_v2_long_{idx:08d}"

    if language == "kk":
        ratio = (
            sum(1 for row in main_rows if row.get("has_kazakh_specific")) / len(main_rows)
            if main_rows else 0
        )
        if ratio < args.min_kk_specific_ratio:
            raise RuntimeError(
                f"KK specific-char ratio {ratio:.4f} is below required {args.min_kk_specific_ratio:.4f}"
            )

    output_path = args.output_dir / f"{language}_mixed_text_pool_v2.jsonl"
    reserve_path = args.output_dir / f"{language}_mixed_text_pool_v2_long_reserve.jsonl"
    dropped_path = args.output_dir / f"{language}_mixed_text_pool_v2_dropped.jsonl"
    write_jsonl(output_path, main_rows)
    write_jsonl(reserve_path, reserve_rows)
    write_jsonl(dropped_path, dropped_rows)

    summary = {
        "language": language,
        "input": str(input_path),
        "output": str(output_path),
        "long_reserve_output": str(reserve_path),
        "dropped_output": str(dropped_path),
        "cleaning": {
            "unicode_normalization": "NFC",
            "min_len": args.min_len,
            "max_len_main": args.max_len_main,
            "max_len_reserve": args.max_len_reserve,
            "long_reserve_per_lang": args.long_reserve_per_lang,
            "min_ug_arabic_ratio": args.min_ug_arabic_ratio,
            "min_kk_cyrillic_ratio": args.min_kk_cyrillic_ratio,
            "min_zh_han_ratio": args.min_zh_han_ratio,
            "min_kk_specific_ratio": args.min_kk_specific_ratio,
        },
        "stats": dict(stats),
        "main": summarize_rows(main_rows, language),
        "long_reserve": summarize_rows(reserve_rows, language),
        "dropped_reason_counts": {
            key.removeprefix("drop_"): value for key, value in stats.items() if key.startswith("drop_")
        },
    }
    with (args.output_dir / f"{language}_mixed_text_pool_v2_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def drop_row(
    obj: dict[str, object],
    raw_text: str,
    normalized_text: str,
    reason: str,
    line_no: int,
) -> dict[str, object]:
    return {
        "source_id": obj.get("id", ""),
        "language": obj.get("language", ""),
        "source_group": obj.get("source_group", ""),
        "domain": obj.get("domain", ""),
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "reason": reason,
        "input_line": line_no,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {language: clean_language(language, args) for language in LANGUAGES}
    all_summary = {
        "pool_version": "mixed_text_pool_v2",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "languages": summaries,
    }
    with (args.output_dir / "mixed_text_pool_v2_summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
