#!/usr/bin/env python3
"""Build a unified Chinese/Uyghur/Kazakh CTC character dictionary."""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


UG_CORE_CHARS = list("ئاەبپتجچخدرزژسشغفقكگڭلمنھوۇۆۈۋېىي")
KK_SPECIFIC_CHARS = list("ӘәҒғҚқҢңӨөҰұҮүҺһІі")
LATIN = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
DIGITS = list("0123456789")
COMMON_PUNCT = list(
    " .,!?;:'\"()[]{}<>-_/\\|@#$%&*+=~`^，。！？；：、（）【】《》“”‘’…·"
)
MEME_SYMBOLS = list("！？!?~～哈哈哈hhhHHLUuOo")


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-metadata", type=Path, default=defaults["real_metadata"])
    parser.add_argument("--text-pool-dir", type=Path, default=defaults["text_pool_dir"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--include-long-reserve", action="store_true")
    parser.add_argument("--low-freq-threshold", type=int, default=3)
    return parser.parse_args()


def default_paths() -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    return {
        "real_metadata": svtr_root / "01_data_preparation" / "real_line_dataset_eval_reviewed" / "metadata.csv",
        "text_pool_dir": svtr_root / "02_corpus_preparation" / "mixed_text_pool_v2",
        "output_dir": svtr_root / "04_model_training" / "character_dict_hz_ug_kk_v1",
    }


def iter_real_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            yield {
                "text": row.get("text") or row.get("logical_text") or "",
                "language": row.get("language", ""),
                "split": row.get("split", ""),
                "source": "real_line",
                "id": row.get("candidate_id") or row.get("id") or "",
            }


def iter_pool_rows(text_pool_dir: Path, include_long_reserve: bool):
    for language in ("zh", "ug", "kk"):
        paths = [text_pool_dir / f"{language}_mixed_text_pool_v2.jsonl"]
        if include_long_reserve:
            paths.append(text_pool_dir / f"{language}_mixed_text_pool_v2_long_reserve.jsonl")
        for path in paths:
            if not path.exists() or path.stat().st_size == 0:
                continue
            with path.open("r", encoding="utf-8-sig") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    yield {
                        "text": obj.get("normalized_text") or obj.get("text") or "",
                        "language": language,
                        "split": "synthetic_train",
                        "source": "mixed_text_pool_v2",
                        "id": obj.get("id", ""),
                    }


def char_script(ch: str) -> str:
    cp = ord(ch)
    if "\u4e00" <= ch <= "\u9fff":
        return "han"
    if (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F):
        return "arabic"
    if 0x0400 <= cp <= 0x04FF:
        return "cyrillic"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isdigit():
        return "digit"
    if unicodedata.category(ch).startswith("P"):
        return "punctuation"
    if unicodedata.category(ch).startswith("S"):
        return "symbol"
    if ch.isspace():
        return "space"
    return "other"


def stable_order(chars: set[str]) -> list[str]:
    groups = {
        "han": [],
        "arabic": [],
        "cyrillic": [],
        "latin": [],
        "digit": [],
        "punctuation": [],
        "symbol": [],
        "other": [],
    }
    for ch in chars:
        script = char_script(ch)
        if script == "space":
            continue
        groups.setdefault(script, []).append(ch)

    ordered = []
    for name in ("han", "arabic", "cyrillic", "latin", "digit", "punctuation", "symbol", "other"):
        ordered.extend(sorted(set(groups.get(name, [])), key=lambda ch: ord(ch)))
    return ordered


def add_required_chars(chars: set[str]) -> None:
    chars.update(UG_CORE_CHARS)
    chars.update(KK_SPECIFIC_CHARS)
    chars.update(LATIN)
    chars.update(DIGITS)
    chars.update(COMMON_PUNCT)
    chars.update(MEME_SYMBOLS)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    freq = Counter()
    by_lang = defaultdict(Counter)
    by_split = defaultdict(Counter)
    examples = {}

    rows = list(iter_real_rows(args.real_metadata)) + list(iter_pool_rows(args.text_pool_dir, args.include_long_reserve))
    for row in rows:
        text = str(row["text"])
        language = str(row["language"])
        split = str(row["split"])
        for ch in text:
            if ch.isspace():
                continue
            freq[ch] += 1
            by_lang[language][ch] += 1
            by_split[split][ch] += 1
            examples.setdefault(ch, {"text": text, "language": language, "split": split, "source": row["source"], "id": row["id"]})

    dict_chars = set(freq)
    add_required_chars(dict_chars)
    ordered = stable_order(dict_chars)

    with (args.output_dir / "character_dict.txt").open("w", encoding="utf-8", newline="\n") as f:
        for ch in ordered:
            f.write(ch + "\n")

    with (args.output_dir / "character_frequency.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["char", "codepoint", "unicode_name", "script", "total", "zh", "ug", "kk", "train", "dev", "test", "synthetic_train", "example"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ch in ordered:
            ex = examples.get(ch, {})
            writer.writerow({
                "char": ch,
                "codepoint": f"U+{ord(ch):04X}",
                "unicode_name": unicodedata.name(ch, ""),
                "script": char_script(ch),
                "total": freq[ch],
                "zh": by_lang["zh"][ch],
                "ug": by_lang["ug"][ch],
                "kk": by_lang["kk"][ch],
                "train": by_split["train"][ch],
                "dev": by_split["dev"][ch],
                "test": by_split["test"][ch],
                "synthetic_train": by_split["synthetic_train"][ch],
                "example": ex.get("text", ""),
            })

    dict_set = set(ordered)
    unknown_rows = []
    for row in rows:
        missing = sorted({ch for ch in str(row["text"]) if not ch.isspace() and ch not in dict_set}, key=lambda ch: ord(ch))
        if missing:
            unknown_rows.append({**row, "missing": "".join(missing), "missing_codepoints": " ".join(f"U+{ord(ch):04X}" for ch in missing)})

    with (args.output_dir / "unknown_characters.txt").open("w", encoding="utf-8") as f:
        if not unknown_rows:
            f.write("")
        else:
            for row in unknown_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    low_freq_rows = []
    for ch, count in sorted(freq.items(), key=lambda item: (item[1], ord(item[0]))):
        if count > args.low_freq_threshold:
            continue
        script = char_script(ch)
        if script in {"han", "other", "symbol"}:
            low_freq_rows.append({
                "char": ch,
                "codepoint": f"U+{ord(ch):04X}",
                "unicode_name": unicodedata.name(ch, ""),
                "script": script,
                "count": count,
                **examples.get(ch, {}),
            })
    with (args.output_dir / "low_frequency_anomalies.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["char", "codepoint", "unicode_name", "script", "count", "text", "language", "split", "source", "id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(low_freq_rows)

    summary = {
        "real_metadata": str(args.real_metadata),
        "text_pool_dir": str(args.text_pool_dir),
        "include_long_reserve": args.include_long_reserve,
        "rows_read": len(rows),
        "dictionary_size": len(ordered),
        "frequency_unique_chars": len(freq),
        "unknown_rows": len(unknown_rows),
        "script_counts": dict(Counter(char_script(ch) for ch in ordered)),
        "required_checks": {
            "ug_core_missing": [ch for ch in UG_CORE_CHARS if ch not in dict_set],
            "kk_specific_missing": [ch for ch in KK_SPECIFIC_CHARS if ch not in dict_set],
            "latin_missing": [ch for ch in LATIN if ch not in dict_set],
            "digit_missing": [ch for ch in DIGITS if ch not in dict_set],
        },
        "real_split_char_coverage": {
            split: coverage_for_split(by_split[split], dict_set)
            for split in ("train", "dev", "test")
        },
        "outputs": {
            "character_dict": str(args.output_dir / "character_dict.txt"),
            "character_frequency": str(args.output_dir / "character_frequency.csv"),
            "unknown_characters": str(args.output_dir / "unknown_characters.txt"),
            "low_frequency_anomalies": str(args.output_dir / "low_frequency_anomalies.csv"),
        },
    }
    with (args.output_dir / "character_dict_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def coverage_for_split(counter: Counter, dict_set: set[str]) -> dict[str, float | int]:
    total = sum(counter.values())
    covered = sum(count for ch, count in counter.items() if ch in dict_set)
    missing = sorted([ch for ch in counter if ch not in dict_set], key=lambda ch: ord(ch))
    return {
        "chars": total,
        "covered": covered,
        "coverage": covered / total if total else 1.0,
        "missing_unique": len(missing),
    }


if __name__ == "__main__":
    main()
