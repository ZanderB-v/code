#!/usr/bin/env python3
"""Build mixed synthetic text pools from parallel phrases and real train meme text."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


CSV_ENCODING = "utf-8-sig"
KAZAKH_SPECIFIC_CHARS = set(
    chr(code)
    for code in [
        0x04D8, 0x04D9,  # Ә ә
        0x0492, 0x0493,  # Ғ ғ
        0x049A, 0x049B,  # Қ қ
        0x04A2, 0x04A3,  # Ң ң
        0x04E8, 0x04E9,  # Ө ө
        0x04B0, 0x04B1,  # Ұ ұ
        0x04AE, 0x04AF,  # Ү ү
        0x04BA, 0x04BB,  # Һ һ
        0x0406, 0x0456,  # І і
    ]
)


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel-pool-dir", type=Path, default=defaults["parallel_pool_dir"])
    parser.add_argument("--real-metadata", type=Path, default=defaults["real_metadata"])
    parser.add_argument("--out-dir", type=Path, default=defaults["out_dir"])
    parser.add_argument("--samples-per-lang", type=int, default=50000)
    parser.add_argument("--parallel-ratio", type=float, default=0.70)
    parser.add_argument("--real-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max-len-zh", type=int, default=90)
    parser.add_argument("--max-len-ugkk", type=int, default=90)
    return parser.parse_args()


def default_paths() -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    return {
        "parallel_pool_dir": svtr_root / "02_corpus_preparation" / "synthetic_text_pool",
        "real_metadata": svtr_root
        / "01_data_preparation"
        / "real_line_dataset_eval_reviewed"
        / "metadata.csv",
        "out_dir": svtr_root / "02_corpus_preparation" / "mixed_text_pool_v1",
    }


def clean_text(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"[\r\n\u2028\u2029\x85]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def length_bucket(length: int) -> str:
    if length <= 15:
        return "short"
    if length <= 35:
        return "medium"
    return "long"


def max_len_for_language(language: str, args: argparse.Namespace) -> int:
    if language == "zh":
        return args.max_len_zh
    return args.max_len_ugkk


def has_kazakh_specific(text: str) -> bool:
    return any(ch in KAZAKH_SPECIFIC_CHARS for ch in text)


def load_parallel_rows(path: Path, language: str, max_len: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = clean_text(obj.get("text", ""))
            if not text or text in seen:
                continue
            if len(text) > max_len:
                continue
            seen.add(text)
            row = {
                "language": language,
                "text": text,
                "text_length": len(text),
                "length_bucket": length_bucket(len(text)),
                "source": obj.get("source", "parallel_phrase"),
                "source_group": "parallel_corpus",
                "domain": obj.get("domain", "unknown"),
                "source_pool_id": obj.get("id", ""),
                "source_file": str(path),
                "source_row_id": obj.get("source_row_id", ""),
                "source_row_index": obj.get("source_row_index", ""),
                "phrase_index": obj.get("phrase_index", ""),
                "source_text": obj.get("source_text", ""),
            }
            if language == "kk":
                row["has_kazakh_specific"] = has_kazakh_specific(text)
            rows.append(row)
    return rows


def load_real_train_rows(path: Path, languages: set[str], args: argparse.Namespace) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {lang: [] for lang in languages}
    seen_by_lang: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        for obj in csv.DictReader(f):
            if obj.get("split") != "train":
                continue
            language = obj.get("language", "")
            if language not in languages:
                continue
            text = clean_text(obj.get("text") or obj.get("logical_text") or "")
            if not text or text in seen_by_lang[language]:
                continue
            if len(text) > max_len_for_language(language, args):
                continue
            seen_by_lang[language].add(text)
            row = {
                "language": language,
                "text": text,
                "text_length": len(text),
                "length_bucket": length_bucket(len(text)),
                "source": "real_train_meme_text",
                "source_group": "real_train_meme",
                "domain": "meme_train",
                "candidate_id": obj.get("candidate_id", ""),
                "source_id": obj.get("source_id", ""),
                "source_image": obj.get("source_image", ""),
                "source_file": str(path),
            }
            if language == "kk":
                row["has_kazakh_specific"] = has_kazakh_specific(text)
            out[language].append(row)
    return out


def source_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    norm = sum(ratios.values())
    if norm <= 0:
        raise ValueError("source ratios must sum to a positive value")
    raw = {name: total * ratio / norm for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def sample_rows(rows: list[dict[str, object]], count: int, rng: random.Random) -> list[dict[str, object]]:
    rows = list(rows)
    rng.shuffle(rows)
    if len(rows) < count:
        return rows
    return rows[:count]


def fill_deficit(
    selected: list[dict[str, object]],
    candidates: list[dict[str, object]],
    total: int,
    used_texts: set[str],
    rng: random.Random,
) -> list[dict[str, object]]:
    if len(selected) >= total:
        return selected[:total]
    leftovers = [row for row in candidates if row["text"] not in used_texts]
    rng.shuffle(leftovers)
    need = total - len(selected)
    selected.extend(leftovers[:need])
    return selected


def build_language_pool(
    language: str,
    parallel_rows: list[dict[str, object]],
    real_rows: list[dict[str, object]],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    counts = source_counts(
        args.samples_per_lang,
        {"parallel_corpus": args.parallel_ratio, "real_train_meme": args.real_ratio},
    )
    selected: list[dict[str, object]] = []
    used_texts: set[str] = set()

    for source_name, rows in [
        ("real_train_meme", real_rows),
        ("parallel_corpus", parallel_rows),
    ]:
        take = counts[source_name]
        for row in sample_rows(rows, take, rng):
            text = str(row["text"])
            if text in used_texts:
                continue
            selected.append(row)
            used_texts.add(text)

    all_candidates = list(real_rows) + list(parallel_rows)
    selected = fill_deficit(selected, all_candidates, args.samples_per_lang, used_texts, rng)
    rng.shuffle(selected)

    output_rows: list[dict[str, object]] = []
    for idx, row in enumerate(selected[: args.samples_per_lang], 1):
        out = dict(row)
        out["id"] = f"{language}_mixed_v1_{idx:08d}"
        out["seed"] = args.seed
        out["pool_version"] = "mixed_text_pool_v1"
        output_rows.append(out)

    summary = summarize(language, output_rows, parallel_rows, real_rows, counts)
    return output_rows, summary


def describe_lengths(values: list[int]) -> dict[str, int | None]:
    values = sorted(values)
    if not values:
        return {key: None for key in ["min", "p25", "median", "p75", "p90", "p95", "p99", "max"]}
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


def summarize(
    language: str,
    rows: list[dict[str, object]],
    parallel_rows: list[dict[str, object]],
    real_rows: list[dict[str, object]],
    target_source_counts: dict[str, int],
) -> dict[str, object]:
    source_counts_actual = Counter(str(row.get("source_group", "")) for row in rows)
    bucket_counts = Counter(str(row.get("length_bucket", "")) for row in rows)
    domain_counts = Counter(str(row.get("domain", "")) for row in rows)
    summary: dict[str, object] = {
        "language": language,
        "samples": len(rows),
        "target_source_counts": target_source_counts,
        "source_counts": dict(source_counts_actual),
        "source_ratios": {k: v / len(rows) for k, v in source_counts_actual.items()} if rows else {},
        "available": {
            "parallel_corpus": len(parallel_rows),
            "real_train_meme": len(real_rows),
        },
        "length": describe_lengths([int(row["text_length"]) for row in rows]),
        "length_bucket_counts": dict(bucket_counts),
        "length_bucket_ratios": {k: v / len(rows) for k, v in bucket_counts.items()} if rows else {},
        "domain_top20": domain_counts.most_common(20),
        "duplicates_in_output": len(rows) - len({str(row["text"]) for row in rows}),
    }
    if language == "kk":
        specific = sum(1 for row in rows if row.get("has_kazakh_specific"))
        summary["kazakh_specific_chars"] = {
            "chars": "".join(sorted(KAZAKH_SPECIFIC_CHARS)),
            "output_count": specific,
            "output_ratio": specific / len(rows) if rows else 0,
        }
    return summary


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    languages = ["zh", "ug", "kk"]
    real_rows_by_lang = load_real_train_rows(args.real_metadata, set(languages), args)

    summaries: dict[str, object] = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for language in languages:
        parallel_path = args.parallel_pool_dir / f"{language}_text_pool_50000.jsonl"
        parallel_rows = load_parallel_rows(parallel_path, language, max_len_for_language(language, args))
        rows, summary = build_language_pool(
            language=language,
            parallel_rows=parallel_rows,
            real_rows=real_rows_by_lang[language],
            args=args,
            rng=rng,
        )
        output_path = args.out_dir / f"{language}_mixed_text_pool_50000.jsonl"
        write_jsonl(output_path, rows)
        summary["input"] = {
            "parallel_pool": str(parallel_path),
            "real_metadata": str(args.real_metadata),
        }
        summary["output"] = str(output_path)
        summary["seed"] = args.seed
        summaries[language] = summary
        with (args.out_dir / f"{language}_mixed_text_pool_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    all_summary = {
        "pool_version": "mixed_text_pool_v1",
        "samples_per_lang": args.samples_per_lang,
        "parallel_ratio": args.parallel_ratio,
        "real_ratio": args.real_ratio,
        "languages": summaries,
    }
    with (args.out_dir / "mixed_text_pool_summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
