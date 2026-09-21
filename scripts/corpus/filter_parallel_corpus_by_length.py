import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


SPACE_RE = re.compile(r"\s+")


def normalize_text(text):
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in {"Cf", "Cc"} or ch in "\t\n\r"
    )
    return SPACE_RE.sub(" ", text).strip()


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    index = round((len(sorted_values) - 1) * q)
    return sorted_values[index]


def length_bucket(length, min_len, max_len):
    if length == 0:
        return "empty"
    if length < min_len:
        return f"too_short_lt_{min_len}"
    if length > max_len:
        return f"too_long_gt_{max_len}"
    if length <= 15:
        return "short_3_15"
    if length <= 35:
        return "medium_16_35"
    return f"long_36_{max_len}"


def analyze_file(input_path, language, min_len, max_len, output_jsonl=None):
    total = 0
    bad_json = 0
    kept = 0
    unique_kept = set()
    all_lengths = []
    kept_lengths = []
    buckets = Counter()
    domain_total = Counter()
    domain_kept = Counter()

    writer = None
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        writer = output_jsonl.open("w", encoding="utf-8")

    with input_path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            source = normalize_text(obj.get("source"))
            target = normalize_text(obj.get("target"))
            domain = obj.get("domain") or "unknown"
            length = len(target)

            all_lengths.append(length)
            buckets[length_bucket(length, min_len, max_len)] += 1
            domain_total[domain] += 1

            if min_len <= length <= max_len:
                kept += 1
                kept_lengths.append(length)
                unique_kept.add(target)
                domain_kept[domain] += 1
                if writer is not None:
                    out = dict(obj)
                    out["source"] = source
                    out["target"] = target
                    out["language"] = language
                    out["target_length"] = length
                    writer.write(json.dumps(out, ensure_ascii=False) + "\n")

    if writer is not None:
        writer.close()

    all_lengths = sorted(all_lengths)
    kept_lengths = sorted(kept_lengths)
    return {
        "input": str(input_path),
        "language": language,
        "min_len": min_len,
        "max_len": max_len,
        "total_rows": total,
        "bad_json": bad_json,
        "kept": kept,
        "kept_ratio": kept / total if total else 0,
        "unique_kept": len(unique_kept),
        "duplicate_kept_removed": kept - len(unique_kept),
        "target_length": describe_lengths(all_lengths),
        "kept_target_length": describe_lengths(kept_lengths),
        "buckets": dict(buckets),
        "domain_total_top20": domain_total.most_common(20),
        "domain_kept_top20": domain_kept.most_common(20),
        "output_jsonl": str(output_jsonl) if output_jsonl else "",
    }


def describe_lengths(values):
    if not values:
        return {
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
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


def default_paths():
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    corpus_root = svtr_root / "02_corpus_preparation"
    return {
        "ug": corpus_root / "raw_parallel" / "zh_ug.jsonl",
        "kk": corpus_root / "raw_parallel" / "zh_kk.jsonl",
        "out_dir": corpus_root / "parallel_corpus_stats",
    }


def main():
    defaults = default_paths()
    parser = argparse.ArgumentParser(description="Filter zh-ug/zh-kk parallel corpus by target-side text length.")
    parser.add_argument("--ug", type=Path, default=defaults["ug"])
    parser.add_argument("--kk", type=Path, default=defaults["kk"])
    parser.add_argument("--out-dir", type=Path, default=defaults["out_dir"])
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=45)
    parser.add_argument("--write-filtered", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for language, input_path in (("ug", args.ug), ("kk", args.kk)):
        if not input_path.exists():
            raise FileNotFoundError(f"{language} corpus not found: {input_path}")
        output_jsonl = None
        if args.write_filtered:
            filtered_dir = args.out_dir.parent / "filtered_parallel"
            output_jsonl = filtered_dir / f"{language}_parallel_len{args.min_len}_{args.max_len}.jsonl"
        summary = analyze_file(input_path, language, args.min_len, args.max_len, output_jsonl)
        results[language] = summary
        summary_path = args.out_dir / f"{language}_parallel_len{args.min_len}_{args.max_len}_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    all_path = args.out_dir / f"parallel_len{args.min_len}_{args.max_len}_summary_all.json"
    all_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
