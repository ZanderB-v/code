import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


SPACE_RE = re.compile(r"\s+")
SPLIT_RE = re.compile(r"[\r\n\t。！？!?；;，、：:]+")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")


def normalize_text(text):
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in {"Cf", "Cc"} or ch in "\t\n\r"
    )
    return SPACE_RE.sub(" ", text).strip()


def strip_edge_punctuation(text):
    return text.strip(" \t\r\n\"'“”‘’《》〈〉()[]{}<>.,，。!?！？;；:：")


def han_ratio(text):
    if not text:
        return 0.0
    return len(HAN_RE.findall(text)) / len(text)


def split_chinese_text(text, max_len):
    pieces = []
    for piece in SPLIT_RE.split(text):
        piece = strip_edge_punctuation(normalize_text(piece))
        if not piece:
            continue
        if len(piece) <= max_len:
            pieces.append(piece)
            continue
        # Chinese has no explicit word boundary in many cases; chunk only after
        # punctuation has failed, preserving order and avoiding overlap.
        for start in range(0, len(piece), max_len):
            chunk = strip_edge_punctuation(piece[start:start + max_len])
            if chunk:
                pieces.append(chunk)
    return pieces


def length_bucket(length):
    if length <= 15:
        return "short"
    return "medium"


def parse_bucket_ratios(value):
    ratios = {}
    for part in value.split(","):
        if not part.strip():
            continue
        key, raw = part.split(":", 1)
        ratios[key.strip()] = float(raw)
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError(f"Invalid bucket ratios: {value}")
    return {key: val / total for key, val in ratios.items()}


def target_counts(total, ratios):
    raw = {key: total * val for key, val in ratios.items()}
    counts = {key: int(raw[key]) for key in raw}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def collect_candidates(input_paths, min_len, max_len, min_han_ratio):
    rows = []
    seen = set()
    stats = Counter()
    domain_total = Counter()
    domain_kept = Counter()

    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8-sig") as f:
            for row_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                stats["input_rows"] += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_json"] += 1
                    continue
                source = normalize_text(obj.get("source"))
                domain = obj.get("domain") or "unknown"
                domain_total[domain] += 1
                phrases = split_chinese_text(source, max_len)
                stats["candidate_phrases_before_filter"] += len(phrases)
                for phrase_idx, text in enumerate(phrases):
                    length = len(text)
                    if length < min_len:
                        stats["drop_too_short"] += 1
                        continue
                    if length > max_len:
                        stats["drop_too_long"] += 1
                        continue
                    if han_ratio(text) < min_han_ratio:
                        stats["drop_han_ratio"] += 1
                        continue
                    if text in seen:
                        stats["drop_duplicate"] += 1
                        continue
                    seen.add(text)
                    stats["kept_candidates"] += 1
                    domain_kept[domain] += 1
                    rows.append(
                        {
                            "text": text,
                            "language": "zh",
                            "text_length": length,
                            "length_bucket": length_bucket(length),
                            "source": "parallel_source_phrase",
                            "domain": domain,
                            "source_file": str(input_path),
                            "source_row_id": obj.get("id", ""),
                            "source_row_index": row_idx,
                            "phrase_index": phrase_idx,
                            "source_text": source,
                        }
                    )

    return rows, {
        "stats": dict(stats),
        "domain_total_top20": domain_total.most_common(20),
        "domain_kept_top20": domain_kept.most_common(20),
    }


def sample_pool(rows, total, ratios, rng):
    by_bucket = defaultdict(list)
    for row in rows:
        by_bucket[row["length_bucket"]].append(row)
    for bucket_rows in by_bucket.values():
        rng.shuffle(bucket_rows)

    desired = target_counts(total, ratios)
    selected = []
    deficit = 0
    for bucket in ("short", "medium"):
        want = desired.get(bucket, 0)
        items = by_bucket.get(bucket, [])
        take = min(want, len(items))
        selected.extend(items[:take])
        by_bucket[bucket] = items[take:]
        deficit += want - take

    if deficit > 0:
        leftovers = by_bucket.get("short", []) + by_bucket.get("medium", [])
        rng.shuffle(leftovers)
        selected.extend(leftovers[:deficit])
    return selected[:total]


def write_pool(path, rows, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, 1):
            out = dict(row)
            out["id"] = f"zh_pool_{idx:08d}"
            out["seed"] = seed
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def summarize(rows, available_rows, ratios, input_paths, output_path, seed, collect_summary):
    lengths = sorted(row["text_length"] for row in rows)
    buckets = Counter(row["length_bucket"] for row in rows)
    domains = Counter(row.get("domain", "unknown") for row in rows)
    return {
        "language": "zh",
        "samples": len(rows),
        "available_rows": len(available_rows),
        "target_bucket_ratios": ratios,
        "bucket_counts": dict(buckets),
        "bucket_ratios": {k: v / len(rows) for k, v in buckets.items()} if rows else {},
        "length": describe_lengths(lengths),
        "domain_top20": domains.most_common(20),
        "duplicates_in_output": len(rows) - len({row["text"] for row in rows}),
        "inputs": [str(path) for path in input_paths],
        "output": str(output_path),
        "seed": seed,
        **collect_summary,
    }


def describe_lengths(values):
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "p90": None, "p95": None, "p99": None, "max": None}
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


def percentile(values, q):
    return values[round((len(values) - 1) * q)] if values else None


def default_paths():
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    corpus_root = svtr_root / "02_corpus_preparation"
    return {
        "raw_dir": corpus_root / "raw_parallel",
        "out_dir": corpus_root / "synthetic_text_pool",
    }


def main():
    defaults = default_paths()
    parser = argparse.ArgumentParser(description="Build Chinese synthetic text pool from source side of parallel corpus.")
    parser.add_argument("--raw-dir", type=Path, default=defaults["raw_dir"])
    parser.add_argument("--out-dir", type=Path, default=defaults["out_dir"])
    parser.add_argument("--inputs", nargs="*", default=["zh_ug.jsonl", "zh_kk.jsonl"])
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=35)
    parser.add_argument("--min-han-ratio", type=float, default=0.50)
    parser.add_argument("--bucket-ratios", default="short:0.70,medium:0.30")
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    input_paths = [Path(x) if Path(x).is_absolute() else args.raw_dir / x for x in args.inputs]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    ratios = parse_bucket_ratios(args.bucket_ratios)
    rng = random.Random(args.seed)
    candidates, collect_summary = collect_candidates(input_paths, args.min_len, args.max_len, args.min_han_ratio)
    selected = sample_pool(candidates, args.samples, ratios, rng)
    rng.shuffle(selected)

    out_jsonl = args.out_dir / f"zh_text_pool_{args.samples}.jsonl"
    out_summary = args.out_dir / "zh_text_pool_summary.json"
    write_pool(out_jsonl, selected, args.seed)
    summary = summarize(selected, candidates, ratios, input_paths, out_jsonl, args.seed, collect_summary)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
