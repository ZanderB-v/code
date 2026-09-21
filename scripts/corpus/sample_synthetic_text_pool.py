import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


KAZAKH_SPECIFIC_CHARS = set("ӘәҒғҚқҢңӨөҰұҮүҺһІі")


def text_length_bucket(length):
    if length <= 15:
        return "short"
    if length <= 35:
        return "medium"
    return "long"


def has_kazakh_specific(text):
    return any(ch in KAZAKH_SPECIFIC_CHARS for ch in text)


def load_phrase_rows(path):
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = (obj.get("text") or obj.get("target") or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            length = int(obj.get("text_length") or len(text))
            row = {
                "text": text,
                "language": obj.get("language") or obj.get("target_lang") or "",
                "text_length": length,
                "length_bucket": text_length_bucket(length),
                "source": "parallel_phrase",
                "domain": obj.get("domain") or "unknown",
                "source_file": obj.get("source_file") or str(path),
                "source_row_id": obj.get("source_row_id") or "",
                "source_row_index": obj.get("source_row_index") or "",
                "phrase_index": obj.get("phrase_index") or "",
                "target_full": obj.get("target_full") or "",
                "source_text": obj.get("source") or "",
            }
            if row["language"] == "kk":
                row["has_kazakh_specific"] = has_kazakh_specific(text)
            rows.append(row)
    return rows


def parse_bucket_ratios(value):
    ratios = {}
    for part in value.split(","):
        if not part.strip():
            continue
        name, raw = part.split(":", 1)
        ratios[name.strip()] = float(raw)
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError(f"Invalid bucket ratios: {value}")
    return {name: ratio / total for name, ratio in ratios.items()}


def target_counts(total, ratios):
    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(raw[name]) for name in raw}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def sample_by_bucket(rows, total, ratios, rng):
    by_bucket = defaultdict(list)
    for row in rows:
        by_bucket[row["length_bucket"]].append(row)
    for items in by_bucket.values():
        rng.shuffle(items)

    desired = target_counts(total, ratios)
    selected = []
    deficits = 0
    for bucket in ("short", "medium", "long"):
        want = desired.get(bucket, 0)
        items = by_bucket.get(bucket, [])
        take = min(want, len(items))
        selected.extend(items[:take])
        by_bucket[bucket] = items[take:]
        deficits += want - take

    if deficits > 0:
        leftovers = []
        for bucket in ("medium", "short", "long"):
            leftovers.extend(by_bucket.get(bucket, []))
        rng.shuffle(leftovers)
        selected.extend(leftovers[:deficits])

    return selected[:total]


def improve_kk_specific_ratio(selected, all_rows, min_ratio, rng):
    if not selected:
        return selected, {"requested_min_ratio": min_ratio, "replaced": 0}

    selected_texts = {row["text"] for row in selected}
    current = sum(1 for row in selected if row.get("has_kazakh_specific"))
    target = int(round(len(selected) * min_ratio))
    needed = max(0, target - current)
    if needed == 0:
        return selected, {"requested_min_ratio": min_ratio, "replaced": 0}

    candidate_add = [
        row for row in all_rows
        if row.get("has_kazakh_specific") and row["text"] not in selected_texts
    ]
    candidate_drop = [
        idx for idx, row in enumerate(selected)
        if not row.get("has_kazakh_specific")
    ]
    rng.shuffle(candidate_add)
    rng.shuffle(candidate_drop)
    replace_n = min(needed, len(candidate_add), len(candidate_drop))

    selected = list(selected)
    for add_row, drop_idx in zip(candidate_add[:replace_n], candidate_drop[:replace_n]):
        selected[drop_idx] = add_row

    return selected, {
        "requested_min_ratio": min_ratio,
        "target_specific_count": target,
        "initial_specific_count": current,
        "replaced": replace_n,
    }


def write_pool(path, rows, language, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, 1):
            out = dict(row)
            out["id"] = f"{language}_pool_{idx:08d}"
            out["seed"] = seed
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def summarize(rows, all_rows, language, bucket_ratios, kk_specific_adjustment=None):
    lengths = sorted(row["text_length"] for row in rows)
    bucket_counts = Counter(row["length_bucket"] for row in rows)
    domain_counts = Counter(row.get("domain", "unknown") for row in rows)
    summary = {
        "language": language,
        "samples": len(rows),
        "available_rows": len(all_rows),
        "target_bucket_ratios": bucket_ratios,
        "bucket_counts": dict(bucket_counts),
        "bucket_ratios": {k: v / len(rows) for k, v in bucket_counts.items()} if rows else {},
        "length": describe_lengths(lengths),
        "domain_top20": domain_counts.most_common(20),
        "duplicates_in_output": len(rows) - len({row["text"] for row in rows}),
    }
    if language == "kk":
        specific_count = sum(1 for row in rows if row.get("has_kazakh_specific"))
        available_specific = sum(1 for row in all_rows if row.get("has_kazakh_specific"))
        summary["kazakh_specific_chars"] = {
            "output_count": specific_count,
            "output_ratio": specific_count / len(rows) if rows else 0,
            "available_count": available_specific,
            "available_ratio": available_specific / len(all_rows) if all_rows else 0,
            "chars": "".join(sorted(KAZAKH_SPECIFIC_CHARS)),
            "adjustment": kk_specific_adjustment or {},
        }
    return summary


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
    if not values:
        return None
    return values[round((len(values) - 1) * q)]


def default_paths():
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    corpus_root = svtr_root / "02_corpus_preparation"
    return {
        "ug": corpus_root / "filtered_parallel_phrases" / "ug_parallel_phrases_len3_45.jsonl",
        "kk": corpus_root / "filtered_parallel_phrases" / "kk_parallel_phrases_len3_45.jsonl",
        "out_dir": corpus_root / "synthetic_text_pool",
    }


def main():
    defaults = default_paths()
    parser = argparse.ArgumentParser(description="Sample balanced synthetic text pools from filtered parallel phrase corpus.")
    parser.add_argument("--ug", type=Path, default=defaults["ug"])
    parser.add_argument("--kk", type=Path, default=defaults["kk"])
    parser.add_argument("--out-dir", type=Path, default=defaults["out_dir"])
    parser.add_argument("--samples-per-lang", type=int, default=50000)
    parser.add_argument("--bucket-ratios", default="short:0.30,medium:0.50,long:0.20")
    parser.add_argument("--kk-min-specific-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ratios = parse_bucket_ratios(args.bucket_ratios)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for language, input_path in (("ug", args.ug), ("kk", args.kk)):
        if not input_path.exists():
            raise FileNotFoundError(f"{language} phrase corpus not found: {input_path}")
        rows = load_phrase_rows(input_path)
        rows = [row for row in rows if row["language"] in ("", language)]
        selected = sample_by_bucket(rows, args.samples_per_lang, ratios, rng)

        kk_adjustment = None
        if language == "kk":
            selected, kk_adjustment = improve_kk_specific_ratio(
                selected,
                rows,
                args.kk_min_specific_ratio,
                rng,
            )

        rng.shuffle(selected)
        out_jsonl = args.out_dir / f"{language}_text_pool_{args.samples_per_lang}.jsonl"
        summary_path = args.out_dir / f"{language}_text_pool_summary.json"
        write_pool(out_jsonl, selected, language, args.seed)
        summary = summarize(selected, rows, language, ratios, kk_adjustment)
        summary["input"] = str(input_path)
        summary["output"] = str(out_jsonl)
        summary["seed"] = args.seed
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        all_summaries[language] = summary

    all_path = args.out_dir / "text_pool_summary_all.json"
    all_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
