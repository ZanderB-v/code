import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


SPACE_RE = re.compile(r"\s+")
SPLIT_RE = re.compile(r"[\r\n\t。！？!?；;،，、؛：:]+")

UG_CHARS_RE = re.compile(r"[\u0600-\u06ff]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
KAZAKH_SPECIFIC_RE = re.compile(r"[ӘәҒғҚқҢңӨөҰұҮүҺһІі]")


def normalize_text(text):
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in {"Cf", "Cc"} or ch in "\t\n\r"
    )
    text = SPACE_RE.sub(" ", text).strip()
    return text


def strip_edge_punctuation(text):
    return text.strip(" \t\r\n\"'“”‘’«»()[]{}<>.,，。!?！？;；:：،؛؟")


def char_ratio(text, pattern):
    if not text:
        return 0.0
    return len(pattern.findall(text)) / len(text)


def valid_for_language(text, language, min_lang_ratio, require_kk_specific):
    if language == "ug":
        return char_ratio(text, UG_CHARS_RE) >= min_lang_ratio
    if language == "kk":
        if char_ratio(text, CYRILLIC_RE) < min_lang_ratio:
            return False
        if require_kk_specific and not KAZAKH_SPECIFIC_RE.search(text):
            return False
    return True


def split_on_punctuation(text):
    pieces = []
    for piece in SPLIT_RE.split(text):
        piece = strip_edge_punctuation(normalize_text(piece))
        if piece:
            pieces.append(piece)
    return pieces


def split_by_words(text, max_len):
    words = text.split()
    if len(words) <= 1:
        return [text]

    chunks = []
    current = []
    current_len = 0
    for word in words:
        word_len = len(word)
        if word_len > max_len:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            # Do not cut through a word. Drop this token later by length filter.
            chunks.append(word)
            continue
        next_len = word_len if not current else current_len + 1 + word_len
        if current and next_len > max_len:
            chunks.append(" ".join(current))
            current = [word]
            current_len = word_len
        else:
            current.append(word)
            current_len = next_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def split_target_to_phrases(text, max_len):
    phrases = []
    for piece in split_on_punctuation(text):
        if len(piece) <= max_len:
            phrases.append(piece)
        else:
            phrases.extend(split_by_words(piece, max_len))
    return [strip_edge_punctuation(normalize_text(p)) for p in phrases if strip_edge_punctuation(normalize_text(p))]


def length_bucket(length, min_len, max_len):
    if length < min_len:
        return f"too_short_lt_{min_len}"
    if length > max_len:
        return f"too_long_gt_{max_len}"
    if length <= 15:
        return "short_3_15"
    if length <= 35:
        return "medium_16_35"
    return f"long_36_{max_len}"


def process_file(
    input_path,
    language,
    output_path,
    min_len,
    max_len,
    min_lang_ratio,
    dedupe,
    require_kk_specific,
):
    stats = Counter()
    length_buckets = Counter()
    domain_total = Counter()
    domain_kept = Counter()
    seen = set()
    kept_lengths = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8-sig") as src, output_path.open("w", encoding="utf-8") as dst:
        for row_idx, line in enumerate(src, 1):
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
            target = normalize_text(obj.get("target"))
            domain = obj.get("domain") or "unknown"
            domain_total[domain] += 1

            phrases = split_target_to_phrases(target, max_len=max_len)
            stats["candidate_phrases_before_filter"] += len(phrases)

            for phrase_idx, phrase in enumerate(phrases):
                length = len(phrase)
                length_buckets[length_bucket(length, min_len, max_len)] += 1

                if length < min_len:
                    stats["drop_too_short"] += 1
                    continue
                if length > max_len:
                    stats["drop_too_long"] += 1
                    continue
                if not valid_for_language(phrase, language, min_lang_ratio, require_kk_specific):
                    stats["drop_language_ratio"] += 1
                    continue
                if dedupe and phrase in seen:
                    stats["drop_duplicate"] += 1
                    continue

                seen.add(phrase)
                kept_lengths.append(length)
                domain_kept[domain] += 1
                stats["kept"] += 1

                out = {
                    "id": f"{language}_{stats['kept']:08d}",
                    "language": language,
                    "source_file": str(input_path),
                    "source_row_id": obj.get("id", ""),
                    "source_row_index": row_idx,
                    "phrase_index": phrase_idx,
                    "source": source,
                    "target_full": target,
                    "text": phrase,
                    "text_length": length,
                    "domain": domain,
                    "source_lang": obj.get("source_lang", "zh"),
                    "target_lang": obj.get("target_lang", language),
                }
                dst.write(json.dumps(out, ensure_ascii=False) + "\n")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "language": language,
        "min_len": min_len,
        "max_len": max_len,
        "min_lang_ratio": min_lang_ratio,
        "dedupe": dedupe,
        "require_kk_specific": require_kk_specific,
        "stats": dict(stats),
        "length_buckets_before_final_filter": dict(length_buckets),
        "kept_length": describe_lengths(sorted(kept_lengths)),
        "domain_total_top20": domain_total.most_common(20),
        "domain_kept_top20": domain_kept.most_common(20),
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


def percentile(values, q):
    if not values:
        return None
    index = round((len(values) - 1) * q)
    return values[index]


def default_paths():
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    corpus_root = svtr_root / "02_corpus_preparation"
    return {
        "ug": corpus_root / "raw_parallel" / "zh_ug.jsonl",
        "kk": corpus_root / "raw_parallel" / "zh_kk.jsonl",
        "out_dir": corpus_root / "filtered_parallel_phrases",
        "summary_dir": corpus_root / "parallel_corpus_stats",
    }


def main():
    defaults = default_paths()
    parser = argparse.ArgumentParser(description="Split long zh-ug/zh-kk parallel corpus targets into OCR-line-length phrases.")
    parser.add_argument("--ug", type=Path, default=defaults["ug"])
    parser.add_argument("--kk", type=Path, default=defaults["kk"])
    parser.add_argument("--out-dir", type=Path, default=defaults["out_dir"])
    parser.add_argument("--summary-dir", type=Path, default=defaults["summary_dir"])
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=45)
    parser.add_argument("--min-lang-ratio", type=float, default=0.60)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--require-kk-specific", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for language, input_path in (("ug", args.ug), ("kk", args.kk)):
        if not input_path.exists():
            raise FileNotFoundError(f"{language} corpus not found: {input_path}")
        output_path = args.out_dir / f"{language}_parallel_phrases_len{args.min_len}_{args.max_len}.jsonl"
        results[language] = process_file(
            input_path=input_path,
            language=language,
            output_path=output_path,
            min_len=args.min_len,
            max_len=args.max_len,
            min_lang_ratio=args.min_lang_ratio,
            dedupe=not args.no_dedupe,
            require_kk_specific=args.require_kk_specific if language == "kk" else False,
        )

        summary_path = args.summary_dir / f"{language}_parallel_phrases_len{args.min_len}_{args.max_len}_summary.json"
        summary_path.write_text(json.dumps(results[language], ensure_ascii=False, indent=2), encoding="utf-8")

    all_summary_path = args.summary_dir / f"parallel_phrases_len{args.min_len}_{args.max_len}_summary_all.json"
    all_summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
