#!/usr/bin/env python3
"""Validate five formal shards and build deterministic S10/S25/S50 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


LANGUAGES = ("zh", "ug", "kk")
SUBSET_SIZES = {"s10": 10000, "s25": 25000, "s50": 50000}
KK_SPECIFIC = set("ӘәҒғҚқҢңӨөҰұҮүҺһІі")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/data_home/wudayu/experiments/multilingual_meme_ocr/"
            "svtrv2_line_recognition"
        ),
    )
    parser.add_argument("--formal-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--exclude-target-eval-texts",
        action="store_true",
        help="Remove exact normalized dev/test label matches from S50.",
    )
    parser.add_argument(
        "--repair-metadata",
        type=Path,
        nargs="*",
        default=[],
        help="Postprocessed replacement metadata JSONL files.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.replace("\u00a0", " ").split())


def near_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", normalize_text(text)).casefold()
    return "".join(
        char
        for char in text
        if not unicodedata.category(char).startswith(("P", "Z"))
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    formal_dir = (
        args.formal_dir.resolve()
        if args.formal_dir
        else root / "03_synthetic_generation" / "synthetic_formal_v2"
    )
    dictionary_path = (
        root
        / "04_model_training"
        / "character_dict_hz_ug_kk_v1"
        / "character_dict.txt"
    )
    dictionary = {
        line.rstrip("\r\n")
        for line in dictionary_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.rstrip("\r\n")
    }

    all_rows: list[dict] = []
    shard_summaries = {}
    for shard_index in range(5):
        shard_name = f"synthetic_shard_{shard_index:04d}_parallel"
        shard_dir = formal_dir / shard_name
        metadata_path = shard_dir / "metadata.jsonl"
        summary_path = shard_dir / "merged_summary.json"
        check_path = shard_dir / "merge_check_errors.json"
        if not metadata_path.is_file() or not summary_path.is_file():
            raise SystemExit(f"Incomplete formal shard: {shard_dir}")
        check_errors = (
            json.loads(check_path.read_text(encoding="utf-8-sig"))
            if check_path.is_file()
            else ["missing merge_check_errors.json"]
        )
        if check_errors:
            raise SystemExit(
                f"Shard validation failed for {shard_name}: {check_errors}"
            )
        rows = read_jsonl(metadata_path)
        counts = Counter(row.get("language") for row in rows)
        if dict(counts) != {"zh": 10000, "ug": 10000, "kk": 10000}:
            raise SystemExit(
                f"Unexpected counts in {shard_name}: {dict(counts)}"
            )
        for row in rows:
            normalized = dict(row)
            image = (row.get("image") or "").replace("\\", "/")
            normalized["formal_shard"] = shard_name
            normalized["formal_image"] = f"{shard_name}/{image}"
            normalized["image"] = normalized["formal_image"]
            all_rows.append(normalized)
        shard_summaries[shard_name] = {
            "rows": len(rows),
            "counts": dict(counts),
            "metadata_sha256": sha256_file(metadata_path),
        }

    repair_summary = {
        "enabled": args.exclude_target_eval_texts,
        "removed": {},
        "added": {},
        "repair_metadata": [str(path.resolve()) for path in args.repair_metadata],
    }
    if args.exclude_target_eval_texts:
        target_path = (
            root
            / "01_data_preparation"
            / "real_line_dataset_eval_reviewed"
            / "metadata.jsonl"
        )
        forbidden: dict[str, set[str]] = defaultdict(set)
        forbidden_near: dict[str, set[str]] = defaultdict(set)
        for row in read_jsonl(target_path):
            if row.get("split") not in ("dev", "test"):
                continue
            language = str(row.get("language") or "")
            logical = normalize_text(
                str(row.get("logical_text") or row.get("text") or "")
            )
            forbidden[language].add(logical)
            forbidden_near[language].add(near_key(logical))

        kept_rows = []
        removed = Counter()
        for row in all_rows:
            language = str(row.get("language") or "")
            logical = normalize_text(str(row.get("logical_text") or ""))
            if (
                logical in forbidden[language]
                or near_key(logical) in forbidden_near[language]
            ):
                removed[language] += 1
            else:
                kept_rows.append(row)
        all_rows = kept_rows

        repair_candidates: dict[str, list[dict]] = defaultdict(list)
        for metadata_path in args.repair_metadata:
            metadata_path = metadata_path.resolve()
            for row in read_jsonl(metadata_path):
                language = str(row.get("language") or "")
                logical = normalize_text(str(row.get("logical_text") or ""))
                if language not in LANGUAGES:
                    raise SystemExit(
                        f"Invalid repair language in {metadata_path}: {language}"
                    )
                if (
                    logical in forbidden[language]
                    or near_key(logical) in forbidden_near[language]
                ):
                    raise SystemExit(
                        f"Repair text still overlaps target eval: "
                        f"{row.get('id')}"
                    )
                image = (row.get("image") or "").replace("\\", "/")
                image_path = metadata_path.parent / image
                if not image_path.is_file():
                    raise SystemExit(f"Missing repair image: {image_path}")
                try:
                    formal_image = image_path.relative_to(formal_dir).as_posix()
                except ValueError as exc:
                    raise SystemExit(
                        f"Repair data must live under {formal_dir}: {image_path}"
                    ) from exc
                normalized = dict(row)
                normalized["formal_shard"] = "leakage_repair_v1"
                normalized["formal_image"] = formal_image
                normalized["image"] = formal_image
                normalized["data_repair"] = "exclude_target_eval_text_v1"
                repair_candidates[language].append(normalized)

        existing_texts: dict[str, Counter] = {
            language: Counter(
                normalize_text(str(row.get("logical_text") or ""))
                for row in all_rows
                if row.get("language") == language
            )
            for language in LANGUAGES
        }
        added = Counter()
        for language in LANGUAGES:
            need = removed[language]
            for row in repair_candidates[language]:
                if added[language] >= need:
                    break
                logical = normalize_text(str(row.get("logical_text") or ""))
                if existing_texts[language][logical] >= 2:
                    continue
                existing_texts[language][logical] += 1
                all_rows.append(row)
                added[language] += 1
            if added[language] != need:
                raise SystemExit(
                    f"Insufficient {language} repair rows: "
                    f"need {need}, added {added[language]}"
                )
        repair_summary.update(
            {
                "target_metadata": str(target_path),
                "removed": dict(removed),
                "added": dict(added),
                "exact_eval_text_overlap_after": 0,
                "normalized_near_eval_text_overlap_after": 0,
            }
        )

    ids = [row.get("id") for row in all_rows]
    if len(ids) != len(set(ids)):
        duplicates = [
            item
            for item, count in Counter(ids).items()
            if count > 1
        ][:20]
        raise SystemExit(f"Duplicate IDs across shards: {duplicates}")

    missing_images = []
    unknown_chars = Counter()
    texts_by_language: dict[str, Counter] = {
        language: Counter() for language in LANGUAGES
    }
    image_hashes: dict[str, list[str]] = defaultdict(list)
    image_jobs = []
    for row in all_rows:
        language = row.get("language") or ""
        logical = row.get("logical_text") or ""
        ctc = row.get("ctc_text") or ""
        image = row.get("image") or ""
        image_path = formal_dir / image
        if not image_path.is_file():
            missing_images.append(image)
            continue
        image_jobs.append((image, image_path))
        texts_by_language[language][logical] += 1
        for char in ctc:
            if char != " " and char not in dictionary:
                unknown_chars[char] += 1

    def hash_job(item: tuple[str, Path]) -> tuple[str, str]:
        image, image_path = item
        return image, sha256_file(image_path)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for image, digest in executor.map(hash_job, image_jobs):
            image_hashes[digest].append(image)

    duplicate_image_hashes = {
        digest: values
        for digest, values in image_hashes.items()
        if len(values) > 1
    }
    if missing_images:
        raise SystemExit(f"Missing formal images: {len(missing_images)}")
    if unknown_chars:
        raise SystemExit(
            f"Unknown dictionary characters: {dict(unknown_chars)}"
        )
    if duplicate_image_hashes:
        examples = dict(list(duplicate_image_hashes.items())[:10])
        raise SystemExit(f"Duplicate image bytes across shards: {examples}")

    max_text_occurrence = {
        language: max(counter.values(), default=0)
        for language, counter in texts_by_language.items()
    }
    if any(value > 2 for value in max_text_occurrence.values()):
        raise SystemExit(
            f"Text occurrence exceeds 2: {max_text_occurrence}"
        )

    rows_by_language = {
        language: [
            row for row in all_rows if row.get("language") == language
        ]
        for language in LANGUAGES
    }
    for language, rows in rows_by_language.items():
        if len(rows) != 50000:
            raise SystemExit(f"{language} has {len(rows)} rows, expected 50000")

    subsets_root = formal_dir / "subsets"
    if subsets_root.exists() and not args.replace:
        raise SystemExit(
            f"Subset directory exists: {subsets_root}. Use --replace."
        )
    subsets_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "synthetic_formal_v2",
        "formal_dir": str(formal_dir),
        "shards": shard_summaries,
        "total_counts": {
            language: len(rows_by_language[language])
            for language in LANGUAGES
        },
        "max_text_occurrence_per_language": max_text_occurrence,
        "global_duplicate_image_hashes": 0,
        "dictionary_unknown_characters": 0,
        "data_repair": repair_summary,
        "subsets": {},
    }

    previous_ids: set[str] = set()
    for subset_name, per_language in SUBSET_SIZES.items():
        subset_dir = subsets_root / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)
        subset_rows = []
        for language in LANGUAGES:
            subset_rows.extend(rows_by_language[language][:per_language])
        subset_rows.sort(
            key=lambda row: (
                LANGUAGES.index(row["language"]),
                row["formal_shard"],
                row["id"],
            )
        )
        current_ids = {row["id"] for row in subset_rows}
        if previous_ids and not previous_ids.issubset(current_ids):
            raise SystemExit(f"Nested subset invariant failed at {subset_name}")
        previous_ids = current_ids

        metadata_path = subset_dir / "metadata.jsonl"
        write_jsonl(metadata_path, subset_rows)
        labels_dir = subset_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        all_lines = []
        for language in LANGUAGES:
            lang_rows = [
                row for row in subset_rows if row["language"] == language
            ]
            lines = [
                f"{row['image']}\t{row['ctc_text']}" for row in lang_rows
            ]
            (labels_dir / f"train_{language}.txt").write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )
            all_lines.extend(lines)
        (labels_dir / "train_all.txt").write_text(
            "\n".join(all_lines) + "\n",
            encoding="utf-8",
        )

        kk_rows = [
            row for row in subset_rows if row["language"] == "kk"
        ]
        ug_rows = [
            row for row in subset_rows if row["language"] == "ug"
        ]
        subset_summary = {
            "per_language": per_language,
            "total": len(subset_rows),
            "counts": dict(Counter(row["language"] for row in subset_rows)),
            "metadata_sha256": sha256_file(metadata_path),
            "unique_ids": len(current_ids),
            "unique_texts": {
                language: len(
                    {
                        row["logical_text"]
                        for row in subset_rows
                        if row["language"] == language
                    }
                )
                for language in LANGUAGES
            },
            "kk_specific_ratio": sum(
                any(char in KK_SPECIFIC for char in row["logical_text"])
                for row in kk_rows
            )
            / max(1, len(kk_rows)),
            "ug_u2_python_bidi_ratio": sum(
                row.get("ctc_text_u2_status") == "python_bidi"
                and row.get("ctc_text") == row.get("ctc_text_u2")
                for row in ug_rows
            )
            / max(1, len(ug_rows)),
            "nested_parent": (
                None
                if subset_name == "s10"
                else ("s10" if subset_name == "s25" else "s25")
            ),
        }
        (subset_dir / "summary.json").write_text(
            json.dumps(subset_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["subsets"][subset_name] = subset_summary

    (subsets_root / "s50_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
