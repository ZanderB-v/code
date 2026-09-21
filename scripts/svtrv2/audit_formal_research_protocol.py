#!/usr/bin/env python3
"""Audit formal data for leakage and direction-protocol consistency."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LANGUAGES = ("zh", "ug", "kk")
EVAL_SPLITS = ("dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}") from exc
    return rows


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    target_path = (
        root
        / "01_data_preparation"
        / "real_line_dataset_eval_reviewed"
        / "metadata.jsonl"
    )
    subsets_root = (
        root
        / "03_synthetic_generation"
        / "synthetic_formal_v2"
        / "subsets"
    )
    dictionary_path = (
        root
        / "04_model_training"
        / "character_dict_hz_ug_kk_v1"
        / "character_dict.txt"
    )
    frozen_root = root / "00_docs" / "frozen_protocol_v2"
    target_hash_path = frozen_root / "target_image_hashes.jsonl"

    for path in (
        target_path,
        dictionary_path,
        target_hash_path,
        *(subsets_root / scale / "metadata.jsonl" for scale in ("s10", "s25", "s50")),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    try:
        from bidi.algorithm import get_display  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Formal U2 audit requires python-bidi in openocr_svtrv2"
        ) from exc

    errors: list[str] = []
    warnings: list[str] = []
    target_rows = read_jsonl(target_path)
    target_texts: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    target_near: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    target_by_image: dict[str, dict[str, str]] = {}
    source_splits: dict[str, set[str]] = defaultdict(set)
    target_ids: set[str] = set()

    for row in target_rows:
        row_id = str(row.get("id") or "")
        language = str(row.get("language") or "")
        split = str(row.get("split") or "")
        source_id = str(row.get("source_id") or "")
        image = str(row.get("image") or "").replace("\\", "/")
        logical = normalize_text(
            str(row.get("logical_text") or row.get("text") or "")
        )
        if row_id in target_ids:
            errors.append(f"Duplicate target id: {row_id}")
        target_ids.add(row_id)
        target_texts[language][split].add(logical)
        target_near[language][split].add(near_key(logical))
        target_by_image[image] = {
            "id": row_id,
            "split": split,
            "language": language,
            "logical_text": logical,
        }
        source_splits[source_id].add(split)

    leaked_sources = {
        source_id: sorted(splits)
        for source_id, splits in source_splits.items()
        if source_id and len(splits) > 1
    }
    if leaked_sources:
        errors.append(
            "Target source_id leakage across splits: "
            + json.dumps(
                dict(list(leaked_sources.items())[:10]),
                ensure_ascii=False,
            )
        )

    target_text_overlap: dict[str, dict[str, Any]] = {}
    for language in LANGUAGES:
        train_texts = target_texts[language]["train"]
        target_text_overlap[language] = {}
        for split in EVAL_SPLITS:
            overlap = sorted(train_texts & target_texts[language][split])
            target_text_overlap[language][split] = {
                "unique_overlap_count": len(overlap),
                "examples": overlap[:20],
            }
            if overlap:
                warnings.append(
                    f"Target {language} train shares {len(overlap)} exact "
                    f"text strings with target {split}; source/image split "
                    "is clean, so this is disclosed rather than removed."
                )

    hashes_by_digest: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_jsonl(target_hash_path):
        image = str(row.get("image") or "").replace("\\", "/")
        digest = str(row.get("sha256") or "")
        target = target_by_image.get(image)
        if digest and target:
            hashes_by_digest[digest].append(target)
    cross_split_hashes = [
        {"sha256": digest, "rows": rows}
        for digest, rows in hashes_by_digest.items()
        if len({row["split"] for row in rows}) > 1
    ]
    if cross_split_hashes:
        errors.append(
            f"Identical target image bytes cross splits: "
            f"{len(cross_split_hashes)} groups"
        )

    subset_rows = {
        scale: read_jsonl(subsets_root / scale / "metadata.jsonl")
        for scale in ("s10", "s25", "s50")
    }
    subset_ids = {
        scale: {str(row.get("id") or "") for row in rows}
        for scale, rows in subset_rows.items()
    }
    if not subset_ids["s10"].issubset(subset_ids["s25"]):
        errors.append("S10 is not an ID subset of S25")
    if not subset_ids["s25"].issubset(subset_ids["s50"]):
        errors.append("S25 is not an ID subset of S50")

    synthetic_texts: dict[str, set[str]] = defaultdict(set)
    synthetic_near: dict[str, set[str]] = defaultdict(set)
    synthetic_chars: set[str] = set()
    synthetic_counts = Counter()
    u2_metadata_mismatches: list[dict[str, str]] = []
    for row in subset_rows["s50"]:
        language = str(row.get("language") or "")
        logical = normalize_text(str(row.get("logical_text") or ""))
        synthetic_counts[language] += 1
        synthetic_texts[language].add(logical)
        synthetic_near[language].add(near_key(logical))
        synthetic_chars.update(char for char in logical if not char.isspace())
        if language == "ug":
            expected_u2 = normalize_text(get_display(logical, base_dir="R"))
            actual_u2 = normalize_text(str(row.get("ctc_text") or ""))
            if expected_u2 != actual_u2:
                u2_metadata_mismatches.append(
                    {
                        "id": str(row.get("id") or ""),
                        "logical": logical,
                        "expected_u2": expected_u2,
                        "actual_u2": actual_u2,
                    }
                )
    if u2_metadata_mismatches:
        errors.append(
            f"S50 stored U2 differs from python-bidi: "
            f"{len(u2_metadata_mismatches)} rows"
        )

    overlap_summary: dict[str, dict[str, Any]] = {}
    for language in LANGUAGES:
        overlap_summary[language] = {}
        for split in EVAL_SPLITS:
            exact = sorted(
                synthetic_texts[language] & target_texts[language][split]
            )
            near = synthetic_near[language] & target_near[language][split]
            near_only_count = max(0, len(near) - len(exact))
            overlap_summary[language][split] = {
                "exact_count": len(exact),
                "exact_examples": exact[:20],
                "near_only_count": near_only_count,
            }
            if exact:
                errors.append(
                    f"S50 {language} text overlaps target {split}: "
                    f"{len(exact)} unique strings"
                )
            if near_only_count:
                errors.append(
                    f"S50 {language} has {near_only_count} normalized-near "
                    f"text overlaps with target {split}"
                )

    dictionary_chars = {
        line
        for line in dictionary_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line
    }
    target_train_chars = {
        char
        for language in LANGUAGES
        for text in target_texts[language]["train"]
        for char in text
        if not char.isspace()
    }
    train_observed_chars = target_train_chars | synthetic_chars
    dictionary_provenance: dict[str, Any] = {
        "protocol": "closed_vocabulary",
        "dictionary_size": len(dictionary_chars),
        "target_train_observed_characters": len(target_train_chars),
        "synthetic_s50_observed_characters": len(synthetic_chars),
        "combined_train_observed_characters": len(train_observed_chars),
        "splits": {},
    }
    for split in EVAL_SPLITS:
        eval_chars = {
            char
            for language in LANGUAGES
            for text in target_texts[language][split]
            for char in text
            if not char.isspace()
        }
        missing_from_dictionary = sorted(eval_chars - dictionary_chars)
        eval_only = sorted(eval_chars - train_observed_chars)
        dictionary_provenance["splits"][split] = {
            "observed_characters": len(eval_chars),
            "missing_from_dictionary_count": len(missing_from_dictionary),
            "missing_from_dictionary_codepoints": [
                f"U+{ord(char):04X}" for char in missing_from_dictionary
            ],
            "not_observed_in_target_train_or_s50_count": len(eval_only),
            "not_observed_in_target_train_or_s50_codepoints": [
                f"U+{ord(char):04X}" for char in eval_only
            ],
        }
        if missing_from_dictionary:
            errors.append(
                f"Character dictionary cannot represent "
                f"{len(missing_from_dictionary)} characters observed in "
                f"target {split}"
            )
        if eval_only:
            warnings.append(
                f"Closed-vocabulary dictionary contains {len(eval_only)} "
                f"target {split} characters not observed in target train "
                "or S50. Report this closed-vocabulary protocol and do not "
                "claim open-vocabulary recognition."
            )

    u2_roundtrip: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        mismatches = []
        total = 0
        for row in target_rows:
            if row.get("language") != "ug" or row.get("split") != split:
                continue
            total += 1
            logical = normalize_text(
                str(row.get("logical_text") or row.get("text") or "")
            )
            visual = normalize_text(get_display(logical, base_dir="R"))
            recovered = normalize_text(get_display(visual, base_dir="R"))
            if recovered != logical:
                mismatches.append(
                    {
                        "id": str(row.get("id") or ""),
                        "logical": logical,
                        "visual": visual,
                        "recovered": recovered,
                    }
                )
        u2_roundtrip[split] = {
            "rows": total,
            "mismatches": len(mismatches),
            "examples": mismatches[:20],
        }
        if split == "dev" and mismatches:
            errors.append(
                f"U2 get_display-twice recovery is not exact on target "
                f"{split}: {len(mismatches)}/{total}"
            )
        elif split == "test" and mismatches:
            warnings.append(
                f"U2 get_display-twice recovery is not exact on target "
                f"test: {len(mismatches)}/{total}. Final reporting must "
                "separate visual-order CER from heuristic logical output."
            )

    result = {
        "status": "passed" if not errors else "failed",
        "audit": "formal_research_protocol_v1",
        "root": str(root),
        "errors": errors,
        "warnings": warnings,
        "target_rows": len(target_rows),
        "synthetic_s50_counts": dict(synthetic_counts),
        "target_source_split_leaks": len(leaked_sources),
        "target_train_eval_text_overlap": target_text_overlap,
        "target_cross_split_image_hash_groups": {
            "count": len(cross_split_hashes),
            "examples": cross_split_hashes[:10],
        },
        "synthetic_target_text_overlap": overlap_summary,
        "character_dictionary_provenance": dictionary_provenance,
        "u2_metadata_mismatches": {
            "count": len(u2_metadata_mismatches),
            "examples": u2_metadata_mismatches[:20],
        },
        "u2_get_display_twice_roundtrip": u2_roundtrip,
    }
    return result


def main() -> None:
    args = parse_args()
    result = audit(args.root)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
