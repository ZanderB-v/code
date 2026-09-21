#!/usr/bin/env python3
"""Freeze the V2 data and evaluation protocol.

This freeze intentionally covers data and evaluation only:

* target-domain train/dev/test membership, labels, source splits, and image hashes
* synthetic_formal_v2 S10/S25/S50 manifests, labels, generation evidence, and image hashes
* character dictionary, Uyghur U2-bidi conversion, and metric implementation

It intentionally does not freeze model architecture, training hyperparameters,
fixed resize, MSR, or any other input preprocessing choice.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL_ID = "multilingual_meme_ocr_data_eval_protocol_v2"
LANGUAGES = ("zh", "ug", "kk")
TARGET_EXPECTED_COUNTS = {
    "train": {"zh": 23954, "ug": 21899, "kk": 21001},
    "dev": {"zh": 346, "ug": 295, "kk": 310},
    "test": {"zh": 1142, "ug": 931, "kk": 993},
}
SYNTHETIC_SUBSET_COUNTS = {
    "s10": {"zh": 10000, "ug": 10000, "kk": 10000},
    "s25": {"zh": 25000, "ug": 25000, "kk": 25000},
    "s50": {"zh": 50000, "ug": 50000, "kk": 50000},
}
EXPECTED_DICTIONARY_SIZE = 4891
EXPECTED_SYNTHETIC_DATASET = "synthetic_formal_v2"
EXPECTED_STYLE_PROFILE = "formal_diverse"
EXPECTED_DIFFICULTY_PROFILE = "formal_diverse"
EXPECTED_TEXT_PLAN_SEED = 20260728
EXPECTED_MAX_TEXT_OCCURRENCES = 2

# Ә ә Ғ ғ Қ қ Ң ң Ө ө Ұ ұ Ү ү Һ һ І і
KK_SPECIFIC = {
    chr(cp)
    for cp in (
        0x04D8,
        0x04D9,
        0x0492,
        0x0493,
        0x049A,
        0x049B,
        0x04A2,
        0x04A3,
        0x04E8,
        0x04E9,
        0x04B0,
        0x04B1,
        0x04AE,
        0x04AF,
        0x04BA,
        0x04BB,
        0x0406,
        0x0456,
    )
}


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
    parser.add_argument("--mode", choices=("freeze", "verify"), default="freeze")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--skip-image-hashes",
        action="store_true",
        help="For quick dry-runs only. Do not use for the official freeze.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def near_text_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", normalize_text(text)).casefold()
    return "".join(
        char
        for char in text
        if not unicodedata.category(char).startswith(("P", "Z"))
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_hash_rows(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


VOLATILE_MANIFEST_KEYS = {
    "build_summary",
    "created_utc",
    "implementation",
    "manifest",
    "membership_manifest",
    "metadata",
    "mode",
    "output_dir",
    "outputs",
    "path",
    "root",
    "subset_manifest",
    "summary",
    "verification_fingerprint_sha256",
}


def stable_manifest_view(value: Any) -> Any:
    """Remove paths and timestamps while preserving every frozen hash/value."""

    if isinstance(value, dict):
        return {
            key: stable_manifest_view(item)
            for key, item in value.items()
            if key not in VOLATILE_MANIFEST_KEYS
        }
    if isinstance(value, list):
        return [stable_manifest_view(item) for item in value]
    return value


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json(stable_manifest_view(manifest)).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def load_python_bidi():
    try:
        from bidi.algorithm import get_display  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Protocol V2 requires python-bidi. Install it in the active env with: "
            "python -m pip install python-bidi"
        ) from exc
    try:
        version = importlib.metadata.version("python-bidi")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return get_display, version


def load_metric_module(path: Path):
    spec = importlib.util.spec_from_file_location("protocol_v2_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import metric module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_dictionary(path: Path) -> tuple[list[str], set[str]]:
    chars = [
        line.rstrip("\r\n")
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.rstrip("\r\n")
    ]
    if len(chars) != len(set(chars)):
        duplicates = [ch for ch, count in Counter(chars).items() if count > 1]
        raise ValueError(f"Duplicate dictionary entries: {duplicates[:20]}")
    return chars, set(chars)


def hash_job(item: tuple[str, Path]) -> dict[str, Any]:
    image, path = item
    if not path.is_file():
        return {
            "image": image,
            "exists": False,
            "size": None,
            "sha256": None,
        }
    return {
        "image": image,
        "exists": True,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def file_hashes(root: Path, relative_paths: list[str]) -> dict[str, Any]:
    out = {}
    for rel in relative_paths:
        path = root / rel
        out[rel] = {
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    return out


def scan_target(
    root: Path,
    output_dir: Path,
    dictionary: set[str],
    workers: int,
    skip_image_hashes: bool,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    target_dir = root / "01_data_preparation" / "real_line_dataset_eval_reviewed"
    metadata_path = target_dir / "metadata.jsonl"
    if not metadata_path.is_file():
        return {}, [f"Missing target metadata: {metadata_path}"]

    rows = read_jsonl(metadata_path)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    ids: set[str] = set()
    source_splits: dict[str, set[str]] = defaultdict(set)
    unknown_chars: Counter[str] = Counter()
    membership_rows: list[dict[str, Any]] = []
    hash_label_rows: list[dict[str, Any]] = []
    image_jobs: list[tuple[str, Path]] = []

    for idx, row in enumerate(rows, 1):
        row_id = str(row.get("id") or row.get("candidate_id") or "")
        split = str(row.get("split") or "")
        language = str(row.get("language") or "")
        source_id = str(row.get("source_id") or "")
        image = str(row.get("image") or "").replace("\\", "/")
        logical = normalize_text(str(row.get("logical_text") or row.get("text") or ""))
        review_status = str(row.get("review_status") or "")

        if split not in TARGET_EXPECTED_COUNTS:
            errors.append(f"Invalid target split at row {idx}: {split!r}")
        if language not in LANGUAGES:
            errors.append(f"Invalid target language at row {idx}: {language!r}")
        if not row_id or row_id in ids:
            errors.append(f"Missing or duplicate target id at row {idx}: {row_id!r}")
        ids.add(row_id)
        if review_status != "pass":
            errors.append(f"Target row is not pass: {row_id}={review_status!r}")
        if not source_id:
            errors.append(f"Missing source_id for target row: {row_id}")
        if not image or not logical:
            errors.append(f"Missing image or label for target row: {row_id}")

        counts[split][language] += 1
        if source_id:
            source_splits[source_id].add(split)
        for char in logical:
            if char != " " and char not in dictionary:
                unknown_chars[char] += 1

        membership_rows.append(
            {
                "id": row_id,
                "split": split,
                "language": language,
                "source_id": source_id,
                "image": image,
                "logical_text": logical,
            }
        )
        hash_label_rows.append(membership_rows[-1])
        image_jobs.append((image, target_dir / image))

    counts_dict = {
        split: {language: counts[split][language] for language in LANGUAGES}
        for split in TARGET_EXPECTED_COUNTS
    }
    if counts_dict != TARGET_EXPECTED_COUNTS:
        errors.append(
            f"Target counts differ: expected={TARGET_EXPECTED_COUNTS}, actual={counts_dict}"
        )
    leaked_sources = {
        source_id: sorted(splits)
        for source_id, splits in source_splits.items()
        if len(splits) > 1
    }
    if leaked_sources:
        examples = dict(list(leaked_sources.items())[:20])
        errors.append(f"Target source_id split leakage: {examples}")
    if unknown_chars:
        errors.append(f"Target unknown dictionary characters: {dict(unknown_chars)}")

    membership_rows.sort(
        key=lambda row: (row["split"], row["language"], row["source_id"], row["id"])
    )
    membership_path = output_dir / "target_membership_manifest.jsonl"
    write_jsonl(membership_path, membership_rows)

    image_hash_path = output_dir / "target_image_hashes.jsonl"
    image_hash_summary: dict[str, Any]
    if skip_image_hashes:
        image_hash_summary = {"skipped": True}
    else:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            image_hash_rows = list(executor.map(hash_job, image_jobs))
        missing = [row for row in image_hash_rows if not row["exists"]]
        hashes = Counter(row["sha256"] for row in image_hash_rows if row["sha256"])
        duplicate_hashes = {
            digest: count for digest, count in hashes.items() if count > 1
        }
        if missing:
            errors.append(f"Missing target images: {len(missing)}")
        if duplicate_hashes:
            labels_by_hash: dict[str, set[str]] = defaultdict(set)
            splits_by_hash: dict[str, set[str]] = defaultdict(set)
            for member, hash_row in zip(hash_label_rows, image_hash_rows):
                digest = hash_row.get("sha256")
                if digest:
                    labels_by_hash[digest].add(member["logical_text"])
                    splits_by_hash[digest].add(member["split"])
            conflicts = {
                digest: sorted(values)
                for digest, values in labels_by_hash.items()
                if len(values) > 1
            }
            if conflicts:
                examples = dict(list(conflicts.items())[:20])
                errors.append(f"Target duplicate image hash label conflicts: {examples}")
            cross_split_duplicates = {
                digest: sorted(values)
                for digest, values in splits_by_hash.items()
                if len(values) > 1
            }
            if cross_split_duplicates:
                examples = dict(list(cross_split_duplicates.items())[:20])
                errors.append(
                    "Target identical image bytes cross splits: "
                    f"{examples}"
                )
        else:
            cross_split_duplicates = {}
        write_jsonl(image_hash_path, image_hash_rows)
        image_hash_summary = {
            "skipped": False,
            "rows": len(image_hash_rows),
            "missing": len(missing),
            "unique_hashes": len(hashes),
            "duplicate_hash_groups": len(duplicate_hashes),
            "cross_split_duplicate_hash_groups": len(
                cross_split_duplicates
            ),
            "manifest": str(image_hash_path),
            "manifest_sha256": sha256_file(image_hash_path),
        }

    return {
        "root": str(target_dir),
        "metadata": str(metadata_path),
        "rows": len(rows),
        "counts": counts_dict,
        "membership_manifest": str(membership_path),
        "membership_manifest_sha256": sha256_file(membership_path),
        "metadata_sha256": sha256_file(metadata_path),
        "source_ids": len(source_splits),
        "source_split_leaks": 0,
        "unknown_characters": 0,
        "image_hashes": image_hash_summary,
        "data_characterization": {
            "zh": "natural Chinese meme lines",
            "ug": "in-domain re-rendered Uyghur meme lines",
            "kk": "in-domain re-rendered Kazakh meme lines",
        },
    }, errors


def scan_synthetic(
    root: Path,
    output_dir: Path,
    dictionary: set[str],
    workers: int,
    skip_image_hashes: bool,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    formal_dir = root / "03_synthetic_generation" / "synthetic_formal_v2"
    subsets_dir = formal_dir / "subsets"
    build_summary_path = subsets_dir / "s50_build_summary.json"
    if not build_summary_path.is_file():
        return {}, [f"Missing S50 build summary: {build_summary_path}"]

    build_summary = read_json(build_summary_path)
    if build_summary.get("dataset") != EXPECTED_SYNTHETIC_DATASET:
        errors.append(f"Unexpected synthetic dataset: {build_summary.get('dataset')}")
    if build_summary.get("total_counts") != SYNTHETIC_SUBSET_COUNTS["s50"]:
        errors.append(f"Unexpected S50 total counts: {build_summary.get('total_counts')}")
    if build_summary.get("max_text_occurrence_per_language") != {
        language: EXPECTED_MAX_TEXT_OCCURRENCES for language in LANGUAGES
    }:
        errors.append(
            "Unexpected max text occurrence per language: "
            f"{build_summary.get('max_text_occurrence_per_language')}"
        )
    if build_summary.get("global_duplicate_image_hashes") != 0:
        errors.append(
            f"Synthetic duplicate image hashes: {build_summary.get('global_duplicate_image_hashes')}"
        )
    if build_summary.get("dictionary_unknown_characters") != 0:
        errors.append(
            f"Synthetic unknown dictionary characters: {build_summary.get('dictionary_unknown_characters')}"
        )

    subset_summaries = {}
    label_hashes = {}
    subset_manifest_rows: list[dict[str, Any]] = []
    image_jobs_by_subset: dict[str, list[tuple[str, Path]]] = {}

    for subset_name, expected_counts in SYNTHETIC_SUBSET_COUNTS.items():
        subset_dir = subsets_dir / subset_name
        metadata_path = subset_dir / "metadata.jsonl"
        summary_path = subset_dir / "summary.json"
        labels_dir = subset_dir / "labels"
        if not metadata_path.is_file():
            errors.append(f"Missing synthetic metadata: {metadata_path}")
            continue
        rows = read_jsonl(metadata_path)
        counts = {language: 0 for language in LANGUAGES}
        ids = set()
        image_paths = set()
        logical_texts: dict[str, set[str]] = {language: set() for language in LANGUAGES}
        unknown_chars: Counter[str] = Counter()
        ug_u2_ok = 0
        kk_specific = 0
        style_profiles = Counter()
        difficulty_profiles = Counter()
        text_plan_seeds = Counter()
        image_jobs: list[tuple[str, Path]] = []

        for row in rows:
            language = str(row.get("language") or "")
            row_id = str(row.get("id") or "")
            image = str(row.get("image") or "").replace("\\", "/")
            logical = normalize_text(str(row.get("logical_text") or ""))
            ctc = normalize_text(str(row.get("ctc_text") or ""))
            if language not in LANGUAGES:
                errors.append(f"Invalid synthetic language in {subset_name}: {language!r}")
                continue
            counts[language] += 1
            if not row_id or row_id in ids:
                errors.append(f"Missing or duplicate synthetic id in {subset_name}: {row_id!r}")
            ids.add(row_id)
            if not image or image in image_paths:
                errors.append(f"Missing or duplicate synthetic image in {subset_name}: {image!r}")
            image_paths.add(image)
            if not logical or not ctc:
                errors.append(f"Missing synthetic text in {subset_name}: {row_id}")
            logical_texts[language].add(logical)
            for char in ctc:
                if char != " " and char not in dictionary:
                    unknown_chars[char] += 1
            if language == "ug" and row.get("ctc_text_u2_status") == "python_bidi" and ctc == normalize_text(str(row.get("ctc_text_u2") or "")):
                ug_u2_ok += 1
            if language == "kk" and any(char in KK_SPECIFIC for char in logical):
                kk_specific += 1
            style_profiles[str(row.get("style_profile") or "")] += 1
            difficulty_profiles[str(row.get("difficulty_profile") or "")] += 1
            text_plan_seeds[str(row.get("text_plan_seed") or "")] += 1
            image_jobs.append((image, formal_dir / image))

        if counts != expected_counts:
            errors.append(
                f"{subset_name} counts differ: expected={expected_counts}, actual={counts}"
            )
        if unknown_chars:
            errors.append(
                f"{subset_name} unknown dictionary characters: {dict(unknown_chars)}"
            )
        if style_profiles != Counter({EXPECTED_STYLE_PROFILE: len(rows)}):
            errors.append(f"{subset_name} style profile mismatch: {dict(style_profiles)}")
        if difficulty_profiles != Counter({EXPECTED_DIFFICULTY_PROFILE: len(rows)}):
            errors.append(
                f"{subset_name} difficulty profile mismatch: {dict(difficulty_profiles)}"
            )
        if text_plan_seeds != Counter({str(EXPECTED_TEXT_PLAN_SEED): len(rows)}):
            errors.append(f"{subset_name} text plan seed mismatch: {dict(text_plan_seeds)}")
        if ug_u2_ok != expected_counts["ug"]:
            errors.append(
                f"{subset_name} Uyghur U2-bidi mismatch: {ug_u2_ok}/{expected_counts['ug']}"
            )
        kk_ratio = kk_specific / expected_counts["kk"] if expected_counts["kk"] else 0.0
        if kk_ratio < 0.30:
            errors.append(f"{subset_name} Kazakh specific-character ratio too low: {kk_ratio}")

        label_files = {}
        for language in (*LANGUAGES, "all"):
            name = "train_all.txt" if language == "all" else f"train_{language}.txt"
            label_path = labels_dir / name
            if not label_path.is_file():
                errors.append(f"Missing synthetic label file: {label_path}")
                continue
            line_count = sum(
                1
                for line in label_path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            )
            expected_lines = sum(expected_counts.values()) if language == "all" else expected_counts[language]
            if line_count != expected_lines:
                errors.append(
                    f"{subset_name}/{name} line count differs: expected={expected_lines}, actual={line_count}"
                )
            label_files[name] = {
                "path": str(label_path),
                "lines": line_count,
                "sha256": sha256_file(label_path),
            }
        label_hashes[subset_name] = label_files

        subset_manifest_rows.append(
            {
                "subset": subset_name,
                "metadata": str(metadata_path),
                "summary": str(summary_path),
                "counts": counts,
                "rows": len(rows),
                "metadata_sha256": sha256_file(metadata_path),
                "summary_sha256": sha256_file(summary_path) if summary_path.is_file() else None,
                "unique_texts": {
                    language: len(logical_texts[language]) for language in LANGUAGES
                },
                "ug_u2_python_bidi_ratio": ug_u2_ok / expected_counts["ug"],
                "kk_specific_ratio": kk_ratio,
            }
        )
        image_jobs_by_subset[subset_name] = image_jobs
        subset_summaries[subset_name] = subset_manifest_rows[-1]

    subset_manifest_path = output_dir / "synthetic_subset_manifest.jsonl"
    write_jsonl(subset_manifest_path, subset_manifest_rows)

    synthetic_hash_summary = {}
    if skip_image_hashes:
        synthetic_hash_summary["skipped"] = True
    else:
        # S50 contains all images in S10/S25, so hash S50 once and record subset
        # membership through manifests instead of duplicating hashes three times.
        s50_jobs = image_jobs_by_subset.get("s50", [])
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            image_hash_rows = list(executor.map(hash_job, s50_jobs))
        missing = [row for row in image_hash_rows if not row["exists"]]
        hashes = Counter(row["sha256"] for row in image_hash_rows if row["sha256"])
        duplicate_hashes = {
            digest: count for digest, count in hashes.items() if count > 1
        }
        if missing:
            errors.append(f"Missing S50 synthetic images: {len(missing)}")
        if duplicate_hashes:
            examples = dict(list(duplicate_hashes.items())[:20])
            errors.append(f"Duplicate S50 synthetic image bytes: {examples}")
        synthetic_image_hash_path = output_dir / "synthetic_s50_image_hashes.jsonl"
        write_jsonl(synthetic_image_hash_path, image_hash_rows)
        synthetic_hash_summary = {
            "skipped": False,
            "scope": "s50",
            "rows": len(image_hash_rows),
            "missing": len(missing),
            "unique_hashes": len(hashes),
            "duplicate_hash_groups": len(duplicate_hashes),
            "manifest": str(synthetic_image_hash_path),
            "manifest_sha256": sha256_file(synthetic_image_hash_path),
        }

    return {
        "root": str(formal_dir),
        "build_summary": str(build_summary_path),
        "build_summary_sha256": sha256_file(build_summary_path),
        "dataset": build_summary.get("dataset"),
        "shards": build_summary.get("shards"),
        "total_counts": build_summary.get("total_counts"),
        "max_text_occurrence_per_language": build_summary.get(
            "max_text_occurrence_per_language"
        ),
        "global_duplicate_image_hashes": build_summary.get("global_duplicate_image_hashes"),
        "dictionary_unknown_characters": build_summary.get("dictionary_unknown_characters"),
        "subsets": subset_summaries,
        "subset_manifest": str(subset_manifest_path),
        "subset_manifest_sha256": sha256_file(subset_manifest_path),
        "label_files": label_hashes,
        "image_hashes": synthetic_hash_summary,
        "generation_frozen_fields": {
            "style_profile": EXPECTED_STYLE_PROFILE,
            "difficulty_profile": EXPECTED_DIFFICULTY_PROFILE,
            "text_plan_seed": EXPECTED_TEXT_PLAN_SEED,
            "max_text_occurrences": EXPECTED_MAX_TEXT_OCCURRENCES,
            "formal_shard_count": 5,
            "text_shard_count": 2,
            "per_part": 5000,
        },
    }, errors


def protocol_file_hashes(root: Path) -> dict[str, Any]:
    files = [
        "scripts/protocol/freeze_protocol_v2.py",
        "scripts/protocol/freeze_target_benchmark_v2.py",
        "scripts/svtrv2/metrics_v1.py",
        "scripts/svtrv2/audit_formal_research_protocol.py",
        "scripts/svtrv2/prepare_ug_direction_labels.py",
        "scripts/svtrv2/u2_visual_predictions_to_logical_metrics.py",
        "scripts/synth/06_generate_samples.py",
        "scripts/synth/07_postprocess_images.py",
        "scripts/synth/08_validate_dataset.py",
        "scripts/synth/10_dataset_statistics.py",
        "scripts/synth/run_formal_parallel_shards_tmux.sh",
        "scripts/synth/build_s50_nested_subsets.py",
        "scripts/synth/prepare_s50_leakage_repair.py",
        "scripts/synth/finalize_s50_repair_metadata.py",
        "scripts/synth/run_s50_leakage_repair_tmux.sh",
        "03_synthetic_generation/font_library/final_check/font_manifest_final.json",
        "03_synthetic_generation/font_library/audit/font_coverage_report.csv",
        "03_synthetic_generation/background_library_v1/manifest.jsonl",
        "02_corpus_preparation/mixed_text_pool_v2/zh_mixed_text_pool_v2.jsonl",
        "02_corpus_preparation/mixed_text_pool_v2/ug_mixed_text_pool_v2.jsonl",
        "02_corpus_preparation/mixed_text_pool_v2/kk_mixed_text_pool_v2.jsonl",
        "02_corpus_preparation/mixed_text_pool_v2/zh_mixed_text_pool_v2_summary.json",
        "02_corpus_preparation/mixed_text_pool_v2/ug_mixed_text_pool_v2_summary.json",
        "02_corpus_preparation/mixed_text_pool_v2/kk_mixed_text_pool_v2_summary.json",
        "02_corpus_preparation/mixed_text_pool_v2/mixed_text_pool_v2_summary_all.json",
        "04_model_training/character_dict_hz_ug_kk_v1/character_dict.txt",
        "04_model_training/character_dict_hz_ug_kk_v1/character_frequency.csv",
        "00_docs/TARGET_BENCHMARK_PROTOCOL_V2.md",
        "00_docs/PREPROCESS_AND_S50_V2.md",
    ]
    return file_hashes(root, files)


def validate_u2_and_metrics(root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    get_display, bidi_version = load_python_bidi()
    u2_vectors = [
        "بۈگۈن OCR سىنىقى",
        "2026-يىلى SVTRv2 بىلەن 123 قۇر",
        "A12 نۇسخىسى ياخشىمۇ",
    ]
    u2_results = [
        {
            "logical": text,
            "u2_visual": get_display(normalize_text(text), base_dir="R"),
        }
        for text in u2_vectors
    ]

    metric_path = root / "scripts" / "svtrv2" / "metrics_v1.py"
    metric_module = load_metric_module(metric_path)
    bucket = defaultdict(float)
    metric_module.update_metric_bucket(bucket, "abc", "adc")
    metric_vector = metric_module.finalize_metric_bucket(bucket)
    expected_cer = 1 / 3
    if abs(metric_vector["cer"] - expected_cer) > 1e-12:
        errors.append(f"Metric CER test vector changed: {metric_vector}")

    return {
        "ug_u2_bidi": {
            "method": "python-bidi get_display(text, base_dir='R')",
            "python_bidi_version": bidi_version,
            "vectors": u2_results,
            "forbidden": "text[::-1]",
        },
        "metrics": {
            "implementation": str(metric_path),
            "implementation_sha256": sha256_file(metric_path),
            "protocol_version": getattr(
                metric_module, "METRIC_PROTOCOL_VERSION", "unknown"
            ),
            "test_vector": {
                "gt": "abc",
                "pred": "adc",
                "result": metric_vector,
            },
            "primary_selection_metric": "macro dev CER over zh/ug/kk",
            "reported_metrics": ["CER", "WER", "1-NED macro", "Line Accuracy"],
        },
    }, errors


def validate_synthetic_eval_text_isolation(
    root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Ensure no actual synthetic training label copies target dev/test text."""

    errors: list[str] = []
    target_path = (
        root
        / "01_data_preparation"
        / "real_line_dataset_eval_reviewed"
        / "metadata.jsonl"
    )
    synthetic_path = (
        root
        / "03_synthetic_generation"
        / "synthetic_formal_v2"
        / "subsets"
        / "s50"
        / "metadata.jsonl"
    )
    target_eval: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    target_eval_near: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in read_jsonl(target_path):
        split = str(row.get("split") or "")
        language = str(row.get("language") or "")
        if split not in ("dev", "test") or language not in LANGUAGES:
            continue
        logical = normalize_text(
            str(row.get("logical_text") or row.get("text") or "")
        )
        target_eval[language][split].add(logical)
        target_eval_near[language][split].add(near_text_key(logical))

    synthetic: dict[str, set[str]] = defaultdict(set)
    synthetic_near: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(synthetic_path):
        language = str(row.get("language") or "")
        logical = normalize_text(str(row.get("logical_text") or ""))
        synthetic[language].add(logical)
        synthetic_near[language].add(near_text_key(logical))

    summary: dict[str, dict[str, Any]] = {}
    for language in LANGUAGES:
        summary[language] = {}
        for split in ("dev", "test"):
            overlap = sorted(
                synthetic[language] & target_eval[language][split]
            )
            near_overlap = (
                synthetic_near[language]
                & target_eval_near[language][split]
            )
            near_only_count = max(0, len(near_overlap) - len(overlap))
            summary[language][split] = {
                "unique_overlap_count": len(overlap),
                "normalized_near_only_count": near_only_count,
                "examples": overlap[:20],
            }
            if overlap:
                errors.append(
                    f"S50 {language} text overlaps target {split}: "
                    f"{len(overlap)} unique strings"
                )
            if near_only_count:
                errors.append(
                    f"S50 {language} normalized-near text overlaps target "
                    f"{split}: {near_only_count} additional strings"
                )
    return summary, errors


def build_markdown(manifest: dict[str, Any], report: dict[str, Any]) -> str:
    target_counts = manifest["target_domain"]["counts"]
    synthetic = manifest["synthetic_formal_v2"]
    lines = [
        "# Frozen Protocol V2",
        "",
        f"Status: {report['status']}",
        "",
        "Protocol V2 freezes data and evaluation only. It does not freeze model architecture, training hyperparameters, fixed resize, dynamic resize, or MSR.",
        "",
        "## Frozen Target Domain",
        "",
        "| Split | zh | ug | kk | total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "dev", "test"):
        counts = target_counts[split]
        lines.append(
            f"| {split} | {counts['zh']} | {counts['ug']} | {counts['kk']} | {sum(counts.values())} |"
        )
    lines += [
        "",
        "- Chinese target samples are natural Chinese meme lines.",
        "- Uyghur and Kazakh target samples are in-domain re-rendered meme lines, not naturally collected Uyghur/Kazakh memes.",
        "- Split leakage is checked by `source_id`.",
        "",
        "## Frozen Synthetic Data",
        "",
        "| Subset | zh | ug | kk | total | unique zh | unique ug | unique kk | KK specific ratio | UG U2 ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for subset_name in ("s10", "s25", "s50"):
        subset = synthetic["subsets"][subset_name]
        counts = subset["counts"]
        unique = subset["unique_texts"]
        lines.append(
            f"| {subset_name.upper()} | {counts['zh']} | {counts['ug']} | {counts['kk']} | {subset['rows']} | "
            f"{unique['zh']} | {unique['ug']} | {unique['kk']} | "
            f"{subset['kk_specific_ratio']:.4f} | {subset['ug_u2_python_bidi_ratio']:.4f} |"
        )
    lines += [
        "",
        "S10 is nested in S25, and S25 is nested in S50.",
        "Synthetic text occurrence is frozen at at most two occurrences per language.",
        "",
        "## Frozen Labels, Dictionary, And Metrics",
        "",
        f"- Character dictionary size: {manifest['dictionary']['size']}",
        f"- Character dictionary SHA-256: `{manifest['dictionary']['sha256']}`",
        "- Uyghur U2 uses `python-bidi` `get_display(text, base_dir='R')`.",
        "- Metrics are CER, whitespace-token WER, macro 1-NED, and exact line accuracy.",
        "- Primary selection metric is macro dev CER over zh/ug/kk.",
        "",
        "## Frozen Evidence Files",
        "",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
        f"- Verification report: `{manifest['outputs']['verification_report']}`",
        f"- Target image hashes: `{manifest['target_domain']['image_hashes'].get('manifest', 'skipped')}`",
        f"- Synthetic S50 image hashes: `{manifest['synthetic_formal_v2']['image_hashes'].get('manifest', 'skipped')}`",
        "",
        "## Explicitly Not Frozen",
        "",
        "- fixed 48x640 input preprocessing;",
        "- MSR / RatioDataSetTVResize;",
        "- model architecture;",
        "- training schedule, optimizer, batch size, or data mixing ratio.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    frozen_output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "00_docs" / "frozen_protocol_v2"
    )

    if args.mode == "freeze":
        if frozen_output_dir.exists() and args.replace:
            shutil.rmtree(frozen_output_dir)
        frozen_output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = frozen_output_dir
        temporary_output = None
    elif not frozen_output_dir.is_dir():
        raise SystemExit(
            f"Missing frozen output dir for verify mode: {frozen_output_dir}"
        )
    else:
        temporary_output = tempfile.TemporaryDirectory(
            prefix="protocol_v2_verify_"
        )
        output_dir = Path(temporary_output.name)

    errors: list[str] = []
    dictionary_path = (
        root
        / "04_model_training"
        / "character_dict_hz_ug_kk_v1"
        / "character_dict.txt"
    )
    dictionary, dictionary_set = read_dictionary(dictionary_path)
    if len(dictionary) != EXPECTED_DICTIONARY_SIZE:
        errors.append(
            f"Dictionary size differs: expected={EXPECTED_DICTIONARY_SIZE}, actual={len(dictionary)}"
        )

    target_summary, target_errors = scan_target(
        root,
        output_dir,
        dictionary_set,
        args.workers,
        args.skip_image_hashes,
    )
    errors.extend(target_errors)
    synthetic_summary, synthetic_errors = scan_synthetic(
        root,
        output_dir,
        dictionary_set,
        args.workers,
        args.skip_image_hashes,
    )
    errors.extend(synthetic_errors)
    eval_summary, eval_errors = validate_u2_and_metrics(root)
    errors.extend(eval_errors)
    text_isolation, text_isolation_errors = (
        validate_synthetic_eval_text_isolation(root)
    )
    errors.extend(text_isolation_errors)

    file_hash_summary = protocol_file_hashes(root)
    missing_frozen_files = [
        path for path, info in file_hash_summary.items() if not info["exists"]
    ]
    if missing_frozen_files:
        errors.append(f"Missing frozen evidence files: {missing_frozen_files}")

    now = datetime.now(timezone.utc).isoformat()
    current_manifest = {
        "protocol_id": PROTOCOL_ID,
        "created_utc": now,
        "mode": args.mode,
        "root": str(root),
        "scope": {
            "frozen": [
                "target_domain_data",
                "synthetic_formal_v2_s10_s25_s50",
                "character_dictionary",
                "uyghur_u2_bidi_conversion",
                "metrics_v1",
                "data_hashes",
                "generator_scripts_and_font_background_manifests",
            ],
            "not_frozen": [
                "fixed_48x640_input_preprocessing",
                "msr_ratio_dataset_resize",
                "model_architecture",
                "training_hyperparameters",
                "optimizer_or_scheduler",
            ],
        },
        "target_domain": target_summary,
        "synthetic_formal_v2": synthetic_summary,
        "dictionary": {
            "path": str(dictionary_path),
            "size": len(dictionary),
            "sha256": sha256_file(dictionary_path),
            "required_checks": {
                "ug_core_present": True,
                "kk_specific_present": all(ch in dictionary_set for ch in KK_SPECIFIC),
            },
        },
        "evaluation": eval_summary,
        "synthetic_target_text_isolation": text_isolation,
        "frozen_file_hashes": file_hash_summary,
        "outputs": {
            "output_dir": str(output_dir),
            "manifest": str(output_dir / "protocol_v2_manifest.json"),
            "verification_report": str(output_dir / "verification_report.json"),
            "markdown": str(output_dir / "PROTOCOL_V2.md"),
        },
    }
    current_manifest["verification_fingerprint_sha256"] = (
        manifest_fingerprint(current_manifest)
    )

    if args.mode == "freeze":
        manifest_path = frozen_output_dir / "protocol_v2_manifest.json"
        report_path = frozen_output_dir / "verification_report.json"
        markdown_path = frozen_output_dir / "PROTOCOL_V2.md"
        file_hashes_path = frozen_output_dir / "frozen_file_hashes.json"
        write_json(manifest_path, current_manifest)
        write_json(file_hashes_path, file_hash_summary)
        report = {
            "protocol_id": PROTOCOL_ID,
            "checked_utc": now,
            "status": "passed" if not errors else "failed",
            "errors": errors,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "verification_fingerprint_sha256": current_manifest[
                "verification_fingerprint_sha256"
            ],
            "skip_image_hashes": args.skip_image_hashes,
        }
        write_json(report_path, report)
        markdown_path.write_text(
            build_markdown(current_manifest, report),
            encoding="utf-8",
        )
    else:
        manifest_path = frozen_output_dir / "protocol_v2_manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"Missing frozen manifest: {manifest_path}")
        frozen_manifest = read_json(manifest_path)
        frozen_fingerprint = manifest_fingerprint(frozen_manifest)
        current_fingerprint = manifest_fingerprint(current_manifest)
        if frozen_fingerprint != current_fingerprint:
            errors.append(
                "Frozen Protocol V2 fingerprint mismatch: "
                f"expected={frozen_fingerprint}, current={current_fingerprint}"
            )
        changed_sections = [
            section
            for section in (
                "scope",
                "target_domain",
                "synthetic_formal_v2",
                "dictionary",
                "evaluation",
                "synthetic_target_text_isolation",
                "frozen_file_hashes",
            )
            if stable_manifest_view(frozen_manifest.get(section))
            != stable_manifest_view(current_manifest.get(section))
        ]
        report = {
            "protocol_id": PROTOCOL_ID,
            "checked_utc": now,
            "status": "passed" if not errors else "failed",
            "errors": errors,
            "frozen_manifest": str(manifest_path),
            "frozen_manifest_sha256": sha256_file(manifest_path),
            "frozen_fingerprint_sha256": frozen_fingerprint,
            "current_fingerprint_sha256": current_fingerprint,
            "changed_sections": changed_sections,
            "skip_image_hashes": args.skip_image_hashes,
            "note": "Verify mode never rewrites the frozen manifest.",
        }
        write_json(
            frozen_output_dir / "verification_latest.json",
            report,
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if temporary_output is not None:
        temporary_output.cleanup()
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
