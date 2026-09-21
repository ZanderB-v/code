#!/usr/bin/env python3
"""Freeze and verify the v1 data and evaluation protocol.

The freeze records canonical dataset membership, exact training/evaluation
labels, image bytes, the character dictionary, U2-bidi conversion, input
preprocessing, and metric implementations. It never modifies source data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL_ID = "multilingual_meme_line_recognition_protocol_v1"
LANGS = ("zh", "ug", "kk")
SPLITS = ("train", "dev", "test")
DEFAULT_SHARDS = tuple(f"synthetic_shard_{index:04d}_parallel" for index in range(5))

EXPECTED_TARGET_COUNTS = {
    "train": {"zh": 23954, "ug": 21899, "kk": 21001},
    "dev": {"zh": 346, "ug": 295, "kk": 310},
    "test": {"zh": 1142, "ug": 931, "kk": 993},
}
EXPECTED_SYNTHETIC_PER_LANGUAGE = 50000
EXPECTED_SYNTHETIC_PER_SHARD_LANGUAGE = 10000
EXPECTED_DICTIONARY_ENTRIES = 4891


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <root>/00_docs/frozen_protocol_v1.",
    )
    parser.add_argument("--mode", choices=["freeze", "verify"], default="freeze")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Skip image-byte hashes in verify mode.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--synthetic-shards", nargs="+", default=list(DEFAULT_SHARDS))
    return parser.parse_args()


def normalize_spaces(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def update_canonical_hash(digest, value) -> None:
    digest.update(canonical_json_bytes(value))
    digest.update(b"\n")


def load_bidi():
    try:
        from bidi.algorithm import get_display  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Strict U2 verification requires python-bidi. Install it in the "
            "OpenOCR environment with: python -m pip install python-bidi"
        ) from exc
    try:
        version = importlib.metadata.version("python-bidi")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return get_display, version


def read_character_dictionary(path: Path) -> tuple[list[str], set[str]]:
    characters = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            value = line.rstrip("\r\n")
            if value:
                characters.append(value)
    if len(characters) != len(set(characters)):
        raise ValueError(f"Duplicate entries in character dictionary: {path}")
    return characters, set(characters)


def label_file_bytes(lines: list[str]) -> bytes:
    text = "\n".join(lines) + ("\n" if lines else "")
    return text.encode("utf-8")


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(bool(line.strip()) for line in handle)


def compare_label_file(path: Path, expected_lines: list[str], errors: list[str]) -> dict:
    if not path.exists():
        errors.append(f"Missing label file: {path}")
        return {"path": str(path), "exists": False}
    expected = label_file_bytes(expected_lines)
    actual = path.read_bytes()
    if actual != expected:
        errors.append(f"Label content differs from metadata/U2 protocol: {path}")
    return {
        "path": str(path),
        "exists": True,
        "rows": len(expected_lines),
        "sha256": sha256_bytes(actual),
        "expected_sha256": sha256_bytes(expected),
        "matches_expected": actual == expected,
    }


def scan_target(
    root: Path,
    dictionary: set[str],
    get_display,
) -> tuple[dict, list[tuple[str, Path]], list[Path], list[str], list[str]]:
    target_root = root / "01_data_preparation" / "real_line_dataset_eval_reviewed"
    metadata_path = target_root / "metadata.jsonl"
    labels_dir = root / "04_model_training" / "datasets" / "e1_target_only" / "labels"
    errors: list[str] = []
    warnings: list[str] = []
    counts = {split: Counter() for split in SPLITS}
    source_splits: dict[str, set[str]] = defaultdict(set)
    source_counts = {split: set() for split in SPLITS}
    seen_ids: set[str] = set()
    missing_chars: Counter[str] = Counter()
    membership_hash = hashlib.sha256()
    image_refs: list[tuple[str, Path]] = []
    labels: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    with metadata_path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Bad target JSON at line {line_no}: {exc}")
                continue

            row_id = row.get("id") or row.get("candidate_id") or ""
            split = row.get("split") or ""
            language = row.get("language") or ""
            source_id = row.get("source_id") or ""
            image = (row.get("image") or "").replace("\\", "/")
            logical = normalize_spaces(row.get("logical_text") or row.get("text") or "")

            if split not in SPLITS:
                errors.append(f"Invalid target split at line {line_no}: {split!r}")
                continue
            if language not in LANGS:
                errors.append(f"Invalid target language at line {line_no}: {language!r}")
                continue
            if not row_id or row_id in seen_ids:
                errors.append(f"Missing or duplicate target id at line {line_no}: {row_id!r}")
            seen_ids.add(row_id)
            if not source_id:
                errors.append(f"Missing source_id for target row {row_id}")
            if row.get("review_status") != "pass":
                errors.append(
                    f"Non-pass target row in frozen benchmark: "
                    f"{row_id}={row.get('review_status')!r}"
                )
            if not image or not logical:
                errors.append(f"Missing image/text for target row {row_id}")
                continue

            counts[split][language] += 1
            source_splits[source_id].add(split)
            source_counts[split].add(source_id)
            image_path = target_root / image
            if not image_path.is_file():
                errors.append(f"Missing target image: {image_path}")
            image_refs.append((f"target/{image}", image_path))

            for char in logical:
                if char != " " and char not in dictionary:
                    missing_chars[char] += 1

            u2 = get_display(logical, base_dir="R") if language == "ug" else logical
            labels[(split, language, "logical")].append(f"{image}\t{logical}")
            labels[(split, language, "u2")].append(f"{image}\t{u2}")
            update_canonical_hash(
                membership_hash,
                {
                    "id": row_id,
                    "split": split,
                    "language": language,
                    "source_id": source_id,
                    "image": image,
                    "logical_text": logical,
                    "ctc_text": u2,
                },
            )

    overlaps = {
        source_id: sorted(split_set)
        for source_id, split_set in source_splits.items()
        if len(split_set) > 1
    }
    if overlaps:
        errors.append(f"source_id leakage across target splits: {len(overlaps)} source ids")
    if missing_chars:
        errors.append(f"Target labels contain {len(missing_chars)} unknown characters")

    normalized_counts = {
        split: {language: counts[split][language] for language in LANGS}
        for split in SPLITS
    }
    if normalized_counts != EXPECTED_TARGET_COUNTS:
        errors.append(
            f"Target counts differ: expected={EXPECTED_TARGET_COUNTS}, "
            f"actual={normalized_counts}"
        )

    label_reports = {}
    label_paths: list[Path] = []
    for split in SPLITS:
        all_u2_lines: list[str] = []
        for language in LANGS:
            for order in ("logical", "u2"):
                path = labels_dir / f"target_{split}_{language}_{order}.txt"
                label_paths.append(path)
                label_reports[path.name] = compare_label_file(
                    path,
                    labels[(split, language, order)],
                    errors,
                )
            all_u2_lines.extend(labels[(split, language, "u2")])
        path = labels_dir / f"target_{split}_all_u2.txt"
        label_paths.append(path)
        label_reports[path.name] = compare_label_file(path, all_u2_lines, errors)

    summary = {
        "metadata": str(metadata_path),
        "rows": sum(sum(value.values()) for value in counts.values()),
        "counts": normalized_counts,
        "source_image_counts": {
            split: len(source_counts[split]) for split in SPLITS
        },
        "source_id_overlap_count": len(overlaps),
        "source_id_overlap_examples": dict(list(overlaps.items())[:20]),
        "unique_ids": len(seen_ids),
        "unknown_character_count": sum(missing_chars.values()),
        "unknown_characters": dict(missing_chars.most_common(50)),
        "membership_sha256": membership_hash.hexdigest(),
        "label_files": label_reports,
    }
    provenance_files = [
        metadata_path,
        target_root / "metadata.csv",
        target_root / "summary.json",
        *label_paths,
    ]
    return summary, image_refs, provenance_files, errors, warnings


def scan_synthetic(
    root: Path,
    shard_names: list[str],
    dictionary: set[str],
    get_display,
) -> tuple[dict, list[tuple[str, Path]], list[Path], list[str], list[str]]:
    synthetic_root = root / "03_synthetic_generation" / "synthetic_formal_v1"
    errors: list[str] = []
    warnings: list[str] = []
    total_counts = Counter()
    shard_counts: dict[str, dict[str, int]] = {}
    missing_chars: Counter[str] = Counter()
    u2_mismatches = 0
    non_ug_ctc_mismatches = 0
    membership_hash = hashlib.sha256()
    image_refs: list[tuple[str, Path]] = []
    provenance_files: list[Path] = []

    for shard_name in shard_names:
        shard_dir = synthetic_root / shard_name
        metadata_path = shard_dir / "metadata.jsonl"
        summary_path = shard_dir / "merged_summary.json"
        if not metadata_path.is_file():
            errors.append(f"Missing synthetic metadata: {metadata_path}")
            continue
        provenance_files.append(metadata_path)
        if summary_path.is_file():
            provenance_files.append(summary_path)
        else:
            warnings.append(f"Missing optional merged summary: {summary_path}")

        counts = Counter()
        seen_keys: set[str] = set()
        with metadata_path.open("r", encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"Bad synthetic JSON {metadata_path}:{line_no}: {exc}")
                    continue

                row_id = row.get("id") or ""
                language = row.get("language") or ""
                image = (row.get("image") or "").replace("\\", "/")
                logical = normalize_spaces(row.get("logical_text") or "")
                ctc_text = normalize_spaces(row.get("ctc_text") or "")
                key = f"{shard_name}:{row_id}"

                if language not in LANGS:
                    errors.append(f"Invalid language for {key}: {language!r}")
                    continue
                if not row_id or key in seen_keys:
                    errors.append(f"Missing or duplicate synthetic key: {key}")
                seen_keys.add(key)
                if not image or not logical or not ctc_text:
                    errors.append(f"Missing image/logical/ctc text for {key}")
                    continue

                counts[language] += 1
                total_counts[language] += 1
                image_path = shard_dir / image
                if not image_path.is_file():
                    errors.append(f"Missing synthetic image: {image_path}")
                image_refs.append((f"synthetic/{shard_name}/{image}", image_path))

                expected_ctc = (
                    get_display(logical, base_dir="R")
                    if language == "ug"
                    else logical
                )
                if language == "ug" and ctc_text != expected_ctc:
                    u2_mismatches += 1
                if language != "ug" and ctc_text != logical:
                    non_ug_ctc_mismatches += 1
                for char in logical:
                    if char != " " and char not in dictionary:
                        missing_chars[char] += 1

                update_canonical_hash(
                    membership_hash,
                    {
                        "shard": shard_name,
                        "id": row_id,
                        "language": language,
                        "image": image,
                        "logical_text": logical,
                        "ctc_text": ctc_text,
                    },
                )

        shard_counts[shard_name] = {
            language: counts[language] for language in LANGS
        }
        expected_shard = {
            language: EXPECTED_SYNTHETIC_PER_SHARD_LANGUAGE for language in LANGS
        }
        if shard_counts[shard_name] != expected_shard:
            errors.append(
                f"Synthetic shard counts differ for {shard_name}: "
                f"expected={expected_shard}, actual={shard_counts[shard_name]}"
            )

    normalized_total = {language: total_counts[language] for language in LANGS}
    expected_total = {
        language: EXPECTED_SYNTHETIC_PER_LANGUAGE for language in LANGS
    }
    if normalized_total != expected_total:
        errors.append(
            f"Synthetic totals differ: expected={expected_total}, "
            f"actual={normalized_total}"
        )
    if u2_mismatches:
        errors.append(f"Synthetic UG U2-bidi mismatches: {u2_mismatches}")
    if non_ug_ctc_mismatches:
        errors.append(f"Synthetic ZH/KK logical CTC mismatches: {non_ug_ctc_mismatches}")
    if missing_chars:
        errors.append(f"Synthetic labels contain {len(missing_chars)} unknown characters")

    d2_labels = root / "04_model_training" / "datasets" / "d2_synth50k" / "labels"
    d2_label_rows = {}
    if d2_labels.is_dir():
        d2_label_paths = sorted(d2_labels.glob("*.txt"))
        provenance_files.extend(d2_label_paths)
        expected_rows = {
            "train_synth50k_zh.txt": 50000,
            "train_synth50k_ug.txt": 50000,
            "train_synth50k_kk.txt": 50000,
            "train_synth50k_all.txt": 150000,
            "real_dev_zh_logical.txt": EXPECTED_TARGET_COUNTS["dev"]["zh"],
            "real_dev_ug_logical.txt": EXPECTED_TARGET_COUNTS["dev"]["ug"],
            "real_dev_kk_logical.txt": EXPECTED_TARGET_COUNTS["dev"]["kk"],
            "real_test_zh_logical.txt": EXPECTED_TARGET_COUNTS["test"]["zh"],
            "real_test_ug_logical.txt": EXPECTED_TARGET_COUNTS["test"]["ug"],
            "real_test_kk_logical.txt": EXPECTED_TARGET_COUNTS["test"]["kk"],
        }
        paths_by_name = {path.name: path for path in d2_label_paths}
        for name, expected in expected_rows.items():
            path = paths_by_name.get(name)
            if path is None:
                errors.append(f"Missing D2 label file: {d2_labels / name}")
                continue
            actual = count_nonempty_lines(path)
            d2_label_rows[name] = actual
            if actual != expected:
                errors.append(
                    f"D2 label row count differs for {name}: "
                    f"expected={expected}, actual={actual}"
                )
    else:
        errors.append(f"Missing D2 label directory: {d2_labels}")

    summary = {
        "root": str(synthetic_root),
        "shards": shard_names,
        "counts_by_shard": shard_counts,
        "total_counts": normalized_total,
        "total_rows": sum(total_counts.values()),
        "ug_u2_mismatch_count": u2_mismatches,
        "non_ug_ctc_mismatch_count": non_ug_ctc_mismatches,
        "unknown_character_count": sum(missing_chars.values()),
        "unknown_characters": dict(missing_chars.most_common(50)),
        "membership_sha256": membership_hash.hexdigest(),
        "d2_label_rows": d2_label_rows,
    }
    return summary, image_refs, provenance_files, errors, warnings


def hash_one_image(item: tuple[str, Path]) -> tuple[str, str, int, str | None]:
    relative_name, path = item
    try:
        return relative_name, sha256_file(path), path.stat().st_size, None
    except Exception as exc:
        return relative_name, "", 0, repr(exc)


def hash_images(
    refs: list[tuple[str, Path]],
    workers: int,
    output_path: Path | None,
) -> dict:
    ordered = sorted(refs, key=lambda item: item[0])
    digest = hashlib.sha256()
    duplicate_counter = Counter()
    errors = []
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for relative_name, file_hash, size, error in executor.map(hash_one_image, ordered):
            if error:
                errors.append(f"{relative_name}: {error}")
            line = f"{file_hash}\t{size}\t{relative_name}"
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
            rows.append(line)
            if file_hash:
                duplicate_counter[file_hash] += 1
    if output_path is not None:
        output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    duplicate_groups = sum(count > 1 for count in duplicate_counter.values())
    return {
        "files": len(rows),
        "aggregate_sha256": digest.hexdigest(),
        "duplicate_hash_groups": duplicate_groups,
        "hash_errors": errors,
    }


def static_protocol() -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "dataset": {
            "target_benchmark": {
                "description": {
                    "zh": "natural Chinese meme lines",
                    "ug": "target-domain re-rendered Uyghur meme lines",
                    "kk": "target-domain re-rendered Kazakh meme lines",
                },
                "split_unit": "source_id/full source image",
                "expected_counts": EXPECTED_TARGET_COUNTS,
                "mutation_policy": "frozen; do not re-render, re-split, or edit test",
            },
            "auxiliary_synthetic_training_corpus": {
                "expected_per_language": EXPECTED_SYNTHETIC_PER_LANGUAGE,
                "role": "training/pretraining only; never a primary test set",
            },
        },
        "label_order": {
            "zh": "Unicode logical LTR order",
            "kk": "Unicode logical LTR order",
            "ug": {
                "logical_storage": "standard Unicode logical order",
                "ctc_training": "U2 visual order",
                "conversion": "bidi.algorithm.get_display(normalize_spaces(text), base_dir='R')",
                "inference_reporting": "convert U2 visual prediction back to logical order before metrics",
                "simple_reverse_fallback": "forbidden",
            },
        },
        "input_preprocessing": {
            "decode": "RGB",
            "operator": "OpenOCR RecTVResize",
            "image_shape": [48, 640],
            "padding": True,
            "interpolation": "bicubic",
            "aspect_ratio": "preserved unless required width exceeds 640",
            "padding_side": "right",
            "padding_value_after_normalization": 0.0,
            "normalization": "ToTensor then Normalize(mean=0.5, std=0.5)",
            "max_text_length": 60,
        },
        "metrics": {
            "text_order": "standard logical order for all reported metrics",
            "normalization": "no case folding or punctuation removal; UG whitespace normalized during bidi conversion",
            "cer": "sum character Levenshtein distance / sum ground-truth characters",
            "wer": "sum whitespace-token Levenshtein distance / sum ground-truth tokens",
            "one_minus_ned": {
                "reported_variant": "macro",
                "formula": "mean_i(1 - ED(gt_i,pred_i)/max(len(gt_i),len(pred_i),1))",
            },
            "line_accuracy": "exact logical-string matches / samples",
            "primary_selection_metric": "arithmetic mean of per-language dev CER",
            "test_policy": "test is evaluated only after dev checkpoint selection",
        },
    }


def check_training_configs(root: Path) -> tuple[dict, list[str]]:
    errors = []
    result = {}
    config_names = (
        "svtrv2_s_d2_synth50k.yml",
        "svtrv2_s_e1_target_only.yml",
        "svtrv2_s_e5_d2_to_target.yml",
    )
    for name in config_names:
        path = root / "04_model_training" / "configs" / name
        if not path.is_file():
            errors.append(f"Missing frozen training config: {path}")
            result[name] = {"exists": False}
            continue
        text = path.read_text(encoding="utf-8-sig")
        image_shape_count = text.count("image_shape: [48, 640]")
        padding_true_count = text.count("padding: True")
        max_length_60 = "max_text_length: &max_text_length 60" in text
        valid = image_shape_count >= 2 and padding_true_count >= 2 and max_length_60
        result[name] = {
            "exists": True,
            "image_shape_48x640_occurrences": image_shape_count,
            "padding_true_occurrences": padding_true_count,
            "max_text_length_60": max_length_60,
            "matches_protocol": valid,
        }
        if not valid:
            errors.append(f"Training config does not match frozen input protocol: {path}")
    return result, errors


def collect_file_hashes(root: Path, files: list[Path]) -> dict[str, dict]:
    required_code_files = [
        root / "scripts" / "svtrv2" / "metrics_v1.py",
        root / "scripts" / "svtrv2" / "infer_label_file_metrics.py",
        root / "scripts" / "svtrv2" / "u2_visual_predictions_to_logical_metrics.py",
        root / "scripts" / "svtrv2" / "evaluate_d2_checkpoint.py",
        root / "scripts" / "svtrv2" / "prepare_e1_target_only.py",
        root / "scripts" / "svtrv2" / "prepare_d2_synth50k.py",
        root / "scripts" / "protocol" / "freeze_experiment_v1.py",
        root / "scripts" / "synth" / "06_generate_samples.py",
        root / "scripts" / "synth" / "07_postprocess_images.py",
        root / "scripts" / "synth" / "08_validate_dataset.py",
        root / "third_party" / "OpenOCR" / "openrec" / "preprocess" / "resize.py",
        root / "04_model_training" / "character_dict_hz_ug_kk_v1" / "character_dict.txt",
        root / "00_docs" / "EXPERIMENT_PROTOCOL_V1.md",
    ]
    optional_provenance = [
        root / "04_model_training" / "configs" / "svtrv2_s_d2_synth50k.yml",
        root / "04_model_training" / "configs" / "svtrv2_s_e1_target_only.yml",
        root / "04_model_training" / "configs" / "svtrv2_s_e5_d2_to_target.yml",
        root / "03_synthetic_generation" / "font_library" / "audit" / "font_manifest_final.json",
        root / "03_synthetic_generation" / "font_library" / "audit" / "font_manifest_candidates.json",
        root / "03_synthetic_generation" / "background_library_v1" / "manifest.jsonl",
        root / "03_synthetic_generation" / "background_library_v1" / "summary.json",
        root / "02_corpus_preparation" / "mixed_text_pool_v2" / "mixed_text_pool_v2_summary_all.json",
    ]
    all_paths = set(files + required_code_files)
    all_paths.update(path for path in optional_provenance if path.is_file())
    result = {}
    for path in sorted(all_paths):
        relative = path.relative_to(root).as_posix()
        if not path.is_file():
            result[relative] = {"exists": False}
            continue
        result[relative] = {
            "exists": True,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def compare_frozen(frozen: dict, current: dict, quick: bool) -> list[str]:
    errors = []
    for key in ("protocol", "target", "synthetic", "character_dictionary", "file_hashes"):
        if frozen.get(key) != current.get(key):
            errors.append(f"Frozen section changed: {key}")
    if not quick and frozen.get("image_hashes") != current.get("image_hashes"):
        errors.append("Frozen image bytes changed")
    return errors


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "00_docs" / "frozen_protocol_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "frozen_manifest.json"
    report_path = output_dir / "verification_report.json"

    if args.mode == "freeze" and manifest_path.exists() and not args.replace:
        raise SystemExit(
            f"Freeze already exists: {manifest_path}\n"
            "Use --mode verify to check it, or --replace to intentionally replace it."
        )
    if args.mode == "verify" and not manifest_path.is_file():
        raise SystemExit(f"Frozen manifest does not exist: {manifest_path}")
    if args.mode == "freeze" and args.quick:
        raise SystemExit("--quick is only valid with --mode verify")

    get_display, bidi_version = load_bidi()
    dictionary_path = (
        root
        / "04_model_training"
        / "character_dict_hz_ug_kk_v1"
        / "character_dict.txt"
    )
    dictionary_chars, dictionary_set = read_character_dictionary(dictionary_path)
    if len(dictionary_chars) != EXPECTED_DICTIONARY_ENTRIES:
        raise SystemExit(
            "Character dictionary size changed: "
            f"expected={EXPECTED_DICTIONARY_ENTRIES}, actual={len(dictionary_chars)}"
        )

    target, target_images, target_files, target_errors, target_warnings = scan_target(
        root,
        dictionary_set,
        get_display,
    )
    synthetic, synthetic_images, synthetic_files, synthetic_errors, synthetic_warnings = scan_synthetic(
        root,
        list(args.synthetic_shards),
        dictionary_set,
        get_display,
    )

    image_hashes = {"skipped": True}
    image_hash_errors: list[str] = []
    if not args.quick:
        target_hash_output = (
            output_dir / "target_images.sha256.tsv"
            if args.mode == "freeze"
            else None
        )
        synthetic_hash_output = (
            output_dir / "synthetic_images.sha256.tsv"
            if args.mode == "freeze"
            else None
        )
        target_image_hashes = hash_images(
            target_images,
            args.workers,
            target_hash_output,
        )
        synthetic_image_hashes = hash_images(
            synthetic_images,
            args.workers,
            synthetic_hash_output,
        )
        image_hash_errors.extend(target_image_hashes["hash_errors"])
        image_hash_errors.extend(synthetic_image_hashes["hash_errors"])
        image_hashes = {
            "target": target_image_hashes,
            "synthetic": synthetic_image_hashes,
        }

    protocol = static_protocol()
    protocol["label_order"]["ug"]["python_bidi_version"] = bidi_version
    config_checks, config_errors = check_training_configs(root)
    protocol["input_preprocessing"]["config_checks"] = config_checks
    file_hashes = collect_file_hashes(root, target_files + synthetic_files)
    dictionary_summary = {
        "path": str(dictionary_path),
        "entries": len(dictionary_chars),
        "sha256": sha256_file(dictionary_path),
    }
    current = {
        "protocol_id": PROTOCOL_ID,
        "protocol": protocol,
        "target": target,
        "synthetic": synthetic,
        "character_dictionary": dictionary_summary,
        "file_hashes": file_hashes,
        "image_hashes": image_hashes,
    }

    validation_errors = (
        target_errors
        + synthetic_errors
        + image_hash_errors
        + config_errors
    )
    warnings = target_warnings + synthetic_warnings
    if not args.quick:
        if image_hashes["target"]["duplicate_hash_groups"]:
            warnings.append(
                "Exact duplicate target image hashes: "
                f"{image_hashes['target']['duplicate_hash_groups']} groups"
            )
        if image_hashes["synthetic"]["duplicate_hash_groups"]:
            warnings.append(
                "Exact duplicate synthetic image hashes: "
                f"{image_hashes['synthetic']['duplicate_hash_groups']} groups"
            )
    verification = {
        "protocol_id": PROTOCOL_ID,
        "mode": args.mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "python": sys.version,
        "platform": platform.platform(),
        "status": "passed",
        "errors": [],
        "warnings": warnings,
    }

    if args.mode == "freeze":
        verification["errors"] = validation_errors
        if validation_errors:
            verification["status"] = "failed"
        else:
            manifest = dict(current)
            manifest["created_utc"] = verification["timestamp_utc"]
            manifest["root_at_freeze"] = str(root)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    else:
        frozen = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        comparison_errors = compare_frozen(frozen, current, args.quick)
        verification["errors"] = validation_errors + comparison_errors
        if verification["errors"]:
            verification["status"] = "failed"

    report_path.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    if verification["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
