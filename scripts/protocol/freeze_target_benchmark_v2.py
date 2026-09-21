#!/usr/bin/env python3
"""Freeze or verify the target-domain benchmark independently of model input.

This protocol intentionally excludes synthetic data, resize settings, and MSR
configuration. Those choices remain experimental until the preprocessing
ablation and S50 validation are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL_ID = "multilingual_meme_target_benchmark_v2"
LANGUAGES = ("zh", "ug", "kk")
SPLITS = ("train", "dev", "test")
EXPECTED_COUNTS = {
    "train": {"zh": 23954, "ug": 21899, "kk": 21001},
    "dev": {"zh": 346, "ug": 295, "kk": 310},
    "test": {"zh": 1142, "ug": 931, "kk": 993},
}
EXPECTED_DICTIONARY_SIZE = 4891
U2_TEST_STRINGS = (
    "مەن سىزنى ياخشى كۆرىمەن",
    "2026-يىلى OCR سىنىقى",
    "SVTRv2 بىلەن 123 قۇر",
    "A12 نۇسخىسى ياخشىمۇ؟",
)


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
    return parser.parse_args()


def normalize_spaces(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_hash(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def load_python_bidi():
    try:
        from bidi.algorithm import get_display  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Target V2 requires python-bidi. Install it with: "
            "python -m pip install python-bidi"
        ) from exc
    try:
        version = importlib.metadata.version("python-bidi")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return get_display, version


def load_metric_module(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_metrics_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import metric module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_dictionary(path: Path) -> tuple[list[str], set[str]]:
    values = [
        line.rstrip("\r\n")
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.rstrip("\r\n")
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate entries in character dictionary: {path}")
    return values, set(values)


def hash_one_image(item: tuple[dict, Path]) -> dict:
    record, image_path = item
    if not image_path.is_file():
        return {**record, "exists": False, "size": None, "sha256": None}
    return {
        **record,
        "exists": True,
        "size": image_path.stat().st_size,
        "sha256": sha256_file(image_path),
    }


def scan_target(root: Path, workers: int) -> tuple[dict, list[str]]:
    errors: list[str] = []
    target_root = (
        root / "01_data_preparation" / "real_line_dataset_eval_reviewed"
    )
    metadata_path = target_root / "metadata.jsonl"
    dictionary_path = (
        root
        / "04_model_training"
        / "character_dict_hz_ug_kk_v1"
        / "character_dict.txt"
    )
    metric_path = root / "scripts" / "svtrv2" / "metrics_v1.py"
    u2_prepare_path = (
        root / "scripts" / "svtrv2" / "prepare_ug_direction_labels.py"
    )
    u2_restore_path = (
        root
        / "scripts"
        / "svtrv2"
        / "u2_visual_predictions_to_logical_metrics.py"
    )
    freeze_script_path = (
        root / "scripts" / "protocol" / "freeze_target_benchmark_v2.py"
    )
    protocol_doc_path = root / "00_docs" / "TARGET_BENCHMARK_PROTOCOL_V2.md"

    required_files = (
        metadata_path,
        dictionary_path,
        metric_path,
        u2_prepare_path,
        u2_restore_path,
        freeze_script_path,
        protocol_doc_path,
    )
    for path in required_files:
        if not path.is_file():
            errors.append(f"Missing required protocol file: {path}")
    if errors:
        return {}, errors

    get_display, bidi_version = load_python_bidi()
    dictionary, dictionary_set = read_dictionary(dictionary_path)
    if len(dictionary) != EXPECTED_DICTIONARY_SIZE:
        errors.append(
            "Character dictionary size differs: "
            f"expected={EXPECTED_DICTIONARY_SIZE}, actual={len(dictionary)}"
        )

    metric_module = load_metric_module(metric_path)
    metric_test_bucket = defaultdict(float)
    metric_module.update_metric_bucket(metric_test_bucket, "abc", "adc")
    metric_test_vector = metric_module.finalize_metric_bucket(metric_test_bucket)

    counts = {split: Counter() for split in SPLITS}
    source_splits: dict[str, set[str]] = defaultdict(set)
    source_rows: dict[str, Counter] = defaultdict(Counter)
    ids: set[str] = set()
    unknown_chars: Counter[str] = Counter()
    membership_rows: list[dict] = []
    label_rows: list[dict] = []
    image_jobs: list[tuple[dict, Path]] = []

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
            logical = normalize_spaces(
                row.get("logical_text") or row.get("text") or ""
            )

            if split not in SPLITS:
                errors.append(f"Invalid split at line {line_no}: {split!r}")
                continue
            if language not in LANGUAGES:
                errors.append(
                    f"Invalid language at line {line_no}: {language!r}"
                )
                continue
            if not row_id or row_id in ids:
                errors.append(f"Missing or duplicate id at line {line_no}: {row_id}")
            ids.add(row_id)
            if not source_id:
                errors.append(f"Missing source_id for {row_id}")
            if row.get("review_status") != "pass":
                errors.append(
                    f"Non-pass row in target benchmark: "
                    f"{row_id}={row.get('review_status')!r}"
                )
            if not image or not logical:
                errors.append(f"Missing image or label for {row_id}")
                continue

            counts[split][language] += 1
            source_splits[source_id].add(split)
            source_rows[source_id][language] += 1
            for char in logical:
                if char != " " and char not in dictionary_set:
                    unknown_chars[char] += 1

            ctc_text = (
                get_display(logical, base_dir="R")
                if language == "ug"
                else logical
            )
            membership_rows.append(
                {
                    "id": row_id,
                    "split": split,
                    "language": language,
                    "source_id": source_id,
                    "image": image,
                }
            )
            label_rows.append(
                {
                    "id": row_id,
                    "split": split,
                    "language": language,
                    "image": image,
                    "logical_text": logical,
                    "ctc_text_u2": ctc_text,
                }
            )
            image_jobs.append(
                (
                    {
                        "id": row_id,
                        "split": split,
                        "language": language,
                        "image": image,
                    },
                    target_root / image,
                )
            )

    normalized_counts = {
        split: {language: counts[split][language] for language in LANGUAGES}
        for split in SPLITS
    }
    if normalized_counts != EXPECTED_COUNTS:
        errors.append(
            f"Target counts differ: expected={EXPECTED_COUNTS}, "
            f"actual={normalized_counts}"
        )

    leakage = {
        source_id: sorted(values)
        for source_id, values in source_splits.items()
        if len(values) > 1
    }
    if leakage:
        errors.append(
            f"source_id leakage across splits: {len(leakage)} groups"
        )
    if unknown_chars:
        errors.append(
            f"Target labels have {len(unknown_chars)} unknown characters"
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        image_rows = list(executor.map(hash_one_image, image_jobs))
    missing_images = [row for row in image_rows if not row["exists"]]
    if missing_images:
        errors.append(f"Missing target images: {len(missing_images)}")

    membership_rows.sort(key=lambda row: row["id"])
    label_rows.sort(key=lambda row: row["id"])
    image_rows.sort(key=lambda row: row["id"])
    source_manifest = [
        {
            "source_id": source_id,
            "split": next(iter(split_values)),
            "language_line_counts": dict(sorted(source_rows[source_id].items())),
        }
        for source_id, split_values in sorted(source_splits.items())
        if len(split_values) == 1
    ]

    artifacts = {}
    for name, path in {
        "metadata_jsonl": metadata_path,
        "character_dict": dictionary_path,
        "metrics_v1": metric_path,
        "u2_label_builder": u2_prepare_path,
        "u2_prediction_restorer": u2_restore_path,
        "freeze_script": freeze_script_path,
        "protocol_document": protocol_doc_path,
    }.items():
        artifacts[name] = {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    u2_test_vectors = [
        {
            "logical_text": text,
            "visual_text": get_display(normalize_spaces(text), base_dir="R"),
        }
        for text in U2_TEST_STRINGS
    ]
    result = {
        "protocol_id": PROTOCOL_ID,
        "target_root": str(target_root.relative_to(root)).replace("\\", "/"),
        "rows": len(label_rows),
        "counts": normalized_counts,
        "unique_ids": len(ids),
        "source_ids": len(source_splits),
        "source_id_overlap_count": len(leakage),
        "source_id_overlap_examples": dict(list(leakage.items())[:20]),
        "dictionary_size": len(dictionary),
        "unknown_character_count": sum(unknown_chars.values()),
        "unknown_characters": dict(unknown_chars.most_common(100)),
        "python_bidi_version": bidi_version,
        "u2_rule": (
            "bidi.algorithm.get_display(normalize_spaces(text), base_dir='R')"
        ),
        "u2_test_vectors": u2_test_vectors,
        "metric_protocol": getattr(
            metric_module, "METRIC_PROTOCOL_VERSION", "unknown"
        ),
        "metric_test_vector": metric_test_vector,
        "membership_sha256": canonical_hash(membership_rows),
        "labels_sha256": canonical_hash(label_rows),
        "source_split_sha256": canonical_hash(source_manifest),
        "image_manifest_sha256": canonical_hash(image_rows),
        "image_count": len(image_rows),
        "missing_image_count": len(missing_images),
        "artifacts": artifacts,
        "not_frozen": [
            "synthetic_data",
            "model_input_size",
            "resize_or_msr_configuration",
            "model_architecture",
            "training_hyperparameters",
        ],
        "_rows": {
            "labels": label_rows,
            "sources": source_manifest,
            "images": image_rows,
        },
    }
    return result, errors


def write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = ["\t".join(header)]
    lines.extend("\t".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def manifest_core(scan: dict) -> dict:
    return {key: value for key, value in scan.items() if key != "_rows"}


def freeze(args: argparse.Namespace, output_dir: Path) -> int:
    if output_dir.exists() and any(output_dir.iterdir()) and not args.replace:
        raise SystemExit(
            f"Freeze directory already exists: {output_dir}. "
            "Use --replace only when intentionally creating a new freeze."
        )
    if output_dir.exists() and args.replace:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scan, errors = scan_target(args.root.resolve(), args.workers)
    if errors:
        report = {"status": "failed", "errors": errors}
        (output_dir / "verification_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    rows = scan.pop("_rows")
    write_tsv(
        output_dir / "target_labels.tsv",
        ("id split language image logical_text_json ctc_text_u2_json".split()),
        [
            [
                row["id"],
                row["split"],
                row["language"],
                row["image"],
                json.dumps(row["logical_text"], ensure_ascii=False),
                json.dumps(row["ctc_text_u2"], ensure_ascii=False),
            ]
            for row in rows["labels"]
        ],
    )
    write_tsv(
        output_dir / "source_image_splits.tsv",
        ["source_id", "split", "language_line_counts_json"],
        [
            [
                row["source_id"],
                row["split"],
                json.dumps(
                    row["language_line_counts"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ]
            for row in rows["sources"]
        ],
    )
    write_tsv(
        output_dir / "target_images.sha256.tsv",
        ["sha256", "size", "split", "language", "id", "image"],
        [
            [
                row["sha256"] or "",
                str(row["size"] or ""),
                row["split"],
                row["language"],
                row["id"],
                row["image"],
            ]
            for row in rows["images"]
        ],
    )

    manifest = {
        **scan,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", ""),
        "python": sys.version,
        "freeze_artifacts": {
            name: {
                "sha256": sha256_file(output_dir / name),
                "size": (output_dir / name).stat().st_size,
            }
            for name in (
                "target_labels.tsv",
                "source_image_splits.tsv",
                "target_images.sha256.tsv",
            )
        },
    }
    manifest_path = output_dir / "frozen_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "status": "passed",
        "mode": "freeze",
        "protocol_id": PROTOCOL_ID,
        "manifest": str(manifest_path),
        "rows": manifest["rows"],
        "counts": manifest["counts"],
        "source_id_overlap_count": 0,
        "unknown_character_count": 0,
        "missing_image_count": 0,
    }
    (output_dir / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def verify(args: argparse.Namespace, output_dir: Path) -> int:
    manifest_path = output_dir / "frozen_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing frozen manifest: {manifest_path}")
    frozen = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    scan, errors = scan_target(args.root.resolve(), args.workers)
    current = manifest_core(scan)

    compare_keys = (
        "protocol_id",
        "rows",
        "counts",
        "source_ids",
        "source_id_overlap_count",
        "dictionary_size",
        "unknown_character_count",
        "python_bidi_version",
        "u2_rule",
        "u2_test_vectors",
        "metric_protocol",
        "metric_test_vector",
        "membership_sha256",
        "labels_sha256",
        "source_split_sha256",
        "image_manifest_sha256",
        "image_count",
        "missing_image_count",
        "artifacts",
        "not_frozen",
    )
    mismatches = {}
    for key in compare_keys:
        if frozen.get(key) != current.get(key):
            mismatches[key] = {
                "frozen": frozen.get(key),
                "current": current.get(key),
            }

    for name, info in frozen.get("freeze_artifacts", {}).items():
        path = output_dir / name
        if not path.is_file():
            mismatches[f"freeze_artifact:{name}"] = {
                "frozen": info,
                "current": "missing",
            }
            continue
        actual = {"sha256": sha256_file(path), "size": path.stat().st_size}
        if actual != info:
            mismatches[f"freeze_artifact:{name}"] = {
                "frozen": info,
                "current": actual,
            }

    status = "passed" if not errors and not mismatches else "failed"
    report = {
        "status": status,
        "mode": "verify",
        "protocol_id": PROTOCOL_ID,
        "errors": errors,
        "mismatches": mismatches,
        "rows": current.get("rows"),
        "counts": current.get("counts"),
    }
    (output_dir / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "passed" else 1


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "00_docs" / "frozen_target_benchmark_v2"
    )
    if args.mode == "freeze":
        raise SystemExit(freeze(args, output_dir))
    raise SystemExit(verify(args, output_dir))


if __name__ == "__main__":
    main()
