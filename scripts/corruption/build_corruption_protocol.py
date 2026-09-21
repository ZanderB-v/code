#!/usr/bin/env python3
"""Build deterministic dev corruptions or a manifest-only test plan."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from straug12_deterministic import (
    CORRUPTIONS,
    IMPLEMENTATION_ID,
    SEVERITIES,
    apply_corruption,
    deterministic_seed,
    file_sha256,
    plan_corruption,
)


PROTOCOL_ID = "multilingual_meme_ocr_corruption_protocol_v1"
DEFAULT_ROOT = Path(
    "/data_home/wudayu/experiments/multilingual_meme_ocr/"
    "svtrv2_line_recognition"
)
LANGUAGES = ("zh", "ug", "kk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--target-root",
        type=Path,
        default=None,
        help="Defaults to 01_data_preparation/real_line_dataset_eval_reviewed.",
    )
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=None,
        help="Defaults to 05_evaluation/corruption_protocol_v1.",
    )
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--review-samples-per-language", type=int, default=2)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def read_metadata(path: Path, split: str) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != split:
                continue
            language = row.get("language") or ""
            image = (row.get("image") or "").replace("\\", "/")
            logical_text = row.get("logical_text") or row.get("text") or ""
            sample_id = row.get("id") or row.get("candidate_id") or ""
            if language not in LANGUAGES or not image or not logical_text or not sample_id:
                raise ValueError(f"Invalid {split} metadata row: {row}")
            rows.append(
                {
                    "id": sample_id,
                    "language": language,
                    "split": split,
                    "source_id": row.get("source_id") or "",
                    "image": image,
                    "logical_text": logical_text,
                }
            )
    rows.sort(key=lambda row: (row["language"], row["id"]))
    if not rows:
        raise ValueError(f"No rows found for split={split} in {path}")
    return rows


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return normalized or "sample"


def condition_rel(corruption: str, severity: int) -> Path:
    return Path("conditions") / corruption / f"level_{severity}"


def planned_variants(
    row: dict[str, str],
    target_root: Path,
    split_root: Path,
    protocol_root: Path,
    global_seed: int,
) -> dict[str, Any]:
    source_path = target_root / row["image"]
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with Image.open(source_path) as source:
        width, height = source.size
    source_hash = file_sha256(source_path)
    output_name = safe_name(row["id"]) + ".png"
    variants = []
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            sample_seed = deterministic_seed(
                global_seed,
                row["id"],
                corruption,
                severity,
            )
            params = plan_corruption(
                corruption,
                severity,
                (width, height),
                sample_seed,
            )
            condition_root = split_root / condition_rel(corruption, severity)
            output_path = (
                condition_root
                / "images"
                / row["language"]
                / output_name
            )
            variants.append(
                {
                    "protocol_id": PROTOCOL_ID,
                    "implementation_id": IMPLEMENTATION_ID,
                    "split": row["split"],
                    "sample_id": row["id"],
                    "source_id": row["source_id"],
                    "language": row["language"],
                    "logical_text": row["logical_text"],
                    "source_image": row["image"],
                    "source_sha256": source_hash,
                    "source_width": width,
                    "source_height": height,
                    "corruption": corruption,
                    "severity": severity,
                    "global_seed": global_seed,
                    "sample_seed": sample_seed,
                    "parameters": params,
                    "output_image": output_path.relative_to(protocol_root).as_posix(),
                    "output_sha256": None,
                    "materialized": False,
                }
            )
    return {
        "source_path": str(source_path),
        "variants": variants,
        "protocol_root": str(protocol_root),
    }


def image_statistics(image: Image.Image) -> dict[str, float]:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def process_source_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    source_path = Path(task["source_path"])
    protocol_root = Path(task["protocol_root"])
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    results = []
    for record in task["variants"]:
        transformed = apply_corruption(source, record["parameters"])
        corruption = record["corruption"]
        if corruption == "rotate":
            if not (
                transformed.width >= source.width
                and transformed.height >= source.height
                and transformed.size != source.size
            ):
                raise ValueError(
                    f"Unsafe rotate canvas for {record['sample_id']}: "
                    f"source={source.size} output={transformed.size}"
                )
        elif corruption == "translate_x":
            if not (
                transformed.width > source.width
                and transformed.height == source.height
            ):
                raise ValueError(
                    f"Unsafe translate_x canvas for {record['sample_id']}: "
                    f"source={source.size} output={transformed.size}"
                )
        elif corruption == "translate_y":
            if not (
                transformed.width == source.width
                and transformed.height > source.height
            ):
                raise ValueError(
                    f"Unsafe translate_y canvas for {record['sample_id']}: "
                    f"source={source.size} output={transformed.size}"
                )
        elif transformed.size != source.size:
            raise ValueError(
                f"Unexpected canvas change for {record['sample_id']} "
                f"{corruption} level {record['severity']}"
            )
        stats = image_statistics(transformed)
        if stats["max"] - stats["min"] < 2.0 or stats["std"] < 0.25:
            raise ValueError(
                "Degenerate transformed image: "
                f"{record['sample_id']} {record['corruption']} "
                f"level {record['severity']} stats={stats}"
            )
        output_path = protocol_root / record["output_image"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".tmp.png")
        transformed.save(temporary, format="PNG", optimize=False)
        os.replace(temporary, output_path)
        updated = dict(record)
        updated["output_sha256"] = file_sha256(output_path)
        updated["output_width"] = transformed.width
        updated["output_height"] = transformed.height
        updated["materialized"] = True
        updated["image_statistics"] = stats
        results.append(updated)
    return results


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_condition_files(
    protocol_root: Path,
    split_root: Path,
    records: list[dict[str, Any]],
) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["corruption"], record["severity"])].append(record)

    for (corruption, severity), condition_rows in grouped.items():
        condition_root = split_root / condition_rel(corruption, severity)
        condition_root.mkdir(parents=True, exist_ok=True)
        metadata_path = condition_root / "metadata.csv"
        fields = (
            "id",
            "language",
            "split",
            "source_id",
            "image",
            "logical_text",
            "corruption",
            "severity",
            "sample_seed",
            "source_sha256",
            "output_sha256",
        )
        with metadata_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in sorted(
                condition_rows,
                key=lambda row: (row["language"], row["sample_id"]),
            ):
                output_abs = protocol_root / record["output_image"]
                writer.writerow(
                    {
                        "id": record["sample_id"],
                        "language": record["language"],
                        "split": record["split"],
                        "source_id": record["source_id"],
                        "image": output_abs.relative_to(condition_root).as_posix(),
                        "logical_text": record["logical_text"],
                        "corruption": corruption,
                        "severity": severity,
                        "sample_seed": record["sample_seed"],
                        "source_sha256": record["source_sha256"],
                        "output_sha256": record["output_sha256"] or "",
                    }
                )

        labels_dir = condition_root / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in condition_rows:
            by_language[record["language"]].append(record)
        for language in LANGUAGES:
            split = condition_rows[0]["split"]
            label_path = labels_dir / f"corrupt_{split}_{language}_logical.txt"
            with label_path.open("w", encoding="utf-8") as handle:
                for item in sorted(
                    by_language[language],
                    key=lambda row: row["sample_id"],
                ):
                    output_abs = protocol_root / item["output_image"]
                    image_rel = output_abs.relative_to(condition_root).as_posix()
                    handle.write(f"{image_rel}\t{item['logical_text']}\n")


def relative_href(from_path: Path, target: Path) -> str:
    return Path(os.path.relpath(target, from_path.parent)).as_posix()


def build_review(
    protocol_root: Path,
    target_root: Path,
    records: list[dict[str, Any]],
    samples_per_language: int,
) -> Path:
    review_path = protocol_root / "dev_calibration_review.html"
    asset_root = protocol_root / "calibration_review_assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    by_key = {
        (
            record["sample_id"],
            record["corruption"],
            record["severity"],
        ): record
        for record in records
    }
    source_lookup: dict[str, dict[str, Any]] = {}
    for record in records:
        source_lookup.setdefault(record["sample_id"], record)
    selected: dict[str, list[str]] = {}
    for language in LANGUAGES:
        ids = sorted(
            record["sample_id"]
            for record in source_lookup.values()
            if record["language"] == language
        )
        if not ids:
            selected[language] = []
            continue
        positions = np.linspace(
            0,
            len(ids) - 1,
            min(samples_per_language, len(ids)),
            dtype=int,
        )
        selected[language] = [ids[index] for index in positions]

    sections = []
    for corruption in CORRUPTIONS:
        rows_html = []
        for language in LANGUAGES:
            for sample_id in selected[language]:
                clean_record = source_lookup[sample_id]
                clean_path = target_root / clean_record["source_image"]
                clean_asset = (
                    asset_root
                    / "clean"
                    / language
                    / (safe_name(sample_id) + clean_path.suffix.lower())
                )
                clean_asset.parent.mkdir(parents=True, exist_ok=True)
                if not clean_asset.exists():
                    shutil.copy2(clean_path, clean_asset)
                cells = [
                    (
                        "Clean",
                        relative_href(review_path, clean_asset),
                        {"severity": 0},
                    )
                ]
                for severity in SEVERITIES:
                    record = by_key[(sample_id, corruption, severity)]
                    output_path = protocol_root / record["output_image"]
                    variant_asset = (
                        asset_root
                        / corruption
                        / f"level_{severity}"
                        / language
                        / output_path.name
                    )
                    variant_asset.parent.mkdir(parents=True, exist_ok=True)
                    if not variant_asset.exists():
                        shutil.copy2(output_path, variant_asset)
                    cells.append(
                        (
                            f"Level {severity}",
                            relative_href(review_path, variant_asset),
                            record["parameters"],
                        )
                    )
                cell_html = []
                for label, href, params in cells:
                    cell_html.append(
                        "<figure>"
                        f"<figcaption>{html.escape(label)}</figcaption>"
                        f'<img src="{html.escape(href)}" loading="lazy">'
                        f"<pre>{html.escape(json.dumps(params, ensure_ascii=False, sort_keys=True))}</pre>"
                        "</figure>"
                    )
                rows_html.append(
                    '<article class="sample">'
                    f"<h3>{html.escape(language.upper())} "
                    f"{html.escape(sample_id)}</h3>"
                    f"<p>{html.escape(clean_record['logical_text'])}</p>"
                    '<div class="variants">'
                    + "".join(cell_html)
                    + "</div></article>"
                )
        sections.append(
            f'<section id="{corruption}"><h2>{corruption}</h2>'
            + "".join(rows_html)
            + "</section>"
        )

    document = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Corruption Protocol V1 Calibration</title>
<style>
body { margin: 0; font-family: Arial, sans-serif; color: #202124; background: #f4f5f2; }
header { position: sticky; top: 0; z-index: 2; background: #fff; border-bottom: 1px solid #bbb; padding: 12px 18px; }
h1 { margin: 0 0 6px; font-size: 22px; }
nav { display: flex; flex-wrap: wrap; gap: 8px; }
nav a { color: #0b57d0; text-decoration: none; }
section { padding: 18px; border-bottom: 2px solid #aaa; }
h2 { margin-top: 0; }
.sample { background: #fff; border: 1px solid #c9cbc7; border-radius: 6px; margin: 12px 0; padding: 12px; }
.sample h3, .sample p { margin: 0 0 8px; }
.variants { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
figure { margin: 0; min-width: 0; }
figcaption { font-weight: 700; margin-bottom: 4px; }
img { display: block; width: 100%; height: 130px; object-fit: contain; background: #eee; border: 1px solid #bbb; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 10px; max-height: 95px; overflow: auto; }
@media (max-width: 900px) { .variants { grid-template-columns: 1fr 1fr; } }
</style>
</head>
<body>
<header>
<h1>Corruption Protocol V1: Human Calibration</h1>
<div>Approve only when Level 1 is mild, Level 2 is clearly harder, and Level 3 remains label-preserving.</div>
<nav>__NAV__</nav>
</header>
<main>__SECTIONS__</main>
</body>
</html>
"""
    navigation = "".join(
        f'<a href="#{name}">{html.escape(name)}</a>' for name in CORRUPTIONS
    )
    review_path.write_text(
        document.replace("__NAV__", navigation).replace(
            "__SECTIONS__",
            "".join(sections),
        ),
        encoding="utf-8",
    )
    return review_path


def summarize(
    split: str,
    source_rows: list[dict[str, str]],
    records: list[dict[str, Any]],
    manifest_only: bool,
) -> dict[str, Any]:
    source_counts = Counter(row["language"] for row in source_rows)
    condition_counts = Counter(
        (row["corruption"], row["severity"], row["language"])
        for row in records
    )
    expected_per_source = len(CORRUPTIONS) * len(SEVERITIES)
    expected_records = len(source_rows) * expected_per_source
    errors = []
    if len(records) != expected_records:
        errors.append(
            f"Expected {expected_records} variant rows, got {len(records)}"
        )
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            for language, expected in source_counts.items():
                actual = condition_counts[(corruption, severity, language)]
                if actual != expected:
                    errors.append(
                        f"{corruption}/level_{severity}/{language}: "
                        f"expected {expected}, got {actual}"
                    )
    materialized = sum(bool(row["materialized"]) for row in records)
    if manifest_only and materialized:
        errors.append("Manifest-only build unexpectedly materialized images")
    if not manifest_only and materialized != len(records):
        errors.append(
            f"Only {materialized}/{len(records)} records were materialized"
        )
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "split": split,
        "source_samples": len(source_rows),
        "source_language_counts": dict(source_counts),
        "corruptions": list(CORRUPTIONS),
        "severities": list(SEVERITIES),
        "severity_zero_policy": "clean_source_image_not_in_corrupted_average",
        "single_factor_policy": "each_variant_applies_exactly_one_corruption",
        "variant_rows": len(records),
        "expected_variant_rows": expected_records,
        "materialized_rows": materialized,
        "manifest_only": manifest_only,
        "condition_language_counts": {
            f"{corruption}/level_{severity}/{language}": count
            for (corruption, severity, language), count in sorted(
                condition_counts.items()
            )
        },
    }


def clean_split_outputs(
    protocol_root: Path,
    split: str,
    replace: bool,
) -> None:
    frozen = protocol_root / "corruption_protocol_v1.json"
    if frozen.is_file():
        raise RuntimeError(
            f"Protocol is already frozen at {frozen}; do not overwrite it."
        )
    split_root = protocol_root / split
    related = (
        protocol_root / f"{split}_manifest.jsonl",
        protocol_root / f"{split}_build_summary.json",
    )
    if split_root.exists() or any(path.exists() for path in related):
        if not replace:
            raise FileExistsError(
                f"{split} outputs already exist under {protocol_root}; use --replace"
            )
        if split_root.exists():
            shutil.rmtree(split_root)
        for path in related:
            if path.exists():
                path.unlink()
        if split == "dev":
            review = protocol_root / "dev_calibration_review.html"
            draft = protocol_root / "corruption_protocol_v1_draft.json"
            assets = protocol_root / "calibration_review_assets"
            if review.exists():
                review.unlink()
            if draft.exists():
                draft.unlink()
            if assets.exists():
                shutil.rmtree(assets)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    target_root = (
        args.target_root.resolve()
        if args.target_root
        else (
            root
            / "01_data_preparation"
            / "real_line_dataset_eval_reviewed"
        )
    )
    protocol_root = (
        args.protocol_root.resolve()
        if args.protocol_root
        else root / "05_evaluation" / "corruption_protocol_v1"
    )
    metadata_path = target_root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if args.split == "dev" and args.manifest_only:
        raise ValueError("Dev calibration must materialize images")
    if args.split == "test" and not args.manifest_only:
        raise ValueError(
            "Test images must not be materialized during protocol calibration; "
            "use --manifest-only"
        )

    protocol_root.mkdir(parents=True, exist_ok=True)
    clean_split_outputs(protocol_root, args.split, args.replace)
    split_root = protocol_root / args.split
    source_rows = read_metadata(metadata_path, args.split)
    tasks = [
        planned_variants(
            row,
            target_root,
            split_root,
            protocol_root,
            args.seed,
        )
        for row in source_rows
    ]

    if args.manifest_only:
        records = [
            record
            for task in tasks
            for record in task["variants"]
        ]
    else:
        records = []
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(process_source_task, task) for task in tasks]
            completed = 0
            for future in as_completed(futures):
                records.extend(future.result())
                completed += 1
                if completed % 25 == 0 or completed == len(tasks):
                    print(
                        json.dumps(
                            {
                                "source_progress": completed,
                                "source_samples": len(tasks),
                                "variant_rows": len(records),
                            }
                        ),
                        flush=True,
                    )

    records.sort(
        key=lambda row: (
            row["corruption"],
            row["severity"],
            row["language"],
            row["sample_id"],
        )
    )
    manifest_path = protocol_root / f"{args.split}_manifest.jsonl"
    write_jsonl(manifest_path, records)
    summary = summarize(
        args.split,
        source_rows,
        records,
        args.manifest_only,
    )
    summary.update(
        {
            "global_seed": args.seed,
            "target_root": str(target_root),
            "target_metadata": str(metadata_path),
            "target_metadata_sha256": file_sha256(metadata_path),
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
        }
    )
    summary_path = protocol_root / f"{args.split}_build_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if summary["status"] != "passed":
        raise RuntimeError(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.manifest_only:
        write_condition_files(protocol_root, split_root, records)
        review = build_review(
            protocol_root,
            target_root,
            records,
            args.review_samples_per_language,
        )
        implementation_path = Path(__file__).with_name(
            "straug12_deterministic.py"
        )
        draft = {
            "status": "draft_pending_human_calibration",
            "protocol_id": PROTOCOL_ID,
            "implementation_id": IMPLEMENTATION_ID,
            "operator_source": {
                "description": (
                    "Deterministic line-OCR evaluation adaptation of the 12 "
                    "STRAug operators used in the second augmentation round "
                    "of Xu et al."
                ),
                "straug_repository": "https://github.com/roatienza/straug",
                "straug_paper": (
                    "Atienza, Data Augmentation for Scene Text Recognition, "
                    "ICCVW 2021"
                ),
                "uyghur_paper": (
                    "Xu et al., Correlation-guided decoding strategy for "
                    "low-resource Uyghur scene text recognition, 2025"
                ),
                "compatibility_note": (
                    "This portable Pillow/NumPy implementation is not claimed "
                    "to be byte-identical to STRAug's OpenCV-TPS backend."
                ),
            },
            "global_seed": args.seed,
            "target_metadata": str(metadata_path),
            "target_metadata_sha256": summary["target_metadata_sha256"],
            "corruptions": list(CORRUPTIONS),
            "severities": list(SEVERITIES),
            "severity_zero_policy": "Clean Dev only",
            "random_magnitude_policy": (
                "forbidden_in_benchmark; each model uses the same frozen "
                "sample-level parameters"
            ),
            "composition_policy": "single corruption only; no chaining",
            "test_policy": "manifest_only_after_dev_calibration; no inference",
            "implementation_file": str(implementation_path),
            "implementation_sha256": file_sha256(implementation_path),
            "builder_sha256": file_sha256(Path(__file__)),
            "dev_manifest": str(manifest_path),
            "dev_manifest_sha256": summary["manifest_sha256"],
            "dev_review": str(review),
        }
        draft_path = protocol_root / "corruption_protocol_v1_draft.json"
        draft_path.write_text(
            json.dumps(draft, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Calibration review: {review}")
        print(
            "Do not freeze until a human confirms all three severity levels "
            "remain label-preserving."
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
