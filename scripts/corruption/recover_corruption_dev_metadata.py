#!/usr/bin/env python3
"""Recover deleted Dev corruption metadata from materialized images."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from build_corruption_protocol import (
    DEFAULT_ROOT,
    build_review,
    image_statistics,
    planned_variants,
    read_metadata,
    summarize,
    write_condition_files,
    write_jsonl,
)
from straug12_deterministic import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--review-samples-per-language", type=int, default=2)
    parser.add_argument("--replace-metadata", action="store_true")
    return parser.parse_args()


def recover_record(
    record: dict[str, Any],
    protocol_root: Path,
) -> dict[str, Any]:
    output_path = protocol_root / record["output_image"]
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    with Image.open(output_path) as opened:
        opened.load()
        width, height = opened.size
        stats = image_statistics(opened)
    if stats["max"] - stats["min"] < 2.0 or stats["std"] < 0.25:
        raise ValueError(f"Degenerate recovered image: {output_path} {stats}")
    updated = dict(record)
    updated["output_sha256"] = file_sha256(output_path)
    updated["output_width"] = width
    updated["output_height"] = height
    updated["materialized"] = True
    updated["image_statistics"] = stats
    return updated


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_root = (
        args.protocol_root.resolve()
        if args.protocol_root
        else root / "05_evaluation" / "corruption_protocol_v1"
    )
    draft_path = protocol_root / "corruption_protocol_v1_draft.json"
    if not draft_path.is_file():
        raise FileNotFoundError(draft_path)
    with draft_path.open("r", encoding="utf-8-sig") as handle:
        draft = json.load(handle)
    if draft.get("status") != "draft_pending_human_calibration":
        raise ValueError(f"Unexpected draft status: {draft.get('status')}")

    target_root = (
        root
        / "01_data_preparation"
        / "real_line_dataset_eval_reviewed"
    )
    metadata_path = target_root / "metadata.csv"
    split_root = protocol_root / "dev"
    manifest_path = protocol_root / "dev_manifest.jsonl"
    summary_path = protocol_root / "dev_build_summary.json"
    review_path = protocol_root / "dev_calibration_review.html"
    outputs = (manifest_path, summary_path, review_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.replace_metadata:
        raise FileExistsError(
            "Metadata outputs already exist; use --replace-metadata:\n"
            + "\n".join(existing)
        )
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)
    if file_sha256(metadata_path) != draft["target_metadata_sha256"]:
        raise ValueError("Target metadata hash differs from the recovered draft")

    source_rows = read_metadata(metadata_path, "dev")
    tasks = [
        planned_variants(
            row,
            target_root,
            split_root,
            protocol_root,
            int(draft["global_seed"]),
        )
        for row in source_rows
    ]
    planned = [record for task in tasks for record in task["variants"]]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for index, record in enumerate(
            executor.map(
                lambda item: recover_record(item, protocol_root),
                planned,
            ),
            1,
        ):
            records.append(record)
            if index % 1000 == 0 or index == len(planned):
                print(
                    json.dumps(
                        {"recovered": index, "expected": len(planned)}
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
    temporary_manifest = protocol_root / "dev_manifest.recovered.tmp.jsonl"
    write_jsonl(temporary_manifest, records)
    actual_manifest_hash = file_sha256(temporary_manifest)
    expected_manifest_hash = draft["dev_manifest_sha256"]
    if actual_manifest_hash != expected_manifest_hash:
        temporary_manifest.unlink(missing_ok=True)
        raise ValueError(
            "Recovered Dev manifest hash mismatch: "
            f"expected={expected_manifest_hash}, actual={actual_manifest_hash}"
        )
    os.replace(temporary_manifest, manifest_path)

    summary = summarize("dev", source_rows, records, manifest_only=False)
    summary.update(
        {
            "global_seed": int(draft["global_seed"]),
            "target_root": str(target_root),
            "target_metadata": str(metadata_path),
            "target_metadata_sha256": file_sha256(metadata_path),
            "manifest": str(manifest_path),
            "manifest_sha256": actual_manifest_hash,
            "recovery": {
                "method": "existing_materialized_images",
                "images_regenerated": 0,
            },
        }
    )
    if summary["status"] != "passed":
        raise RuntimeError(json.dumps(summary, ensure_ascii=False, indent=2))
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_condition_files(protocol_root, split_root, records)
    build_review(
        protocol_root,
        target_root,
        records,
        args.review_samples_per_language,
    )
    print(
        json.dumps(
            {
                "status": "CORRUPTION_DEV_METADATA_RECOVERY_OK",
                "source_samples": len(source_rows),
                "variant_rows": len(records),
                "manifest_sha256": actual_manifest_hash,
                "images_regenerated": 0,
                "outputs": [str(path) for path in outputs],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
