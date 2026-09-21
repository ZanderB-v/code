#!/usr/bin/env python3
"""Freeze an approved dev corruption calibration and create a test manifest."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from build_corruption_protocol import DEFAULT_ROOT, PROTOCOL_ID
from straug12_deterministic import (
    CORRUPTIONS,
    IMPLEMENTATION_ID,
    SEVERITIES,
    file_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=None)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--pilot-review-csv", type=Path, required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def run_test_manifest_builder(
    root: Path,
    protocol_root: Path,
    seed: int,
    workers: int,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("build_corruption_protocol.py")),
        "--root",
        str(root),
        "--protocol-root",
        str(protocol_root),
        "--split",
        "test",
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--manifest-only",
        "--replace",
    ]
    subprocess.run(command, cwd=str(root), check=True)


def validate_dev(
    protocol_root: Path,
    records: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    errors = []
    counts = Counter()
    source_sizes: dict[str, tuple[int, int]] = {}
    for index, row in enumerate(records, 1):
        counts[(row["corruption"], row["severity"], row["language"])] += 1
        if row.get("protocol_id") != PROTOCOL_ID:
            errors.append(f"row {index}: wrong protocol_id")
        if row.get("implementation_id") != IMPLEMENTATION_ID:
            errors.append(f"row {index}: wrong implementation_id")
        if not row.get("materialized"):
            errors.append(f"row {index}: dev image is not materialized")
        output_path = protocol_root / row["output_image"]
        if not output_path.is_file():
            errors.append(f"row {index}: missing {output_path}")
            continue
        if file_sha256(output_path) != row.get("output_sha256"):
            errors.append(f"row {index}: output hash mismatch")
        with Image.open(output_path) as image:
            image.verify()
        with Image.open(output_path) as image:
            size = image.size
        expected_size = (row["source_width"], row["source_height"])
        corruption = row["corruption"]
        if corruption == "rotate":
            canvas_valid = (
                size[0] >= expected_size[0]
                and size[1] >= expected_size[1]
                and size != expected_size
            )
        elif corruption == "translate_x":
            canvas_valid = (
                size[0] > expected_size[0]
                and size[1] == expected_size[1]
            )
        elif corruption == "translate_y":
            canvas_valid = (
                size[0] == expected_size[0]
                and size[1] > expected_size[1]
            )
        else:
            canvas_valid = size == expected_size
        if not canvas_valid:
            errors.append(
                f"row {index}: invalid canvas for {corruption}: "
                f"size={size} source={expected_size}"
            )
        if (row.get("output_width"), row.get("output_height")) != size:
            errors.append(f"row {index}: recorded output size mismatch")
        source_sizes.setdefault(row["sample_id"], expected_size)
    return errors, {
        "rows": len(records),
        "source_samples": len(source_sizes),
        "condition_language_counts": {
            f"{corruption}/level_{severity}/{language}": count
            for (corruption, severity, language), count in sorted(counts.items())
        },
    }


def validate_test_manifest(
    records: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    errors = []
    counts = Counter()
    source_ids = set()
    for index, row in enumerate(records, 1):
        counts[(row["corruption"], row["severity"], row["language"])] += 1
        source_ids.add(row["sample_id"])
        if row.get("materialized"):
            errors.append(f"row {index}: test variant was materialized")
        if row.get("output_sha256") is not None:
            errors.append(f"row {index}: test output hash must be null")
        if row.get("split") != "test":
            errors.append(f"row {index}: wrong split")
    return errors, {
        "rows": len(records),
        "source_samples": len(source_ids),
        "condition_language_counts": {
            f"{corruption}/level_{severity}/{language}": count
            for (corruption, severity, language), count in sorted(counts.items())
        },
    }


def main() -> None:
    args = parse_args()
    if not args.approve:
        raise ValueError(
            "Freezing requires --approve after the human calibration review."
        )
    root = args.root.resolve()
    pilot_review_csv = args.pilot_review_csv.resolve()
    if not pilot_review_csv.is_file():
        raise FileNotFoundError(pilot_review_csv)
    with pilot_review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        pilot_review_rows = list(csv.DictReader(handle))
    if len(pilot_review_rows) != 100:
        raise ValueError(
            f"Expected 100 pilot review rows, got {len(pilot_review_rows)}"
        )
    pilot_decisions = Counter(
        (row.get("decision") or "").strip().lower()
        for row in pilot_review_rows
    )
    non_keep_decisions = {
        decision: count
        for decision, count in pilot_decisions.items()
        if decision != "keep"
    }
    if non_keep_decisions:
        raise ValueError(
            "Pilot review is not fully approved. Every one of the 100 rows "
            "must have decision=keep before freezing; found "
            f"{non_keep_decisions}."
        )
    protocol_root = (
        args.protocol_root.resolve()
        if args.protocol_root
        else root / "05_evaluation" / "corruption_protocol_v1"
    )
    frozen_path = protocol_root / "corruption_protocol_v1.json"
    if frozen_path.exists():
        raise FileExistsError(
            f"Already frozen: {frozen_path}. A frozen protocol is immutable."
        )
    data_protocol_path = (
        root
        / "00_docs"
        / "frozen_protocol_v2"
        / "protocol_v2_manifest.json"
    )
    data_verification_path = (
        root
        / "00_docs"
        / "frozen_protocol_v2"
        / "verification_latest.json"
    )
    for path in (data_protocol_path, data_verification_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    data_verification = read_json(data_verification_path)
    if (
        data_verification.get("status") != "passed"
        or data_verification.get("errors")
    ):
        raise ValueError(
            "Frozen Protocol V2 verification must be passed with errors=[]"
        )

    draft_path = protocol_root / "corruption_protocol_v1_draft.json"
    dev_summary_path = protocol_root / "dev_build_summary.json"
    dev_manifest_path = protocol_root / "dev_manifest.jsonl"
    review_path = protocol_root / "dev_calibration_review.html"
    for path in (draft_path, dev_summary_path, dev_manifest_path, review_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    draft = read_json(draft_path)
    dev_summary = read_json(dev_summary_path)
    if draft.get("status") != "draft_pending_human_calibration":
        raise ValueError(f"Unexpected draft status: {draft.get('status')}")
    if dev_summary.get("status") != "passed":
        raise ValueError("Dev build summary did not pass")
    if draft.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Draft protocol ID mismatch")
    if draft.get("implementation_id") != IMPLEMENTATION_ID:
        raise ValueError("Draft implementation ID mismatch")
    target_metadata_path = Path(draft["target_metadata"])
    if not target_metadata_path.is_file():
        raise FileNotFoundError(target_metadata_path)
    if file_sha256(target_metadata_path) != draft.get("target_metadata_sha256"):
        raise ValueError("Target metadata changed after Dev corruption generation")

    current_implementation_hash = file_sha256(
        Path(__file__).with_name("straug12_deterministic.py")
    )
    if draft.get("implementation_sha256") != current_implementation_hash:
        raise ValueError(
            "Corruption implementation changed after dev generation; rebuild dev."
        )
    current_builder_hash = file_sha256(
        Path(__file__).with_name("build_corruption_protocol.py")
    )
    if draft.get("builder_sha256") != current_builder_hash:
        raise ValueError(
            "Corruption builder changed after dev generation; rebuild dev."
        )
    if draft.get("dev_manifest_sha256") != file_sha256(dev_manifest_path):
        raise ValueError("Dev manifest changed after draft generation")

    dev_records = read_jsonl(dev_manifest_path)
    errors, dev_validation = validate_dev(protocol_root, dev_records)
    if errors:
        raise RuntimeError(
            json.dumps(
                {"status": "failed", "errors": errors[:100]},
                ensure_ascii=False,
                indent=2,
            )
        )

    run_test_manifest_builder(
        root,
        protocol_root,
        int(draft["global_seed"]),
        args.workers,
    )
    test_manifest_path = protocol_root / "test_manifest.jsonl"
    test_summary_path = protocol_root / "test_build_summary.json"
    test_records = read_jsonl(test_manifest_path)
    test_summary = read_json(test_summary_path)
    test_errors, test_validation = validate_test_manifest(test_records)
    if test_summary.get("status") != "passed":
        test_errors.extend(test_summary.get("errors") or ["test summary failed"])
    if test_errors:
        raise RuntimeError(
            json.dumps(
                {"status": "failed", "errors": test_errors[:100]},
                ensure_ascii=False,
                indent=2,
            )
        )

    frozen = {
        **draft,
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration": {
            "reviewer": args.reviewer,
            "approved": True,
            "notes": args.notes,
            "review_file": str(review_path),
            "pilot_review_csv": str(pilot_review_csv),
            "pilot_review_csv_sha256": file_sha256(pilot_review_csv),
            "pilot_review_rows": len(pilot_review_rows),
            "pilot_decision_counts": dict(sorted(pilot_decisions.items())),
            "decision": (
                "Level 1 mild; Level 2 clearly harder; Level 3 difficult but "
                "label-preserving for the reviewed samples."
            ),
        },
        "dev_validation": dev_validation,
        "test_validation": test_validation,
        "test_manifest": str(test_manifest_path),
        "test_manifest_sha256": file_sha256(test_manifest_path),
        "test_policy": (
            "manifest frozen; transformed test images and model predictions "
            "must not be generated during development"
        ),
        "data_protocol_dependency": {
            "manifest": str(data_protocol_path),
            "manifest_sha256": file_sha256(data_protocol_path),
            "verification": str(data_verification_path),
            "verification_sha256": file_sha256(data_verification_path),
        },
        "benchmark_aggregation": {
            "clean": "severity 0, reported separately",
            "corrupted": "12 operators x severities 1/2/3",
            "primary": "mean macro CER over all 36 single-factor conditions",
            "also_report": [
                "per-language CER/WER/1-NED/Line Accuracy",
                "per-operator mean",
                "per-severity mean",
                "absolute CER increase",
                "relative CER increase",
            ],
        },
    }
    frozen_path.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    hashes = {
        "protocol_id": PROTOCOL_ID,
        "files": {
            "corruption_protocol_v1.json": file_sha256(frozen_path),
            "dev_manifest.jsonl": file_sha256(dev_manifest_path),
            "test_manifest.jsonl": file_sha256(test_manifest_path),
            "dev_build_summary.json": file_sha256(dev_summary_path),
            "test_build_summary.json": file_sha256(test_summary_path),
            "dev_calibration_review.html": file_sha256(review_path),
            "pilot_review_csv": file_sha256(pilot_review_csv),
            "straug12_deterministic.py": current_implementation_hash,
            "build_corruption_protocol.py": current_builder_hash,
            "freeze_corruption_protocol.py": file_sha256(Path(__file__)),
            "evaluate_corruption_dev.py": file_sha256(
                Path(__file__).with_name("evaluate_corruption_dev.py")
            ),
            "summarize_corruption_dev.py": file_sha256(
                Path(__file__).with_name("summarize_corruption_dev.py")
            ),
            "metrics_v1.py": file_sha256(
                Path(__file__).parents[1] / "svtrv2" / "metrics_v1.py"
            ),
            "p1_msr_protocol.py": file_sha256(
                Path(__file__).parents[1] / "svtrv2" / "p1_msr_protocol.py"
            ),
            "protocol_v2_manifest.json": file_sha256(data_protocol_path),
            "protocol_v2_verification_latest.json": file_sha256(
                data_verification_path
            ),
            "target_metadata.csv": file_sha256(target_metadata_path),
        },
        "operators": list(CORRUPTIONS),
        "severities": list(SEVERITIES),
    }
    hashes_path = protocol_root / "frozen_hashes.json"
    hashes_path.write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "passed",
        "errors": [],
        "protocol": str(frozen_path),
        "hashes": str(hashes_path),
        "dev": dev_validation,
        "test_manifest_only": test_validation,
        "test_inference_run": False,
    }
    report_path = protocol_root / "freeze_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
