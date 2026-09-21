#!/usr/bin/env python3
"""Authorize the frozen corruption data for the final P1/MSR V3 evaluator.

This revision never regenerates corruptions, changes severity parameters, or
touches the test images. It archives the old evaluation freeze and updates only
code provenance after verifying every immutable data artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/data_home/wudayu/experiments/multilingual_meme_ocr/"
    "svtrv2_line_recognition"
)
AUTHORIZED_CODE_KEYS = {
    "evaluate_corruption_dev.py",
    "p1_msr_protocol.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reason", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def artifact_paths(root: Path, protocol_root: Path) -> dict[str, Path]:
    return {
        "dev_manifest.jsonl": protocol_root / "dev_manifest.jsonl",
        "test_manifest.jsonl": protocol_root / "test_manifest.jsonl",
        "dev_build_summary.json": protocol_root / "dev_build_summary.json",
        "test_build_summary.json": protocol_root / "test_build_summary.json",
        "dev_calibration_review.html": (
            protocol_root / "dev_calibration_review.html"
        ),
        "pilot_review_csv": (
            root
            / "05_evaluation/corruption_protocol_v1_pilot100_v2"
            / "human_review/corruption_pilot100_v2_review.csv"
        ),
        "straug12_deterministic.py": (
            root / "scripts/corruption/straug12_deterministic.py"
        ),
        "build_corruption_protocol.py": (
            root / "scripts/corruption/build_corruption_protocol.py"
        ),
        "freeze_corruption_protocol.py": (
            root / "scripts/corruption/freeze_corruption_protocol.py"
        ),
        "evaluate_corruption_dev.py": (
            root / "scripts/corruption/evaluate_corruption_dev.py"
        ),
        "summarize_corruption_dev.py": (
            root / "scripts/corruption/summarize_corruption_dev.py"
        ),
        "metrics_v1.py": root / "scripts/svtrv2/metrics_v1.py",
        "p1_msr_protocol.py": root / "scripts/svtrv2/p1_msr_protocol.py",
        "protocol_v2_manifest.json": (
            root / "00_docs/frozen_protocol_v2/protocol_v2_manifest.json"
        ),
        "protocol_v2_verification_latest.json": (
            root / "00_docs/frozen_protocol_v2/verification_latest.json"
        ),
        "target_metadata.csv": (
            root
            / "01_data_preparation/real_line_dataset_eval_reviewed"
            / "metadata.csv"
        ),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_root = root / "05_evaluation/corruption_protocol_v1"
    protocol_path = protocol_root / "corruption_protocol_v1.json"
    hashes_path = protocol_root / "frozen_hashes.json"
    report_path = protocol_root / "freeze_report.json"
    for path in (protocol_path, hashes_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    protocol = read_json(protocol_path)
    frozen_hashes = read_json(hashes_path)
    freeze_report = read_json(report_path)
    if protocol.get("status") != "frozen":
        raise ValueError("Corruption protocol is not frozen")
    if freeze_report.get("status") != "passed" or freeze_report.get("errors"):
        raise ValueError("Existing corruption freeze is not valid")
    if freeze_report.get("test_inference_run") is not False:
        raise ValueError("Test inference policy has already been violated")

    eval_root = protocol_root / "eval_reports"
    result_files = list(eval_root.glob("*/model_summary.json")) if eval_root.exists() else []
    if result_files:
        raise RuntimeError(
            "Remove invalid old-model corruption eval reports before the "
            "pre-evaluation revision:\n" + "\n".join(map(str, result_files))
        )

    expected = frozen_hashes.get("files", {})
    paths = artifact_paths(root, protocol_root)
    errors = []
    for name, path in paths.items():
        if not path.is_file():
            errors.append(f"missing artifact: {name}: {path}")
            continue
        if name in AUTHORIZED_CODE_KEYS or name == "protocol_v2_verification_latest.json":
            continue
        wanted = expected.get(name)
        actual = sha256(path)
        if wanted != actual:
            errors.append(
                f"immutable hash mismatch {name}: expected={wanted}, actual={actual}"
            )
    if errors:
        raise ValueError("Cannot revise corruption evaluator:\n" + "\n".join(errors))

    now = datetime.now(timezone.utc)
    revision_id = "p1_msr_v3_evaluator_" + now.strftime("%Y%m%d_%H%M%S")
    archive = protocol_root / "revisions" / revision_id
    archive.mkdir(parents=True, exist_ok=False)
    for path in (protocol_path, hashes_path, report_path):
        shutil.copy2(path, archive / path.name)

    old_protocol_sha = sha256(protocol_path)
    old_evaluator_sha = expected.get("evaluate_corruption_dev.py")
    old_p1_sha = expected.get("p1_msr_protocol.py")
    protocol["evaluation_revision"] = {
        "revision_id": revision_id,
        "revised_at_utc": now.isoformat(),
        "reason": args.reason,
        "scope": "evaluation_code_and_P1_preprocessing_provenance_only",
        "preprocess_protocol": "P1_MSR_V3",
        "corruption_parameters_changed": False,
        "dev_images_or_labels_changed": False,
        "dev_manifest_changed": False,
        "test_manifest_changed": False,
        "test_images_materialized": False,
        "valid_results_produced_under_previous_revision": False,
        "previous_protocol_sha256": old_protocol_sha,
        "previous_evaluator_sha256": old_evaluator_sha,
        "previous_p1_msr_sha256": old_p1_sha,
    }
    write_json_atomic(protocol_path, protocol)

    new_files = dict(expected)
    new_files["corruption_protocol_v1.json"] = sha256(protocol_path)
    for name, path in paths.items():
        new_files[name] = sha256(path)
    frozen_hashes["files"] = new_files
    frozen_hashes["evaluation_revision"] = protocol["evaluation_revision"]
    write_json_atomic(hashes_path, frozen_hashes)

    freeze_report["evaluation_revision"] = protocol["evaluation_revision"]
    freeze_report["protocol"] = str(protocol_path)
    freeze_report["hashes"] = str(hashes_path)
    freeze_report["status"] = "passed"
    freeze_report["errors"] = []
    freeze_report["test_inference_run"] = False
    write_json_atomic(report_path, freeze_report)

    record = {
        "status": "CORRUPTION_EVALUATOR_P1_V3_REVISION_OK",
        **protocol["evaluation_revision"],
        "archive": str(archive),
        "current_protocol_sha256": sha256(protocol_path),
        "current_evaluator_sha256": sha256(paths["evaluate_corruption_dev.py"]),
        "current_p1_msr_sha256": sha256(paths["p1_msr_protocol.py"]),
        "dev_manifest_sha256": sha256(paths["dev_manifest.jsonl"]),
        "test_manifest_sha256": sha256(paths["test_manifest.jsonl"]),
    }
    write_json_atomic(archive / "revision_record.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
