#!/usr/bin/env python3
"""Validate and freeze the authorized target-train HEM V1 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROTOCOL_ID = "near_miss_hem_training_v1"
SAMPLER_PROTOCOL = "NEAR_MISS_HEM_WEIGHTED_RATIO_V1"
EXPECTED_ROWS = 66854
EXPECTED_REVIEW_ROWS = 165
EXPECTED_WEIGHTS = {"1.0": 65271, "1.5": 607, "2.0": 976}
NON_GENUINE = {
    "ambiguous_image",
    "gt_annotation_error",
    "normalization_issue",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_review(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canonicalize_manifest_order(
    train: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Bind audited weights by sample ID, then emit exact LMDB file order."""

    train_ids = [str(row.get("id") or row.get("candidate_id")) for row in train]
    manifest_ids = [str(row["sample_id"]) for row in manifest]
    if not all(train_ids) or len(set(train_ids)) != len(train_ids):
        raise ValueError("Frozen Train sample IDs are missing or duplicated")
    if not all(manifest_ids) or len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("HEM manifest sample IDs are missing or duplicated")
    if set(train_ids) != set(manifest_ids):
        raise ValueError("HEM manifest membership differs from frozen Train")
    by_id = {str(row["sample_id"]): row for row in manifest}
    return [by_id[sample_id] for sample_id in train_ids], manifest_ids == train_ids


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = args.audit_dir.resolve()
    output = args.output.resolve()
    freeze_report = output / "hem_v1_freeze.json"
    if freeze_report.is_file():
        report = json.loads(freeze_report.read_text(encoding="utf-8-sig"))
        for name, expected in report.get("frozen_file_hashes", {}).items():
            path = output / name
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Previously frozen HEM artifact changed: {path}")
        print(json.dumps({**report, "status": "HEM_V1_ALREADY_FROZEN"}, ensure_ascii=False, indent=2))
        return
    if output.exists():
        raise FileExistsError(f"Incomplete HEM freeze directory exists: {output}")

    review_path = audit / "train_hard_candidate_review.csv"
    manifest_path = audit / "formal_hem_manifest.jsonl"
    gate_path = audit / "train_review_gate_report.json"
    train_path = (
        root
        / "01_data_preparation/real_line_dataset_eval_reviewed"
        / "train_reviewed/metadata.jsonl"
    )
    for path in (review_path, manifest_path, gate_path, train_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    if gate.get("status") != "TRAIN_HEM_AUTHORIZED":
        raise ValueError("Target-train review did not authorize HEM")
    if not gate.get("formal_hem_manifest_created") or gate.get("test_evaluated"):
        raise ValueError("HEM gate provenance is invalid")
    if sha256(review_path) != gate.get("review_csv_sha256"):
        raise ValueError("Review CSV hash differs from the authorization report")
    if sha256(manifest_path) != gate.get("formal_hem_manifest_sha256"):
        raise ValueError("HEM manifest hash differs from the authorization report")

    review = read_review(review_path)
    manifest = read_jsonl(manifest_path)
    train = read_jsonl(train_path)
    if len(review) != EXPECTED_REVIEW_ROWS:
        raise ValueError(f"Review rows changed: {len(review)}")
    if len(manifest) != EXPECTED_ROWS or len(train) != EXPECTED_ROWS:
        raise ValueError("HEM manifest and frozen Train must both contain 66,854 rows")
    if any(row.get("split") != "train" for row in train):
        raise ValueError("Train snapshot contains a non-Train row")
    canonical_manifest, source_manifest_already_canonical = (
        canonicalize_manifest_order(train, manifest)
    )
    train_by_id = {str(row.get("id") or row.get("candidate_id")): row for row in train}
    manifest_by_id = {str(row["sample_id"]): row for row in canonical_manifest}
    if len(train_by_id) != EXPECTED_ROWS or set(train_by_id) != set(manifest_by_id):
        raise ValueError("HEM manifest membership differs from frozen Train")
    for sample_id, weighted in manifest_by_id.items():
        source = train_by_id[sample_id]
        if weighted.get("image_path") != source.get("image"):
            raise ValueError(f"HEM image path mismatch: {sample_id}")
        if weighted.get("gt_text") != (source.get("logical_text") or source.get("text")):
            raise ValueError(f"HEM GT mismatch: {sample_id}")
    weight_counts = Counter(
        str(float(row["sample_weight"])) for row in canonical_manifest
    )
    if dict(weight_counts) != EXPECTED_WEIGHTS:
        raise ValueError(f"Unexpected frozen HEM weights: {dict(weight_counts)}")

    decisions = Counter(row["manual_decision"] for row in review)
    genuine_ratio = decisions["genuine_ocr_error"] / len(review)
    if genuine_ratio < 0.70:
        raise ValueError("Reviewed genuine OCR error ratio is below 70%")
    excluded = [row for row in review if row["manual_decision"] in NON_GENUINE]
    if len(excluded) != 49:
        raise ValueError(f"Expected 49 manually excluded rows, found {len(excluded)}")
    wrong_weights = [
        row["sample_id"]
        for row in excluded
        if float(manifest_by_id[row["sample_id"]]["sample_weight"]) != 1.0
    ]
    if wrong_weights:
        raise ValueError(
            f"Manually rejected candidates were upweighted: {wrong_weights[:10]}"
        )

    strata: dict[str, Counter] = defaultdict(Counter)
    languages: dict[str, Counter] = defaultdict(Counter)
    for row in review:
        decision = row["manual_decision"]
        strata[f"{row['language']}_ed{row['ed']}"][decision] += 1
        languages[row["language"]][decision] += 1
    stratum_report = {}
    for name, counts in sorted(strata.items()):
        total = sum(counts.values())
        stratum_report[name] = {
            "reviewed": total,
            "decision_counts": dict(counts),
            "genuine_ocr_error_ratio": counts["genuine_ocr_error"] / total,
        }
    language_report = {}
    for name, counts in sorted(languages.items()):
        total = sum(counts.values())
        language_report[name] = {
            "reviewed": total,
            "decision_counts": dict(counts),
            "genuine_ocr_error_ratio": counts["genuine_ocr_error"] / total,
        }
    if set(language_report) != {"zh", "ug", "kk"}:
        raise ValueError("Review does not cover all three languages")
    if any(item["decision_counts"].get("genuine_ocr_error", 0) == 0 for item in language_report.values()):
        raise ValueError("At least one language has no genuine OCR errors in review")

    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    copies = {
        "train_hard_candidate_review.csv": review_path,
        "target_train_metadata.jsonl": train_path,
    }
    for name, source in copies.items():
        shutil.copy2(source, temporary / name)
    write_jsonl(temporary / "formal_hem_manifest.jsonl", canonical_manifest)
    frozen_names = [*copies, "formal_hem_manifest.jsonl"]
    hashes = {name: sha256(temporary / name) for name in frozen_names}
    report = {
        "status": "HEM_V1_FROZEN",
        "protocol_id": PROTOCOL_ID,
        "sampler_protocol": SAMPLER_PROTOCOL,
        "review_rows": len(review),
        "decision_counts": dict(decisions),
        "genuine_ocr_error_ratio": genuine_ratio,
        "required_ratio": 0.70,
        "review_by_language_ed": stratum_report,
        "review_by_language": language_report,
        "all_languages_contribute_genuine_errors": True,
        "manual_non_genuine_rows": len(excluded),
        "manual_non_genuine_all_weight_1": True,
        "manifest_rows": len(canonical_manifest),
        "weight_counts": dict(weight_counts),
        "all_train_rows_preserved": True,
        "manifest_order_matches_frozen_train": True,
        "source_manifest_already_in_train_order": source_manifest_already_canonical,
        "source_manifest_reordered_by_sample_id": (
            not source_manifest_already_canonical
        ),
        "source_manifest_sha256": sha256(manifest_path),
        "canonical_manifest_sha256": hashes["formal_hem_manifest.jsonl"],
        "training_policy": {
            "physical_gpus": [1],
            "world_size": 1,
            "batch_size_per_card": 32,
            "global_batch_size": 32,
            "control_world_size": 2,
            "control_batch_size_per_rank": 16,
            "optimizer_steps_per_epoch": 3065,
            "single_gpu_schedule": "merge_equal_width_batches_from_the_two_rank_control",
            "weights": {"ED=0": 1.0, "ED=1": 2.0, "ED=2": 1.5, "ED>=3": 1.0},
            "weighted_sampling": "within_ratio_bucket_with_replacement",
            "selection_metric": "Clean Dev Macro CER",
        },
        "test_evaluated": False,
        "frozen_file_hashes": hashes,
    }
    (temporary / "hem_v1_freeze.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.rename(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
