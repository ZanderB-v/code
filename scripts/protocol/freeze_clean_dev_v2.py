#!/usr/bin/env python3
"""Freeze model-blind Clean Dev V2 without modifying or excluding V1 rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from text_normalization_v2 import PROTOCOL_ID as NORMALIZATION_ID
from text_normalization_v2 import normalize_text_v2


PROTOCOL_ID = "CLEAN_DEV_V2_VERIFIED"
LANGUAGES = ("zh", "ug", "kk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("01_data_preparation/clean_dev_v2_protocol/blind_audit_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("01_data_preparation/clean_dev_v2_verified"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalization_yaml(implementation_sha256: str) -> str:
    return f"""protocol_id: {NORMALIZATION_ID}
status: frozen
implementation: scripts/protocol/text_normalization_v2.py
implementation_sha256: {implementation_sha256}
applies_identically_to:
  - ground_truth
  - prediction
operations_in_order:
  - reject_disallowed_zero_width_and_bidi_controls
  - unicode_NFC
  - map_U_FF5E_FULLWIDTH_TILDE_to_U_007E_ASCII_TILDE
  - map_all_Unicode_whitespace_to_U_0020_ASCII_SPACE
  - collapse_consecutive_ASCII_spaces
  - trim_outer_ASCII_spaces
preserved:
  - internal_word_spaces
  - spaces_between_Han_phrases
  - all_punctuation_except_width_equivalence_for_tilde
  - Uyghur_and_Kazakh_word_boundaries
explicitly_disabled:
  - global_NFKC
  - punctuation_removal
  - Han_space_removal
  - silent_zero_width_removal
test_evaluated: false
"""


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    audit = args.audit_dir if args.audit_dir.is_absolute() else root / args.audit_dir
    output = args.output if args.output.is_absolute() else root / args.output
    freeze_path = output / "frozen_clean_dev_v2_manifest.json"
    if freeze_path.exists():
        raise FileExistsError(f"Clean Dev V2 is already frozen: {freeze_path}")
    output.mkdir(parents=True, exist_ok=True)

    audit_summary_path = audit / "blind_audit_summary.json"
    audit_summary = read_json(audit_summary_path)
    if audit_summary.get("status") != "CLEAN_DEV_V2_BLIND_AUDIT_COMPLETE":
        raise ValueError("The full 951-row model-blind audit is not complete")
    if audit_summary.get("model_blind") is not True:
        raise ValueError("Audit is not model-blind")
    if audit_summary.get("model_predictions_loaded") is not False:
        raise ValueError("Audit loaded model predictions")
    if audit_summary.get("rows") != 951 or audit_summary.get("pending_rows") != 0:
        raise ValueError("Unexpected blind-audit coverage")
    if audit_summary.get("invalid_rows"):
        raise ValueError("Blind audit contains invalid rows")
    if audit_summary.get("test_evaluated") is not False:
        raise ValueError("Blind audit does not preserve Test prohibition")

    raw_manifest_path = audit / "clean_dev_v1_raw_manifest.jsonl"
    raw_rows = read_jsonl(raw_manifest_path)
    patch_source = audit / "label_patch_v2_candidate.csv"
    patch_rows = read_csv(patch_source)
    if len(raw_rows) != 951 or len(patch_rows) != 15:
        raise ValueError(
            f"Unexpected V2 source sizes: raw={len(raw_rows)}, patch={len(patch_rows)}"
        )
    patch_by_key = {(row["language"], row["sample_id"]): row for row in patch_rows}
    if len(patch_by_key) != len(patch_rows):
        raise ValueError("Duplicate patch keys")

    reviews = []
    review_hashes = {}
    for language in LANGUAGES:
        path = audit / f"{language}_blind_review.csv"
        review_hashes[language] = sha256(path)
        reviews.extend(read_csv(path))
    if len(reviews) != 951:
        raise ValueError("Blind review rows do not cover all 951 samples")
    review_by_key = {(row["language"], row["sample_id"]): row for row in reviews}
    if len(review_by_key) != len(reviews):
        raise ValueError("Duplicate review keys")

    final_rows = []
    normalized_change_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    consumed_patches = set()
    for raw in raw_rows:
        key = (raw["language"], raw["sample_id"])
        review = review_by_key.get(key)
        if review is None:
            raise ValueError(f"Missing blind review: {key}")
        if review["source_record_sha256"] != raw["source_record_sha256"]:
            raise ValueError(f"Blind review source fingerprint mismatch: {key}")
        decision_counts[review["decision"]] += 1
        patch = patch_by_key.get(key)
        if review["decision"] == "gt_annotation_error":
            if patch is None:
                raise ValueError(f"GT error has no patch: {key}")
            if patch["old_gt"] != raw["raw_gt"]:
                raise ValueError(f"Patch old GT mismatch: {key}")
            patched_gt = patch["new_gt"]
            consumed_patches.add(key)
        else:
            if patch is not None:
                raise ValueError(f"Non-GT-error row has a patch: {key}")
            patched_gt = raw["raw_gt"]
        normalized_gt = normalize_text_v2(patched_gt)
        if normalized_gt != patched_gt:
            if "\uFF5E" in patched_gt:
                normalized_change_counts["fullwidth_tilde_to_ascii"] += 1
            if any(character.isspace() and character != " " for character in patched_gt):
                normalized_change_counts["whitespace_to_ascii"] += 1
            if "  " in patched_gt or patched_gt != patched_gt.strip(" "):
                normalized_change_counts["space_collapse_or_trim"] += 1
        final_rows.append(
            {
                "sample_id": raw["sample_id"],
                "language": raw["language"],
                "split": "dev",
                "image_path": raw["image_path"],
                "raw_gt": raw["raw_gt"],
                "patched_gt": patched_gt,
                "normalized_gt": normalized_gt,
                "audit_decision": review["decision"],
                "excluded": False,
                "source_record_sha256": raw["source_record_sha256"],
            }
        )
    if consumed_patches != set(patch_by_key):
        raise ValueError("At least one patch was not consumed")

    manifest_path = output / "clean_dev_v2_verified_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    labels_dir = output / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for language in LANGUAGES:
        subset = [row for row in final_rows if row["language"] == language]
        with (labels_dir / f"{language}.txt").open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(f"{row['image_path']}\t{row['normalized_gt']}\n")
    with (labels_dir / "all.txt").open("w", encoding="utf-8") as handle:
        for row in final_rows:
            handle.write(f"{row['image_path']}\t{row['normalized_gt']}\n")

    patch_path = output / "label_patch_v2.csv"
    shutil.copyfile(patch_source, patch_path)
    implementation_path = root / "scripts/protocol/text_normalization_v2.py"
    implementation_hash = sha256(implementation_path)
    normalization_path = output / "normalization_v2.yaml"
    normalization_path.write_text(
        normalization_yaml(implementation_hash), encoding="utf-8"
    )

    artifact_paths = {
        "clean_dev_v2_verified_manifest.jsonl": manifest_path,
        "label_patch_v2.csv": patch_path,
        "normalization_v2.yaml": normalization_path,
        "labels/all.txt": labels_dir / "all.txt",
        "labels/zh.txt": labels_dir / "zh.txt",
        "labels/ug.txt": labels_dir / "ug.txt",
        "labels/kk.txt": labels_dir / "kk.txt",
    }
    artifact_hashes = {name: sha256(path) for name, path in artifact_paths.items()}
    frozen = {
        "status": "CLEAN_DEV_V2_VERIFIED_FROZEN",
        "protocol_id": PROTOCOL_ID,
        "normalization_protocol": NORMALIZATION_ID,
        "rows": len(final_rows),
        "language_counts": dict(Counter(row["language"] for row in final_rows)),
        "patch_rows": len(patch_rows),
        "audit_decision_counts": dict(decision_counts),
        "normalization_change_counts": dict(normalized_change_counts),
        "source": {
            "blind_audit_summary_sha256": sha256(audit_summary_path),
            "clean_dev_v1_raw_manifest_sha256": sha256(raw_manifest_path),
            "blind_review_sha256": review_hashes,
            "normalization_implementation_sha256": implementation_hash,
        },
        "artifact_sha256": artifact_hashes,
        "all_original_rows_preserved": True,
        "ambiguous_or_crop_rows_excluded": False,
        "original_clean_dev_modified": False,
        "model_predictions_used_for_label_audit": False,
        "selection_metric": "Clean Dev V2 Macro CER",
        "checkpoint_reselection_required": True,
        "formal_new_module_training_allowed": False,
        "test_evaluated": False,
    }
    freeze_path.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(frozen, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
