#!/usr/bin/env python3
"""Merge both completed Dev reviews into an auditable Clean Dev V3 protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROTOCOL_ID = "CLEAN_DEV_V3_ADJUDICATED"
LANGUAGES = ("zh", "ug", "kk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("01_data_preparation/clean_dev_v3_adjudicated"),
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def keyed(rows: list[dict[str, str]], source: str) -> dict[tuple[str, str], dict[str, str]]:
    result = {}
    for row in rows:
        key = (row["language"], row["sample_id"])
        if key in result:
            raise ValueError(f"Duplicate {source} row: {key}")
        result[key] = row
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"Output exists; use --replace: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    sys.path.insert(0, str(root / "scripts" / "protocol"))
    from text_normalization_v2 import PROTOCOL_ID as NORMALIZATION_ID
    from text_normalization_v2 import normalize_text_v2

    blind_dir = root / "01_data_preparation/clean_dev_v2_protocol/blind_audit_v1"
    model_dir = root / "04_model_training/eval_reports/S50_M3_error_review_v1"
    override_path = (
        root / "01_data_preparation/clean_dev_v3_protocol/adjudication_overrides.csv"
    )
    raw_path = blind_dir / "clean_dev_v1_raw_manifest.jsonl"
    model_patch_path = model_dir / "label_patch_v1.csv"
    blind_patch_path = blind_dir / "label_patch_v2_candidate.csv"
    model_review_path = model_dir / "all_errors.csv"

    raw_rows = read_jsonl(raw_path)
    if len(raw_rows) != 951:
        raise ValueError(f"Expected 951 raw Dev rows, got {len(raw_rows)}")
    raw_by_key = {
        (row["language"], row["sample_id"]): row for row in raw_rows
    }
    if len(raw_by_key) != len(raw_rows):
        raise ValueError("Raw Dev contains duplicate sample keys")

    blind_reviews = []
    blind_review_paths = []
    for language in LANGUAGES:
        path = blind_dir / f"{language}_blind_review.csv"
        blind_review_paths.append(path)
        blind_reviews.extend(read_csv(path))
    if len(blind_reviews) != 951:
        raise ValueError("The model-blind review does not cover all 951 rows")
    blind_by_key = keyed(blind_reviews, "blind review")
    model_reviews = read_csv(model_review_path)
    if len(model_reviews) != 108:
        raise ValueError("Unexpected model-selected review size")
    model_by_key = keyed(model_reviews, "model-selected review")
    model_patches = keyed(read_csv(model_patch_path), "model-selected patch")
    blind_patches = keyed(read_csv(blind_patch_path), "blind patch")
    overrides = keyed(read_csv(override_path), "adjudication override")

    for key, raw in raw_by_key.items():
        blind = blind_by_key.get(key)
        if blind is None:
            raise ValueError(f"Missing blind review: {key}")
        if blind["current_gt"] != raw["raw_gt"]:
            raise ValueError(f"Blind-review GT mismatch: {key}")
        if blind["source_record_sha256"] != raw["source_record_sha256"]:
            raise ValueError(f"Blind-review fingerprint mismatch: {key}")
    for source_name, source in (
        ("model patch", model_patches),
        ("blind patch", blind_patches),
        ("override", overrides),
    ):
        unknown = sorted(set(source) - set(raw_by_key))
        if unknown:
            raise ValueError(f"{source_name} references unknown rows: {unknown}")

    ledger = []
    final_rows = []
    patch_rows = []
    excluded_rows = []
    unresolved = []
    inherited_model_patch_keys = set()
    inherited_blind_patch_keys = set()
    accounted_blind_patch_keys = set()

    for raw in raw_rows:
        key = (raw["language"], raw["sample_id"])
        blind = blind_by_key[key]
        model = model_by_key.get(key)
        model_patch = model_patches.get(key)
        blind_patch = blind_patches.get(key)
        override = overrides.get(key)

        if model is not None and model["gt"] != raw["raw_gt"]:
            raise ValueError(f"Model-selected review GT mismatch: {key}")
        for source_name, patch in (
            ("model", model_patch),
            ("blind", blind_patch),
        ):
            if patch is not None and patch["old_gt"] != raw["raw_gt"]:
                raise ValueError(f"{source_name} patch old GT mismatch: {key}")

        crop_sources = []
        if blind["decision"] == "crop_issue":
            crop_sources.append("blind_audit_crop_issue")
        if model is not None and model["manual_decision"] == "image_or_crop_issue":
            crop_sources.append("model_selected_image_or_crop_issue")

        excluded = bool(crop_sources)
        exclusion_reason = ";".join(crop_sources)
        override_action = override["action"] if override else ""
        if override_action == "exclude":
            excluded = True
            exclusion_reason = override["reason"]
        elif override_action not in ("", "patch", "keep"):
            raise ValueError(f"Unsupported override action for {key}: {override_action}")

        candidates = []
        if model_patch is not None:
            candidates.append(("model_selected_review_v1", model_patch["new_gt"]))
        if blind_patch is not None:
            candidates.append(("model_blind_review_v2", blind_patch["new_gt"]))
        if override_action == "patch":
            candidates.append(("explicit_adjudication_override", override["corrected_gt"]))

        normalized_candidates: dict[str, list[str]] = {}
        for source, candidate in candidates:
            normalized = normalize_text_v2(candidate)
            normalized_candidates.setdefault(normalized, []).append(source)

        patched_gt = raw["raw_gt"]
        patch_sources = []
        if not excluded and normalized_candidates:
            if override_action == "patch":
                patched_gt = normalize_text_v2(override["corrected_gt"])
                patch_sources = ["explicit_adjudication_override"]
            elif len(normalized_candidates) == 1:
                patched_gt, patch_sources = next(iter(normalized_candidates.items()))
            else:
                unresolved.append(
                    {
                        "sample_id": key[1],
                        "language": key[0],
                        "raw_gt": raw["raw_gt"],
                        "model_selected_correction": (
                            model_patch["new_gt"] if model_patch else ""
                        ),
                        "blind_correction": blind_patch["new_gt"] if blind_patch else "",
                    }
                )

        if model_patch is not None and (
            excluded
            or normalize_text_v2(model_patch["new_gt"]) == normalize_text_v2(patched_gt)
        ):
            inherited_model_patch_keys.add(key)
        if blind_patch is not None and (
            excluded
            or normalize_text_v2(blind_patch["new_gt"]) == normalize_text_v2(patched_gt)
        ):
            inherited_blind_patch_keys.add(key)
        if blind_patch is not None and (
            key in inherited_blind_patch_keys or override_action in ("patch", "exclude")
        ):
            accounted_blind_patch_keys.add(key)

        normalized_gt = normalize_text_v2(patched_gt)
        if not excluded and normalized_gt != normalize_text_v2(raw["raw_gt"]):
            patch_rows.append(
                {
                    "sample_id": key[1],
                    "language": key[0],
                    "image_path": raw["image_path"],
                    "old_gt": raw["raw_gt"],
                    "new_gt": normalized_gt,
                    "sources": ";".join(patch_sources),
                }
            )
        if excluded:
            excluded_rows.append(
                {
                    "sample_id": key[1],
                    "language": key[0],
                    "image_path": raw["image_path"],
                    "raw_gt": raw["raw_gt"],
                    "reason": exclusion_reason,
                }
            )

        ambiguous_sources = []
        if blind["decision"] == "ambiguous_image":
            ambiguous_sources.append("blind_audit")
        if model is not None and model["manual_decision"] == "ambiguous_image":
            ambiguous_sources.append("model_selected_review")

        final_rows.append(
            {
                "sample_id": key[1],
                "language": key[0],
                "split": "dev",
                "image_path": raw["image_path"],
                "raw_gt": raw["raw_gt"],
                "patched_gt": patched_gt,
                "normalized_gt": normalized_gt,
                "excluded": excluded,
                "exclusion_reason": exclusion_reason,
                "ambiguous_review_flag": bool(ambiguous_sources),
                "source_record_sha256": raw["source_record_sha256"],
            }
        )
        ledger.append(
            {
                "sample_id": key[1],
                "language": key[0],
                "image_path": raw["image_path"],
                "raw_gt": raw["raw_gt"],
                "model_selected_decision": model["manual_decision"] if model else "",
                "model_selected_corrected_gt": model_patch["new_gt"] if model_patch else "",
                "blind_decision": blind["decision"],
                "blind_corrected_gt": blind_patch["new_gt"] if blind_patch else "",
                "override_action": override_action,
                "override_reason": override["reason"] if override else "",
                "final_gt": normalized_gt,
                "excluded": excluded,
                "exclusion_reason": exclusion_reason,
                "ambiguous_review_flag": bool(ambiguous_sources),
                "patch_sources": ";".join(patch_sources),
            }
        )

    if unresolved:
        path = output / "unresolved_conflicts.csv"
        write_csv(path, unresolved, list(unresolved[0]))
        raise ValueError(f"Unresolved review conflicts remain: {len(unresolved)}; see {path}")
    if inherited_model_patch_keys != set(model_patches):
        missing = sorted(set(model_patches) - inherited_model_patch_keys)
        raise ValueError(f"Model-selected patches were not inherited: {missing}")
    if accounted_blind_patch_keys != set(blind_patches):
        missing = sorted(set(blind_patches) - accounted_blind_patch_keys)
        raise ValueError(f"Blind-audit patches were not adjudicated: {missing}")

    manifest_path = output / "clean_dev_v3_adjudicated_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_csv(
        output / "review_inheritance_ledger.csv",
        ledger,
        list(ledger[0]),
    )
    write_csv(
        output / "label_patch_v3.csv",
        patch_rows,
        ["sample_id", "language", "image_path", "old_gt", "new_gt", "sources"],
    )
    write_csv(
        output / "excluded_samples_v3.csv",
        excluded_rows,
        ["sample_id", "language", "image_path", "raw_gt", "reason"],
    )

    labels_dir = output / "labels"
    labels_dir.mkdir()
    active_rows = [row for row in final_rows if not row["excluded"]]
    for language in LANGUAGES:
        subset = [row for row in active_rows if row["language"] == language]
        with (labels_dir / f"{language}.txt").open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(f"{row['image_path']}\t{row['normalized_gt']}\n")
    with (labels_dir / "all.txt").open("w", encoding="utf-8") as handle:
        for row in active_rows:
            handle.write(f"{row['image_path']}\t{row['normalized_gt']}\n")

    normalization_source = root / "01_data_preparation/clean_dev_v2_verified/normalization_v2.yaml"
    shutil.copyfile(normalization_source, output / "normalization_v2.yaml")

    artifacts = {
        "clean_dev_v3_adjudicated_manifest.jsonl": manifest_path,
        "review_inheritance_ledger.csv": output / "review_inheritance_ledger.csv",
        "label_patch_v3.csv": output / "label_patch_v3.csv",
        "excluded_samples_v3.csv": output / "excluded_samples_v3.csv",
        "normalization_v2.yaml": output / "normalization_v2.yaml",
        "labels/all.txt": labels_dir / "all.txt",
        "labels/zh.txt": labels_dir / "zh.txt",
        "labels/ug.txt": labels_dir / "ug.txt",
        "labels/kk.txt": labels_dir / "kk.txt",
    }
    source_hashes = {
        "raw_manifest": sha256(raw_path),
        "model_selected_review": sha256(model_review_path),
        "model_selected_patches": sha256(model_patch_path),
        "blind_patches": sha256(blind_patch_path),
        "adjudication_overrides": sha256(override_path),
        **{
            f"blind_review_{path.stem.split('_')[0]}": sha256(path)
            for path in blind_review_paths
        },
    }
    unique_patch_candidate_keys = set(model_patches) | set(blind_patches)
    excluded_keys = {
        (row["language"], row["sample_id"]) for row in excluded_rows
    }
    active_patch_keys = {
        (row["language"], row["sample_id"]) for row in patch_rows
    }
    summary = {
        "status": "CLEAN_DEV_V3_ADJUDICATED_FROZEN",
        "protocol_id": PROTOCOL_ID,
        "normalization_protocol": NORMALIZATION_ID,
        "source_rows": len(final_rows),
        "evaluated_rows": len(active_rows),
        "language_source_counts": dict(Counter(row["language"] for row in final_rows)),
        "language_evaluated_counts": dict(Counter(row["language"] for row in active_rows)),
        "active_patch_rows": len(patch_rows),
        "unique_patch_candidate_rows": len(unique_patch_candidate_keys),
        "patch_candidates_suppressed_by_exclusion": len(
            unique_patch_candidate_keys & excluded_keys
        ),
        "patch_candidates_equivalent_after_normalization": len(
            unique_patch_candidate_keys - excluded_keys - active_patch_keys
        ),
        "excluded_rows": len(excluded_rows),
        "ambiguous_rows_retained": sum(
            row["ambiguous_review_flag"] and not row["excluded"] for row in final_rows
        ),
        "model_selected_patch_rows_inherited": len(inherited_model_patch_keys),
        "blind_patch_rows_inherited": len(inherited_blind_patch_keys),
        "blind_patch_rows_accounted": len(accounted_blind_patch_keys),
        "blind_patch_rows_superseded_by_adjudication": len(
            accounted_blind_patch_keys - inherited_blind_patch_keys
        ),
        "unresolved_conflicts": 0,
        "crop_policy": "exclude union of crop/image_or_crop decisions from either completed review",
        "ambiguous_policy": "retain and flag; do not exclude without a crop decision",
        "contains_model_informed_manual_adjudication": True,
        "selection_bias_control": "rescore every saved checkpoint for every model with this single frozen protocol",
        "source_sha256": source_hashes,
        "artifact_sha256": {name: sha256(path) for name, path in artifacts.items()},
        "supersedes_for_future_development": "CLEAN_DEV_V2_VERIFIED",
        "training_run": False,
        "test_evaluated": False,
    }
    (output / "frozen_clean_dev_v3_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
