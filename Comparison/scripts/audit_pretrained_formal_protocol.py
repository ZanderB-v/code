#!/usr/bin/env python3
"""Fail-closed all-label and image audit before long baseline training."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from formal_baseline_data import (
    LANGUAGES,
    PROJECT_ROOT,
    ctc_required_timesteps,
    encode_text,
    load_protocol,
    load_records,
    protocol_artifact_hashes,
    read_dictionary,
    sha256_file,
)


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[round((len(values) - 1) * q)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "Comparison/pretrained_baselines_v1/outputs/formal_protocol_audit.json"
        ),
    )
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    protocol_path = args.protocol or (
        PROJECT_ROOT / "Comparison/pretrained_baselines_v1/protocol.json"
    )
    dictionary_entries = read_dictionary(PROJECT_ROOT / protocol["dictionary"])
    if len(dictionary_entries) != int(protocol["expected_dictionary_size"]):
        raise ValueError("Frozen dictionary size mismatch")
    characters = read_dictionary(
        PROJECT_ROOT / protocol["dictionary"],
        bool(protocol["use_space_char"]),
    )
    if len(characters) != int(protocol["expected_recognition_symbol_count"]):
        raise ValueError("Runtime recognition-symbol count mismatch")
    char_to_id = {character: index + 1 for index, character in enumerate(characters)}
    capacity = int(protocol["max_label_length"])
    errors = []
    datasets = {}

    for dataset_name in ("s50", "target_train", "target_dev"):
        records = load_records(protocol, dataset_name)
        counts = Counter(record.language for record in records)
        expected_counts = protocol["expected_counts"][dataset_name]
        actual_counts = {language: counts[language] for language in LANGUAGES}
        if actual_counts != expected_counts:
            errors.append(
                f"{dataset_name} language counts changed: "
                f"expected={expected_counts}, actual={actual_counts}"
            )
        lengths = defaultdict(list)
        ctc_lengths = defaultdict(list)
        image_paths = set()
        duplicate_ids = []
        seen_ids = set()
        for record in records:
            if record.sample_id in seen_ids:
                duplicate_ids.append(record.sample_id)
            seen_ids.add(record.sample_id)
            image_paths.add(str(record.image.resolve()))
            encode_text(record.train_text, char_to_id)
            lengths[record.language].append(len(record.train_text))
            ctc_lengths[record.language].append(
                ctc_required_timesteps(record.train_text)
            )
        if duplicate_ids:
            errors.append(
                f"{dataset_name} contains duplicate sample IDs: {duplicate_ids[:10]}"
            )
        if len(image_paths) != len(records):
            errors.append(
                f"{dataset_name} maps {len(records)} rows to only "
                f"{len(image_paths)} unique image paths"
            )
        language_summary = {}
        for language in LANGUAGES:
            values = lengths[language]
            required = ctc_lengths[language]
            if not values:
                errors.append(f"{dataset_name} has no {language} records")
                continue
            if max(values) > capacity:
                errors.append(
                    f"{dataset_name}/{language} label length {max(values)} exceeds "
                    f"capacity {capacity}"
                )
            language_summary[language] = {
                "rows": counts[language],
                "length": {
                    "min": min(values),
                    "p50": percentile(values, 0.50),
                    "p90": percentile(values, 0.90),
                    "p95": percentile(values, 0.95),
                    "p99": percentile(values, 0.99),
                    "max": max(values),
                },
                "ctc_required_timesteps": {
                    "max": max(required),
                    "over_128": sum(value > capacity for value in required),
                },
            }
        datasets[dataset_name] = {
            "rows": len(records),
            "unique_ids": len(seen_ids),
            "unique_image_paths": len(image_paths),
            "languages": language_summary,
        }

    model_capacity = {}
    for model, spec in protocol["models"].items():
        output_length = int(spec["output_sequence_length"])
        if model in {"crnn", "svtr"}:
            over = []
            for dataset_name, summary in datasets.items():
                for language, values in summary["languages"].items():
                    required = values["ctc_required_timesteps"]["max"]
                    if required > output_length:
                        over.append(f"{dataset_name}/{language}:{required}")
            if over:
                errors.append(
                    f"{model} CTC sequence capacity {output_length} is insufficient: "
                    + ", ".join(over)
                )
        model_capacity[model] = {
            "input_size_hw": spec["input_size_hw"],
            "output_sequence_length": output_length,
            "all_training_and_dev_labels_fit": not any(
                error.startswith(f"{model} ") for error in errors
            ),
        }

    result = {
        "status": "passed" if not errors else "failed",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "artifact_sha256": protocol_artifact_hashes(protocol),
        "input_policy": protocol["input_policy"],
        "dictionary_file_entries": len(dictionary_entries),
        "recognition_symbol_count": len(characters),
        "use_space_char": bool(protocol["use_space_char"]),
        "datasets": datasets,
        "model_capacity": model_capacity,
        "test_accessed": False,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)
    print("PRETRAINED_BASELINES_FORMAL_PROTOCOL_AUDIT_OK")


if __name__ == "__main__":
    main()
