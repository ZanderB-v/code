#!/usr/bin/env python3
"""Create a strict, machine-checkable loading audit for one public baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pretrained_models import (
    PROJECT_ROOT,
    build_model_bundle,
    sha256_file,
    transfer_checkpoint_fail_closed,
)
from formal_baseline_data import formal_implementation_hashes


def read_dictionary(path: Path) -> list[str]:
    characters = path.read_text(encoding="utf-8").splitlines()
    if not characters or any(len(character) != 1 for character in characters):
        raise ValueError("Every dictionary line must contain exactly one character")
    if len(set(characters)) != len(characters):
        raise ValueError("Character dictionary contains duplicates")
    return characters


def verify_source_provenance(protocol: dict, model_name: str) -> dict:
    manifest_path = PROJECT_ROOT / protocol["implementation_source_provenance"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_name = "mmocr" if model_name == "svtr" else "parseq_model_hub"
    source = manifest["sources"][source_name]
    source_root = PROJECT_ROOT / source["local_path"]
    checked = {}
    checked_licenses = {}
    for relative_path, license_spec in source["licenses"].items():
        license_path = source_root / relative_path
        actual_license_hash = sha256_file(license_path)
        if actual_license_hash != license_spec["sha256"]:
            raise ValueError(f"License hash mismatch: {license_path}")
        checked_licenses[relative_path] = {
            "identifier": license_spec["identifier"],
            "sha256": actual_license_hash,
        }
    for relative_path, expected_hash in source["key_source_files"].items():
        path = source_root / relative_path
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Implementation source hash mismatch for {path}: "
                f"expected={expected_hash}, actual={actual_hash}"
            )
        checked[relative_path] = actual_hash
    return {
        "manifest": str(manifest_path),
        "source": source_name,
        "repository": source["repository"],
        "release": source["release"],
        "licenses": checked_licenses,
        "verified_files": checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("crnn", "svtr", "parseq", "abinet"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "Comparison" / "pretrained_baselines_v1" / "protocol.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Comparison" / "pretrained_baselines_v1" / "outputs",
    )
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_sha256 = sha256_file(args.protocol)
    implementation_sha256 = formal_implementation_hashes()
    model_spec = protocol["models"][args.model]
    source_provenance = verify_source_provenance(protocol, args.model)
    dictionary_path = PROJECT_ROOT / protocol["dictionary"]
    dictionary_entries = read_dictionary(dictionary_path)
    expected_size = int(protocol["expected_dictionary_size"])
    if len(dictionary_entries) != expected_size:
        raise ValueError(
            f"Dictionary size mismatch: expected {expected_size}, got {len(dictionary_entries)}"
        )
    characters = list(dictionary_entries)
    if protocol.get("use_space_char"):
        if " " in characters:
            raise ValueError("Dictionary already contains space")
        characters.append(" ")
    if len(characters) != int(protocol["expected_recognition_symbol_count"]):
        raise ValueError("Runtime recognition-symbol count mismatch")

    checkpoint = PROJECT_ROOT / model_spec["checkpoint"]
    actual_sha256 = sha256_file(checkpoint)
    expected_sha256 = model_spec["checkpoint_sha256"]
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Checkpoint SHA256 mismatch for {args.model}: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )

    torch.manual_seed(20260815)
    bundle = build_model_bundle(
        args.model,
        checkpoint,
        len(characters),
        int(protocol["max_label_length"]),
    )
    audit = transfer_checkpoint_fail_closed(
        bundle, float(model_spec["minimum_backbone_loaded_ratio"])
    )
    audit.update(
        {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha256,
            "implementation_sha256": implementation_sha256,
            "model": args.model,
            "display_name": model_spec["display_name"],
            "implementation": model_spec["implementation"],
            "pretraining_origin": model_spec["pretraining_origin"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_sha256,
            "dictionary": str(dictionary_path),
            "dictionary_file_entries": len(dictionary_entries),
            "recognition_symbol_count": len(characters),
            "use_space_char": bool(protocol.get("use_space_char")),
            "architecture_output_classes": (
                len(characters) + 1
            ),
            "implementation_source_provenance": source_provenance,
            "runtime_compatibility_notes": bundle.runtime_compatibility_notes,
            "special_class_note": (
                "The 4891-entry dictionary file is unchanged and the frozen "
                "use_space_char policy adds one shared recognition symbol. CTC "
                "adds one blank; autoregressive baselines add architecture-specific "
                "EOS/BOS/PAD tokens."
            ),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / f"{args.model}_loading_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)

    initialization_path = args.output_dir / f"{args.model}_4891_initialization.pth"
    torch.save(
        {
            "state_dict": bundle.model.state_dict(),
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha256,
            "implementation_sha256": implementation_sha256,
            "model": args.model,
            "dictionary_file_entries": len(dictionary_entries),
            "recognition_symbol_count": len(characters),
            "source_checkpoint_sha256": actual_sha256,
            "loading_audit": audit,
        },
        initialization_path,
    )
    print(f"LOAD_AUDIT_OK model={args.model} report={audit_path}")


if __name__ == "__main__":
    main()
