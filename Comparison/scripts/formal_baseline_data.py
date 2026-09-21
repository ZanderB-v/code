#!/usr/bin/env python3
"""Frozen data and label handling for public pretrained baseline comparisons."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SVTRV2_SCRIPTS = PROJECT_ROOT / "scripts" / "svtrv2"
if str(SVTRV2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SVTRV2_SCRIPTS))

from scripts.svtrv2.dual_order_protocol import (  # noqa: E402
    bidi_layout,
    normalize_logical_label,
)


LANGUAGES = ("zh", "ug", "kk")
FORMAL_IMPLEMENTATION_FILES = (
    "Comparison/scripts/pretrained_models.py",
    "Comparison/scripts/formal_baseline_data.py",
    "Comparison/scripts/formal_baseline_runtime.py",
    "Comparison/scripts/train_pretrained_baseline.py",
    "Comparison/scripts/evaluate_pretrained_baseline.py",
    "scripts/svtrv2/dual_order_protocol.py",
    "scripts/svtrv2/metrics_v1.py",
)
PREFLIGHT_IMPLEMENTATION_FILES = FORMAL_IMPLEMENTATION_FILES + (
    "Comparison/scripts/audit_pretrained_loading.py",
    "Comparison/scripts/audit_pretrained_formal_protocol.py",
    "Comparison/scripts/preflight_pretrained_baseline.py",
    "Comparison/scripts/preflight_formal_baseline_ddp.py",
    "Comparison/scripts/summarize_pretrained_preflight.py",
)


@dataclass(frozen=True)
class TextRecord:
    sample_id: str
    image: Path
    language: str
    logical_text: str
    train_text: str
    source: str
    split: str
    corruption: str = ""
    severity: int = 0


def load_protocol(path: Path | None = None) -> dict:
    path = path or (
        PROJECT_ROOT / "Comparison" / "pretrained_baselines_v1" / "protocol.json"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("test_policy") != "not_evaluated_during_model_development":
        raise ValueError("Comparison protocol must prohibit Test evaluation")
    if protocol.get("checkpoint_selection") != "clean_dev_macro_cer":
        raise ValueError("Checkpoint selection must remain Clean Dev Macro CER")
    return protocol


def read_dictionary(path: Path, use_space_char: bool = False) -> list[str]:
    characters = path.read_text(encoding="utf-8").splitlines()
    if not characters or any(len(character) != 1 for character in characters):
        raise ValueError("Every dictionary line must contain exactly one character")
    if len(set(characters)) != len(characters):
        raise ValueError("Character dictionary contains duplicates")
    if use_space_char:
        if " " in characters:
            raise ValueError(
                "Dictionary already contains a space while use_space_char is enabled"
            )
        characters.append(" ")
    return characters


def logical_and_visual(text: str, language: str) -> tuple[str, str]:
    logical, _ = normalize_logical_label(text)
    visual = bidi_layout(logical, language)["visual_text"]
    return logical, visual


def _existing_image(root: Path, row: dict, keys: Iterable[str]) -> Path:
    checked = []
    for key in keys:
        value = row.get(key)
        if not value:
            continue
        candidate = Path(value)
        candidate = candidate if candidate.is_absolute() else root / candidate
        checked.append(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No image for {row.get('id') or row.get('candidate_id')}; checked: "
        + ", ".join(map(str, checked))
    )


def iter_s50_records(protocol: dict) -> Iterator[TextRecord]:
    manifest = PROJECT_ROOT / protocol["synthetic_manifest"]
    image_root = PROJECT_ROOT / protocol["synthetic_root"]
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            language = row.get("language")
            if language not in LANGUAGES:
                raise ValueError(f"Invalid S50 language at line {line_number}: {language}")
            logical, expected_visual = logical_and_visual(
                row.get("logical_text") or "", language
            )
            stored = (
                row.get("ctc_text_u2")
                if language == "ug"
                else row.get("ctc_text")
            )
            if stored is None:
                stored = row.get("ctc_text") or row.get("logical_text")
            stored, _ = normalize_logical_label(stored or "")
            if stored != expected_visual:
                raise ValueError(
                    f"S50 U2/label mismatch for {row.get('id')} at line {line_number}"
                )
            yield TextRecord(
                sample_id=str(row.get("id") or f"s50_{line_number}"),
                image=_existing_image(image_root, row, ("formal_image", "image")),
                language=language,
                logical_text=logical,
                train_text=stored,
                source="s50",
                split="synthetic_train",
            )


def iter_target_records(protocol: dict, split: str) -> Iterator[TextRecord]:
    if split not in {"train", "dev"}:
        raise ValueError(
            f"Target split {split!r} is forbidden; only train/dev are available "
            "during model development"
        )
    metadata = PROJECT_ROOT / protocol["target_metadata"]
    image_root = metadata.parent
    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), 2):
            if row.get("split") != split:
                continue
            language = row.get("language")
            if language not in LANGUAGES:
                raise ValueError(
                    f"Invalid target language at line {line_number}: {language}"
                )
            logical, visual = logical_and_visual(
                row.get("logical_text") or row.get("text") or "", language
            )
            yield TextRecord(
                sample_id=str(
                    row.get("candidate_id")
                    or row.get("id")
                    or f"target_{split}_{line_number}"
                ),
                image=_existing_image(
                    image_root,
                    row,
                    ("image", "crop_path_rel", "image_path_resolved"),
                ),
                language=language,
                logical_text=logical,
                train_text=visual,
                source="target",
                split=split,
            )


def load_records(protocol: dict, dataset: str) -> list[TextRecord]:
    if dataset == "s50":
        records = list(iter_s50_records(protocol))
    elif dataset in {"target_train", "target_dev"}:
        records = list(iter_target_records(protocol, dataset.split("_", 1)[1]))
    else:
        raise ValueError(f"Unknown or forbidden dataset: {dataset}")
    if not records:
        raise ValueError(f"Dataset is empty: {dataset}")
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formal_implementation_hashes() -> dict[str, str]:
    return {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in FORMAL_IMPLEMENTATION_FILES
    }


def preflight_implementation_hashes() -> dict[str, str]:
    return {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in PREFLIGHT_IMPLEMENTATION_FILES
    }


def protocol_artifact_hashes(protocol: dict) -> dict[str, str]:
    """Hash the lightweight frozen inputs and all public initializations."""
    relative_paths = [
        protocol["dictionary"],
        protocol["synthetic_manifest"],
        protocol["target_metadata"],
        protocol["corruption_protocol"],
        protocol["implementation_source_provenance"],
    ]
    relative_paths.extend(
        spec["checkpoint"] for spec in protocol["models"].values()
    )
    return {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in relative_paths
    }


def load_corrupted_dev_records(protocol: dict) -> list[TextRecord]:
    frozen_path = PROJECT_ROOT / protocol["corruption_protocol"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen":
        raise ValueError("Corruption protocol is not frozen")
    if "not" not in str(frozen.get("test_policy", "")):
        raise ValueError("Frozen corruption protocol does not prohibit Test")
    protocol_root = frozen_path.parent
    manifest = protocol_root / "dev_manifest.jsonl"
    expected_hash = frozen.get("dev_manifest_sha256")
    actual_hash = sha256_file(manifest)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"Frozen corrupted-dev manifest hash mismatch: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    records = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("split") != "dev":
                raise ValueError(
                    f"Non-dev row in corrupted-dev manifest line {line_number}"
                )
            language = row.get("language")
            if language not in LANGUAGES:
                raise ValueError(
                    f"Invalid language in corrupted-dev line {line_number}: {language}"
                )
            logical, visual = logical_and_visual(row.get("logical_text") or "", language)
            image = Path(row["output_image"])
            image = image if image.is_absolute() else protocol_root / image
            if not image.is_file():
                raise FileNotFoundError(image)
            records.append(
                TextRecord(
                    sample_id=str(row["sample_id"]),
                    image=image,
                    language=language,
                    logical_text=logical,
                    train_text=visual,
                    source="frozen_corruption_dev",
                    split="dev_corrupted",
                    corruption=str(row["corruption"]),
                    severity=int(row["severity"]),
                )
            )
    if not records:
        raise ValueError("Frozen corrupted-dev manifest is empty")
    return records


def encode_text(text: str, char_to_id: dict[str, int]) -> list[int]:
    unknown = sorted(set(text) - set(char_to_id), key=ord)
    if unknown:
        values = ", ".join(f"U+{ord(char):04X}" for char in unknown)
        raise ValueError(f"Label contains characters outside the dictionary: {values}")
    return [char_to_id[character] for character in text]


def ctc_required_timesteps(text: str) -> int:
    return len(text) + sum(
        current == previous for previous, current in zip(text, text[1:])
    )
