#!/usr/bin/env python3
"""Apply count-only code/doc updates after the target conflict-row repair."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


REPLACEMENTS = {
    "scripts/protocol/freeze_experiment_v1.py": (
        (
            '    "dev": {"zh": 346, "ug": 301, "kk": 310},\n'
            '    "test": {"zh": 1142, "ug": 935, "kk": 993},',
            '    "dev": {"zh": 346, "ug": 295, "kk": 310},\n'
            '    "test": {"zh": 1142, "ug": 931, "kk": 993},',
        ),
    ),
    "scripts/maintenance/freeze_dataset_v1.py": (
        (
            '    "dev": {"zh": 346, "ug": 301, "kk": 310},\n'
            '    "test": {"zh": 1142, "ug": 935, "kk": 993},',
            '    "dev": {"zh": 346, "ug": 295, "kk": 310},\n'
            '    "test": {"zh": 1142, "ug": 931, "kk": 993},',
        ),
    ),
    "scripts/svtrv2/prepare_e0_target_only.py": (
        (
            '        "dev": 957,\n'
            '        "test": 3070,',
            '        "dev": 951,\n'
            '        "test": 3066,',
        ),
    ),
    "scripts/svtrv2/prepare_e1_target_only.py": (
        (
            '        "dev": 957,\n'
            '        "test": 3070,',
            '        "dev": 951,\n'
            '        "test": 3066,',
        ),
    ),
    "scripts/svtrv2/prepare_e5_d2_to_target.py": (
        (
            '    "dev": {"zh": 346, "ug": 301, "kk": 310},\n'
            '    "test": {"zh": 1142, "ug": 935, "kk": 993},',
            '    "dev": {"zh": 346, "ug": 295, "kk": 310},\n'
            '    "test": {"zh": 1142, "ug": 931, "kk": 993},',
        ),
    ),
    "00_docs/EXPERIMENT_PROTOCOL_V1.md": (
        (
            "Status: frozen after successful execution of\n"
            "`scripts/protocol/freeze_experiment_v1.py`.",
            "Status: provisional. Target conflict revision applied; final freeze "
            "must be rerun after the synthetic shard duplication repair.",
        ),
        (
            "| dev | 346 | 301 | 310 | 957 |\n"
            "| test | 1,142 | 935 | 993 | 3,070 |",
            "| dev | 346 | 295 | 310 | 951 |\n"
            "| test | 1,142 | 931 | 993 | 3,066 |",
        ),
        (
            "The target benchmark is frozen. Do not re-render, re-split, remove, "
            "relabel, or\notherwise tune the test set after this protocol is accepted.",
            "Ten objectively invalid Uyghur rows were excluded before final freeze: "
            "five exact-image groups carried two different labels. The crop files "
            "remain for provenance in the dataset directory. No model metric was "
            "used to choose these exclusions.\n\nAfter the corrected protocol is "
            "accepted, do not re-render, re-split, remove, relabel, or otherwise "
            "tune the test set.",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/data_home/wudayu/experiments/multilingual_meme_ocr/"
            "svtrv2_line_recognition"
        ),
    )
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".count_revision_tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    changed = []
    already_current = []
    for relative, replacements in REPLACEMENTS.items():
        path = args.root / relative
        text = path.read_text(encoding="utf-8-sig")
        original = text
        for old, new in replacements:
            old_count = text.count(old)
            new_count = text.count(new)
            if old_count == 1 and new_count == 0:
                text = text.replace(old, new, 1)
            elif old_count == 0 and new_count == 1:
                continue
            else:
                raise RuntimeError(
                    f"Unsafe replacement state in {path}: "
                    f"old_count={old_count}, new_count={new_count}"
                )
        if text != original:
            atomic_write(path, text)
            changed.append(relative)
        else:
            already_current.append(relative)
    print(
        {
            "status": "ok",
            "changed": changed,
            "already_current": already_current,
        }
    )


if __name__ == "__main__":
    main()
