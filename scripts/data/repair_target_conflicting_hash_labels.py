#!/usr/bin/env python3
"""Remove confirmed same-image/different-label rows from the target benchmark.

The conflicting crop files are retained for provenance, but excluded from every
canonical metadata and label index. The operation is deterministic and
idempotent. It does not alter any image bytes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


CONFLICT_GROUPS = (
    {
        "split": "dev",
        "ids": ("misogyny_399_ug_L003", "misogyny_399_ug_L004"),
    },
    {
        "split": "dev",
        "ids": ("misogyny_495_ug_L001", "misogyny_495_ug_L002"),
    },
    {
        "split": "dev",
        "ids": ("misogyny_843_ug_L001", "misogyny_843_ug_L002"),
    },
    {
        "split": "test",
        "ids": ("meme_5256_ug_L003", "meme_5256_ug_L004"),
    },
    {
        "split": "test",
        "ids": ("meme_792_ug_L001", "meme_792_ug_L002"),
    },
)

ORIGINAL_COUNTS = {
    "train": {"zh": 23954, "ug": 21899, "kk": 21001},
    "dev": {"zh": 346, "ug": 301, "kk": 310},
    "test": {"zh": 1142, "ug": 935, "kk": 993},
}

REPAIRED_COUNTS = {
    "train": {"zh": 23954, "ug": 21899, "kk": 21001},
    "dev": {"zh": 346, "ug": 295, "kk": 310},
    "test": {"zh": 1142, "ug": 931, "kk": 993},
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
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Defaults to <root>/01_data_preparation/real_line_dataset_eval_reviewed.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite canonical indexes. Without this flag, perform a dry run.",
    )
    return parser.parse_args()


def row_id(row: dict) -> str:
    return str(row.get("id") or row.get("candidate_id") or "")


def logical_text(row: dict) -> str:
    return str(row.get("logical_text") or row.get("text") or "")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}") from exc
    return rows


def image_path(dataset_dir: Path, row: dict) -> Path:
    relative = str(row.get("image") or "").replace("\\", "/")
    if not relative:
        raise ValueError(f"Missing image path for {row_id(row)}")
    return dataset_dir / relative


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def counts_by_split_language(rows: list[dict]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "dev", "test"):
        counter = Counter(
            str(row.get("language") or "")
            for row in rows
            if row.get("split") == split
        )
        result[split] = {
            language: counter[language] for language in ("zh", "ug", "kk")
        }
    return result


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    temporary = path.with_name(path.name + ".repair_tmp")
    temporary.write_text(text, encoding=encoding)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False) + "\n"
        for row in rows
    )
    atomic_write_text(path, text)


def csv_fieldnames(path: Path, rows: list[dict]) -> list[str]:
    if path.is_file():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
        if header:
            return header
    return list(rows[0].keys()) if rows else []


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    temporary = path.with_name(path.name + ".repair_tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_labels(path: Path, rows: list[dict]) -> None:
    lines = []
    for row in rows:
        image = str(row.get("image") or "").replace("\\", "/")
        text = logical_text(row)
        if not image or not text:
            raise ValueError(f"Empty image/text for {row_id(row)}")
        lines.append(f"{image}\t{text}")
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def validate_groups(
    dataset_dir: Path,
    rows_by_id: dict[str, dict],
) -> list[dict]:
    audited_groups = []
    for group_index, group in enumerate(CONFLICT_GROUPS, 1):
        group_rows = [rows_by_id[item] for item in group["ids"]]
        splits = {row.get("split") for row in group_rows}
        languages = {row.get("language") for row in group_rows}
        source_ids = {row.get("source_id") for row in group_rows}
        labels = {logical_text(row) for row in group_rows}
        paths = [image_path(dataset_dir, row) for row in group_rows]
        missing_images = [str(path) for path in paths if not path.is_file()]
        if missing_images:
            raise FileNotFoundError(
                f"Missing conflict-group images: {missing_images}"
            )
        hashes = [sha256_file(path) for path in paths]
        if splits != {group["split"]}:
            raise ValueError(
                f"Unexpected split for group {group_index}: {splits}"
            )
        if languages != {"ug"}:
            raise ValueError(
                f"Unexpected language for group {group_index}: {languages}"
            )
        if len(source_ids) != 1:
            raise ValueError(
                f"Conflict group spans source images: {source_ids}"
            )
        if len(labels) != 2:
            raise ValueError(
                f"Conflict group does not have two labels: {group['ids']}"
            )
        if len(set(hashes)) != 1:
            raise ValueError(
                f"Conflict group images are no longer identical: {group['ids']}"
            )
        audited_groups.append(
            {
                "group": group_index,
                "split": group["split"],
                "language": "ug",
                "source_id": next(iter(source_ids)),
                "sha256": hashes[0],
                "rows": [
                    {
                        "id": row_id(row),
                        "image": str(row.get("image") or "").replace("\\", "/"),
                        "logical_text": logical_text(row),
                    }
                    for row in group_rows
                ],
                "action": "exclude_both_rows_keep_image_files",
                "reason": "exact_same_image_bytes_with_different_labels",
            }
        )
    return audited_groups


def update_summary(
    dataset_dir: Path,
    repaired_rows: list[dict],
    revision_relative: str,
) -> None:
    summary_path = dataset_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8-sig"))
        if summary_path.is_file()
        else {}
    )
    counts = counts_by_split_language(repaired_rows)
    summary["rows"] = {
        split: sum(language_counts.values())
        for split, language_counts in counts.items()
    }
    summary["by_language"] = counts
    summary["total"] = len(repaired_rows)
    summary["dataset_revision"] = {
        "id": "remove_conflicting_duplicate_labels_v1",
        "removed_rows": 10,
        "removed_groups": 5,
        "revision_manifest": revision_relative,
        "image_files_retained": True,
    }
    atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )


def apply_repair(
    dataset_dir: Path,
    original_rows: list[dict],
    repaired_rows: list[dict],
    revision: dict,
) -> None:
    root_csv = dataset_dir / "metadata.csv"
    root_fieldnames = csv_fieldnames(root_csv, original_rows)
    write_jsonl(dataset_dir / "metadata.jsonl", repaired_rows)
    write_csv(root_csv, repaired_rows, root_fieldnames)
    write_labels(dataset_dir / "labels.txt", repaired_rows)

    for split in ("dev", "test"):
        split_dir = dataset_dir / f"{split}_reviewed"
        split_rows = [
            row for row in repaired_rows if row.get("split") == split
        ]
        metadata_csv = split_dir / "metadata.csv"
        fieldnames = csv_fieldnames(metadata_csv, split_rows)
        write_jsonl(split_dir / "metadata.jsonl", split_rows)
        write_csv(metadata_csv, split_rows, fieldnames)
        write_csv(split_dir / f"{split}.csv", split_rows, fieldnames)
        write_labels(split_dir / "labels.txt", split_rows)

    revisions_dir = dataset_dir / "dataset_revisions"
    revisions_dir.mkdir(parents=True, exist_ok=True)
    revision_path = revisions_dir / "remove_conflicting_duplicate_labels_v1.json"
    atomic_write_text(
        revision_path,
        json.dumps(revision, ensure_ascii=False, indent=2) + "\n",
    )
    update_summary(
        dataset_dir,
        repaired_rows,
        revision_path.relative_to(dataset_dir).as_posix(),
    )


def main() -> None:
    args = parse_args()
    dataset_dir = (
        args.dataset_dir
        if args.dataset_dir is not None
        else args.root
        / "01_data_preparation"
        / "real_line_dataset_eval_reviewed"
    )
    metadata_path = dataset_dir / "metadata.jsonl"
    rows = read_jsonl(metadata_path)
    ids = [row_id(row) for row in rows]
    duplicate_ids = [
        item for item, count in Counter(ids).items() if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"Duplicate metadata IDs: {duplicate_ids[:20]}")

    excluded_ids = {
        item
        for group in CONFLICT_GROUPS
        for item in group["ids"]
    }
    present = excluded_ids.intersection(ids)
    original_counts = counts_by_split_language(rows)

    if not present:
        if original_counts != REPAIRED_COUNTS:
            raise ValueError(
                "Conflict IDs are absent, but repaired counts do not match: "
                f"{original_counts}"
            )
        print(
            json.dumps(
                {
                    "status": "already_repaired",
                    "dataset_dir": str(dataset_dir),
                    "rows": len(rows),
                    "counts": original_counts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if present != excluded_ids:
        missing = sorted(excluded_ids - present)
        raise ValueError(
            f"Partially repaired dataset; missing conflict IDs: {missing}"
        )
    if original_counts != ORIGINAL_COUNTS:
        raise ValueError(
            f"Unexpected original dataset counts: {original_counts}"
        )

    rows_by_id = {row_id(row): row for row in rows}
    audited_groups = validate_groups(dataset_dir, rows_by_id)
    repaired_rows = [
        row for row in rows if row_id(row) not in excluded_ids
    ]
    repaired_counts = counts_by_split_language(repaired_rows)
    if repaired_counts != REPAIRED_COUNTS:
        raise AssertionError(
            f"Unexpected repaired counts: {repaired_counts}"
        )

    revision = {
        "revision_id": "remove_conflicting_duplicate_labels_v1",
        "policy": (
            "Conservatively exclude both rows when identical image bytes have "
            "different labels; retain crop files for provenance."
        ),
        "dataset_dir": str(dataset_dir),
        "before": {
            "rows": len(rows),
            "counts": original_counts,
        },
        "after": {
            "rows": len(repaired_rows),
            "counts": repaired_counts,
        },
        "removed_ids": sorted(excluded_ids),
        "groups": audited_groups,
    }
    report = {
        "status": "ready_to_apply" if not args.apply else "repaired",
        **revision,
    }
    if args.apply:
        apply_repair(dataset_dir, rows, repaired_rows, revision)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
