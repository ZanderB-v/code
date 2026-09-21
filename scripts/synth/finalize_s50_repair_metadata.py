#!/usr/bin/env python3
"""Keep only validated repair renders and prove that every language has reserve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LANGUAGES = ("zh", "ug", "kk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    repair_root = args.repair_root.resolve()
    plan = read_json(repair_root / "repair_plan.json")
    summary: dict[str, Any] = {
        "status": "S50_REPAIR_METADATA_READY",
        "repair_root": str(repair_root),
        "languages": {},
    }

    for language in LANGUAGES:
        part = repair_root / "repair_parts" / f"synthetic_repair_{language}"
        metadata_path = part / "metadata.jsonl"
        postprocess_failures_path = part / "postprocess_failures.jsonl"
        validation_failures_path = part / "validation_failures.jsonl"
        rows = read_jsonl(metadata_path)
        postprocess_failures = (
            read_jsonl(postprocess_failures_path)
            if postprocess_failures_path.is_file()
            else []
        )
        validation_failures = (
            read_jsonl(validation_failures_path)
            if validation_failures_path.is_file()
            else []
        )
        invalid_ids = {
            str(row.get("id") or "")
            for row in (*postprocess_failures, *validation_failures)
            if row.get("id")
        }
        valid_rows = [
            row
            for row in rows
            if str(row.get("id") or "") not in invalid_ids
            and (part / str(row.get("image") or "")).is_file()
        ]
        needed = int(
            plan["languages"][language]["leaked_rows_to_replace"]
        )
        if len(valid_rows) < needed:
            raise RuntimeError(
                f"Insufficient validated {language} repair rows: "
                f"need={needed}, valid={len(valid_rows)}"
            )

        output = part / "metadata_repair_valid.jsonl"
        write_jsonl(output, valid_rows)
        summary["languages"][language] = {
            "needed": needed,
            "postprocessed_rows": len(rows),
            "postprocess_failures": len(postprocess_failures),
            "validation_failures": len(validation_failures),
            "validated_rows": len(valid_rows),
            "reserve_after_validation": len(valid_rows) - needed,
            "metadata": str(output),
        }

    output_summary = repair_root / "repair_validation_summary.json"
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
