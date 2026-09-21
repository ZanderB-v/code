#!/usr/bin/env python3
"""Prepare deterministic, evaluation-safe text pools for S50 repair renders."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LANGUAGES = ("zh", "ug", "kk")
DEFAULT_SEED = 20260729


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reserve-per-language", type=int, default=64)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.replace("\u00a0", " ").split())


def near_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", normalize_text(text)).casefold()
    return "".join(
        char
        for char in text
        if not unicodedata.category(char).startswith(("P", "Z"))
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    formal_root = (
        root / "03_synthetic_generation" / "synthetic_formal_v2"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else formal_root / "leakage_repair_v1"
    )
    if output_dir.exists() and not args.replace:
        raise SystemExit(f"Output exists; use --replace: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_out = output_dir / "text_pool"
    pool_out.mkdir(parents=True, exist_ok=True)

    target_path = (
        root
        / "01_data_preparation"
        / "real_line_dataset_eval_reviewed"
        / "metadata.jsonl"
    )
    s50_path = formal_root / "subsets" / "s50" / "metadata.jsonl"
    source_pool = root / "02_corpus_preparation" / "mixed_text_pool_v2"

    forbidden: dict[str, set[str]] = defaultdict(set)
    forbidden_near: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(target_path):
        if row.get("split") not in ("dev", "test"):
            continue
        language = str(row.get("language") or "")
        logical = normalize_text(
            str(row.get("logical_text") or row.get("text") or "")
        )
        forbidden[language].add(logical)
        forbidden_near[language].add(near_key(logical))

    used: dict[str, set[str]] = defaultdict(set)
    removed = Counter()
    removed_ids: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(s50_path):
        language = str(row.get("language") or "")
        logical = normalize_text(str(row.get("logical_text") or ""))
        used[language].add(logical)
        if (
            logical in forbidden[language]
            or near_key(logical) in forbidden_near[language]
        ):
            removed[language] += 1
            removed_ids[language].append(str(row.get("id") or ""))

    selected_summary = {}
    for language in LANGUAGES:
        source_path = (
            source_pool / f"{language}_mixed_text_pool_v2.jsonl"
        )
        max_length = 30 if language == "zh" else 45
        candidates = []
        seen = set()
        for row in read_jsonl(source_path):
            logical = normalize_text(
                str(row.get("normalized_text") or row.get("text") or "")
            )
            compact_length = len(logical.replace(" ", ""))
            if (
                not logical
                or compact_length > max_length
                or logical in forbidden[language]
                or near_key(logical) in forbidden_near[language]
                or logical in used[language]
                or logical in seen
            ):
                continue
            seen.add(logical)
            item = dict(row)
            item["language"] = language
            item["normalized_text"] = logical
            item["repair_selection"] = "exclude_target_dev_test_and_s50"
            candidates.append(item)

        language_seed = int.from_bytes(
            hashlib.sha256(
                f"{args.seed}:{language}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        random.Random(language_seed).shuffle(candidates)
        needed = removed[language]
        selected_count = needed + args.reserve_per_language
        if len(candidates) < selected_count:
            raise RuntimeError(
                f"Not enough clean {language} replacements: "
                f"need {selected_count}, found {len(candidates)}"
            )
        selected = candidates[:selected_count]
        output_path = pool_out / f"{language}_mixed_text_pool_v2.jsonl"
        write_jsonl(output_path, selected)
        selected_summary[language] = {
            "leaked_rows_to_replace": needed,
            "reserved_candidates": args.reserve_per_language,
            "pool_rows": len(selected),
            "pool_path": str(output_path),
            "pool_sha256": sha256(output_path),
            "removed_ids": removed_ids[language],
        }

    result = {
        "status": "S50_LEAKAGE_REPAIR_PLAN_READY",
        "seed": args.seed,
        "target_metadata": str(target_path),
        "s50_metadata": str(s50_path),
        "source_pool": str(source_pool),
        "output_dir": str(output_dir),
        "languages": selected_summary,
        "render_counts": {
            language: removed[language] for language in LANGUAGES
        },
        "total_render_count": sum(removed.values()),
    }
    plan_path = output_dir / "repair_plan.json"
    plan_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
