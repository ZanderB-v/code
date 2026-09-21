#!/usr/bin/env python3
"""Freeze Target-ID v1 and Synthetic-v1 with auditable manifests.

The script is read-only with respect to the datasets. It writes manifests,
checksums, split-integrity checks, and snapshots of small experiment
definition files into 00_docs/dataset_freeze_v1.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator


LANGS = ("zh", "ug", "kk")
SPLITS = ("train", "dev", "test")
EXPECTED_TARGET = {
    "train": {"zh": 23954, "ug": 21899, "kk": 21001},
    "dev": {"zh": 346, "ug": 295, "kk": 310},
    "test": {"zh": 1142, "ug": 931, "kk": 993},
}
EXPECTED_SYNTHETIC_PER_LANG = 50000


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
        "--target-dir",
        type=Path,
        default=None,
        help="Defaults to 01_data_preparation/real_line_dataset_eval_reviewed.",
    )
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=None,
        help="Defaults to 03_synthetic_generation/synthetic_formal_v1.",
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        default=[
            "shard_0000",
            "shard_0001",
            "shard_0002",
            "shard_0003",
            "shard_0004",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to 00_docs/dataset_freeze_v1.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=2000)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip image SHA-256. Full freeze should omit this option.",
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}") from exc


def normalized_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "")
    return " ".join(text.replace("\u00a0", " ").split())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_image(base_dir: Path, image_value: str) -> Path:
    image = Path((image_value or "").replace("\\", "/"))
    if not image_value:
        raise ValueError("Empty image path")
    if image.is_absolute():
        return image
    return base_dir / image


def batched(iterable: Iterable[dict], size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def enrich_image_record(task: dict, hash_images: bool) -> dict:
    path = Path(task.pop("_absolute_image"))
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    task["image_size_bytes"] = stat.st_size
    task["image_sha256"] = sha256_file(path) if hash_images else None
    return task


def write_manifest(
    tasks: Iterable[dict],
    output_path: Path,
    workers: int,
    progress_every: int,
    hash_images: bool,
) -> dict:
    counts = Counter()
    language_counts = Counter()
    split_counts = Counter()
    difficulty_counts = Counter()
    background_counts = Counter()
    image_hash_counts = Counter()
    text_hash_counts = Counter()
    aggregate = hashlib.sha256()
    rows = 0

    with gzip.open(output_path, "wt", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for batch in batched(tasks, 512):
                for record in executor.map(
                    lambda item: enrich_image_record(item, hash_images),
                    batch,
                ):
                    rows += 1
                    counts[record["dataset"]] += 1
                    language_counts[record["language"]] += 1
                    if record.get("split"):
                        split_counts[record["split"]] += 1
                    if record.get("difficulty"):
                        difficulty_counts[record["difficulty"]] += 1
                    if record.get("background_type"):
                        background_counts[record["background_type"]] += 1
                    if record.get("image_sha256"):
                        image_hash_counts[record["image_sha256"]] += 1
                    text_hash_counts[record["logical_text_sha256"]] += 1

                    canonical = json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    output.write(canonical + "\n")
                    aggregate.update(canonical.encode("utf-8"))
                    aggregate.update(b"\n")

                    if progress_every > 0 and rows % progress_every == 0:
                        print(
                            json.dumps(
                                {
                                    "manifest": output_path.name,
                                    "rows": rows,
                                    "hash_images": hash_images,
                                }
                            ),
                            flush=True,
                        )

    duplicate_image_groups = sum(
        1 for count in image_hash_counts.values() if count > 1
    )
    duplicate_image_rows = sum(
        count for count in image_hash_counts.values() if count > 1
    )
    duplicate_text_groups = sum(
        1 for count in text_hash_counts.values() if count > 1
    )
    return {
        "rows": rows,
        "dataset_counts": dict(counts),
        "language_counts": dict(language_counts),
        "split_counts": dict(split_counts),
        "difficulty_counts": dict(difficulty_counts),
        "background_type_counts": dict(background_counts),
        "duplicate_image_hash_groups": duplicate_image_groups,
        "duplicate_image_hash_rows": duplicate_image_rows,
        "duplicate_text_hash_groups": duplicate_text_groups,
        "manifest_content_sha256": aggregate.hexdigest(),
        "manifest_file_sha256": sha256_file(output_path),
        "image_hashes_included": hash_images,
    }


def target_tasks(
    metadata_path: Path,
    target_dir: Path,
    root: Path,
    profile: dict,
) -> Iterator[dict]:
    counts = defaultdict(Counter)
    source_ids = defaultdict(set)
    text_hashes = defaultdict(lambda: defaultdict(set))

    for row in read_jsonl(metadata_path):
        split = row.get("split")
        language = row.get("language")
        if split not in SPLITS or language not in LANGS:
            raise ValueError(
                f"Unexpected target split/language: {split}/{language}"
            )
        text = normalized_text(
            row.get("logical_text") or row.get("text") or ""
        )
        if not text:
            raise ValueError(f"Empty target text: {row.get('id')}")
        image_path = resolve_image(target_dir, row.get("image") or "")
        counts[split][language] += 1
        source_ids[split].add(row.get("source_id"))
        text_hash = sha256_text(text)
        text_hashes[language][split].add(text_hash)

        yield {
            "dataset": "target_id_v1",
            "sample_key": str(row.get("id") or row.get("candidate_id")),
            "language": language,
            "split": split,
            "source_id": row.get("source_id"),
            "image": relative_to_root(image_path, root),
            "logical_text_sha256": text_hash,
            "text_length": len(text),
            "review_status": row.get("review_status"),
            "_absolute_image": str(image_path),
        }

    profile["counts"] = {
        split: dict(counts[split]) for split in SPLITS
    }
    profile["source_counts"] = {
        split: len(source_ids[split]) for split in SPLITS
    }
    profile["source_overlap"] = {
        "train_dev": len(source_ids["train"] & source_ids["dev"]),
        "train_test": len(source_ids["train"] & source_ids["test"]),
        "dev_test": len(source_ids["dev"] & source_ids["test"]),
    }
    profile["normalized_text_overlap"] = {}
    for language in LANGS:
        profile["normalized_text_overlap"][language] = {}
        for split in ("dev", "test"):
            overlap = (
                text_hashes[language]["train"]
                & text_hashes[language][split]
            )
            denominator = len(text_hashes[language][split])
            profile["normalized_text_overlap"][language][split] = {
                "unique_overlap": len(overlap),
                "unique_eval_texts": denominator,
                "ratio": len(overlap) / max(1, denominator),
            }


def synthetic_tasks(
    synthetic_root: Path,
    shards: list[str],
    root: Path,
    profile: dict,
) -> Iterator[dict]:
    counts = defaultdict(Counter)
    difficulty = Counter()
    backgrounds = Counter()
    fonts = Counter()

    for shard in shards:
        shard_dir = synthetic_root / f"synthetic_{shard}_parallel"
        metadata_path = shard_dir / "metadata.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)

        for row in read_jsonl(metadata_path):
            language = row.get("language")
            if language not in LANGS:
                raise ValueError(
                    f"Unexpected synthetic language: {language}"
                )
            text = normalized_text(
                row.get("logical_text") or row.get("text") or ""
            )
            ctc_text = normalized_text(row.get("ctc_text") or text)
            if not text or not ctc_text:
                raise ValueError(
                    f"Empty synthetic text: {shard}/{row.get('id')}"
                )
            image_path = resolve_image(shard_dir, row.get("image") or "")
            counts[shard][language] += 1
            difficulty[row.get("difficulty") or "unknown"] += 1
            backgrounds[row.get("background_type") or "unknown"] += 1
            fonts[row.get("font_name") or row.get("font") or "unknown"] += 1

            yield {
                "dataset": "synthetic_v1",
                "sample_key": f"{shard}:{row.get('id')}",
                "shard": shard,
                "language": language,
                "split": "train",
                "source_pool_id": row.get("source_pool_id"),
                "source": row.get("source"),
                "image": relative_to_root(image_path, root),
                "logical_text_sha256": sha256_text(text),
                "ctc_text_sha256": sha256_text(ctc_text),
                "text_length": len(text),
                "font": row.get("font_name") or row.get("font"),
                "background_id": row.get("background_id"),
                "background_type": row.get("background_type"),
                "difficulty": row.get("difficulty"),
                "contrast_bucket": row.get("contrast_bucket"),
                "seed": row.get("seed"),
                "_absolute_image": str(image_path),
            }

    profile["counts_by_shard"] = {
        shard: dict(counts[shard]) for shard in shards
    }
    total = Counter()
    for shard_counts in counts.values():
        total.update(shard_counts)
    profile["counts"] = dict(total)
    profile["difficulty_counts"] = dict(difficulty)
    profile["background_type_counts"] = dict(backgrounds)
    profile["font_top30"] = fonts.most_common(30)


def collect_definition_files(root: Path, shards: list[str]) -> list[Path]:
    direct = [
        root / "scripts/synth/06_generate_samples.py",
        root / "scripts/synth/07_postprocess_images.py",
        root / "scripts/synth/08_validate_dataset.py",
        root / "scripts/synth/10_dataset_statistics.py",
        root / "scripts/synth/run_formal_parallel_shards_tmux.sh",
        root / "scripts/svtrv2/prepare_d2_synth50k.py",
        root / "scripts/svtrv2/prepare_e0_target_only.py",
        root / "scripts/svtrv2/prepare_e1_target_only.py",
        root / "scripts/svtrv2/prepare_e5_d2_to_target.py",
        root / "04_model_training/configs/svtrv2_s_d2_synth50k.yml",
        root / "04_model_training/configs/svtrv2_s_e0_random_target_only.yml",
        root / "04_model_training/configs/svtrv2_s_e1_target_only.yml",
        root / "04_model_training/configs/svtrv2_s_e5_d2_to_target.yml",
        root
        / "04_model_training/character_dict_hz_ug_kk_v1/character_dict.txt",
        root
        / "01_data_preparation/real_line_dataset_eval_reviewed/metadata.jsonl",
        root
        / "01_data_preparation/real_line_dataset_eval_reviewed/summary.json",
    ]
    for experiment in (
        "d2_synth50k",
        "e0_random_target_only",
        "e1_target_only",
        "e5_d2_to_target",
    ):
        direct.append(
            root
            / "04_model_training"
            / "datasets"
            / experiment
            / "prepare_summary.json"
        )
    synthetic_root = root / "03_synthetic_generation/synthetic_formal_v1"
    for shard in shards:
        shard_dir = synthetic_root / f"synthetic_{shard}_parallel"
        direct.extend(
            [
                shard_dir / "metadata.jsonl",
                shard_dir / "merged_summary.json",
                shard_dir / "statistics/statistics_summary.json",
            ]
        )
    return sorted({path for path in direct if path.is_file()})


def snapshot_definitions(
    root: Path,
    output_dir: Path,
    shards: list[str],
) -> dict:
    snapshot_root = output_dir / "definition_snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    checksum_lines = []
    copied = []
    hashed = []

    for path in collect_definition_files(root, shards):
        relative = relative_to_root(path, root)
        digest = sha256_file(path)
        size = path.stat().st_size
        checksum_lines.append(f"{digest}  {relative}")
        hashed.append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": digest,
            }
        )
        if size <= 5 * 1024 * 1024:
            destination = snapshot_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied.append(relative)

    checksum_path = output_dir / "definition_files.sha256"
    checksum_path.write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
    return {
        "files_hashed": len(hashed),
        "small_files_snapshotted": len(copied),
        "checksums": hashed,
        "checksum_file": str(checksum_path),
    }


def validate_counts(target_profile: dict, synthetic_profile: dict) -> None:
    if target_profile.get("counts") != EXPECTED_TARGET:
        raise RuntimeError(
            "Target-ID counts changed. "
            f"Expected {EXPECTED_TARGET}, got {target_profile.get('counts')}"
        )
    if any(target_profile["source_overlap"].values()):
        raise RuntimeError(
            "Target-ID source leakage detected: "
            f"{target_profile['source_overlap']}"
        )
    expected_synthetic = {
        lang: EXPECTED_SYNTHETIC_PER_LANG for lang in LANGS
    }
    if synthetic_profile.get("counts") != expected_synthetic:
        raise RuntimeError(
            "Synthetic-v1 counts changed. "
            f"Expected {expected_synthetic}, "
            f"got {synthetic_profile.get('counts')}"
        )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    target_dir = (
        args.target_dir.resolve()
        if args.target_dir
        else root
        / "01_data_preparation"
        / "real_line_dataset_eval_reviewed"
    )
    synthetic_root = (
        args.synthetic_root.resolve()
        if args.synthetic_root
        else root / "03_synthetic_generation" / "synthetic_formal_v1"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "00_docs" / "dataset_freeze_v1"
    )

    if output_dir.exists():
        if not args.replace:
            raise SystemExit(
                f"Output already exists: {output_dir}. Use --replace."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hash_images = not args.metadata_only
    target_profile: dict = {}
    synthetic_profile: dict = {}

    print("[1/4] Freezing Target-ID v1", flush=True)
    target_manifest = output_dir / "target_id_v1_manifest.jsonl.gz"
    target_manifest_summary = write_manifest(
        target_tasks(
            target_dir / "metadata.jsonl",
            target_dir,
            root,
            target_profile,
        ),
        target_manifest,
        args.workers,
        args.progress_every,
        hash_images,
    )

    print("[2/4] Freezing Synthetic-v1", flush=True)
    synthetic_manifest = output_dir / "synthetic_v1_manifest.jsonl.gz"
    synthetic_manifest_summary = write_manifest(
        synthetic_tasks(
            synthetic_root,
            args.shards,
            root,
            synthetic_profile,
        ),
        synthetic_manifest,
        args.workers,
        args.progress_every,
        hash_images,
    )

    print("[3/4] Validating frozen counts and source isolation", flush=True)
    validate_counts(target_profile, synthetic_profile)

    print("[4/4] Snapshotting experiment definitions", flush=True)
    definitions = snapshot_definitions(root, output_dir, args.shards)

    summary = {
        "freeze_version": "dataset_freeze_v1",
        "root": str(root),
        "mode": "full_image_sha256" if hash_images else "metadata_only",
        "datasets_are_modified": False,
        "target_id_v1": {
            "directory": str(target_dir),
            "profile": target_profile,
            "manifest": str(target_manifest),
            "manifest_summary": target_manifest_summary,
        },
        "synthetic_v1": {
            "directory": str(synthetic_root),
            "shards": args.shards,
            "profile": synthetic_profile,
            "manifest": str(synthetic_manifest),
            "manifest_summary": synthetic_manifest_summary,
        },
        "experiment_definitions": definitions,
        "frozen_experiment_ids": [
            "E0_random_initialization_target_domain_only",
            "E1_target_domain_only",
            "D2_synthetic_50k_per_language",
            "E5_D2_to_target_finetuning",
        ],
        "policy": {
            "target_split_rule": "source_id_disjoint",
            "primary_test_data": "Target-ID v1 only",
            "synthetic_v1_role": "training resource, not primary test",
            "future_robust_data": (
                "Create a new version and new experiment IDs; do not overwrite "
                "Target-ID v1, Synthetic-v1, or E0/E1/D2/E5 definitions."
            ),
        },
    }
    summary_path = output_dir / "freeze_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Frozen summary: {summary_path}")


if __name__ == "__main__":
    main()
