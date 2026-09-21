#!/usr/bin/env python3
"""Evaluate a selected public baseline on Clean and frozen Corrupted Dev only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from formal_baseline_data import (
    PROJECT_ROOT,
    formal_implementation_hashes,
    load_corrupted_dev_records,
    load_protocol,
    load_records,
    read_dictionary,
)
from formal_baseline_runtime import (
    LineDataset,
    collate_lines,
    evaluate_clean_dev,
)
from pretrained_models import build_model_bundle, sha256_file
from train_pretrained_baseline import BATCH_POLICY, load_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(BATCH_POLICY))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if "test" in str(args.output_dir).lower() or "test" in str(args.checkpoint).lower():
        raise ValueError("Test paths are forbidden during baseline development")
    protocol = load_protocol(args.protocol)
    protocol_path = args.protocol or (
        PROJECT_ROOT / "Comparison/pretrained_baselines_v1/protocol.json"
    )
    protocol_sha256 = sha256_file(protocol_path)
    implementation_sha256 = formal_implementation_hashes()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")
    characters = read_dictionary(
        PROJECT_ROOT / protocol["dictionary"],
        bool(protocol["use_space_char"]),
    )
    if len(characters) != int(protocol["expected_recognition_symbol_count"]):
        raise ValueError("Runtime recognition-symbol count mismatch")
    char_to_id = {character: index + 1 for index, character in enumerate(characters)}
    public_checkpoint = PROJECT_ROOT / protocol["models"][args.model]["checkpoint"]
    bundle = build_model_bundle(
        args.model,
        public_checkpoint,
        len(characters),
        int(protocol["max_label_length"]),
    )
    state, payload = load_state(args.checkpoint)
    if payload.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Selected checkpoint belongs to a stale comparison protocol")
    if payload.get("implementation_sha256") != implementation_sha256:
        raise ValueError("Selected checkpoint belongs to stale comparison code")
    bundle.model.load_state_dict(state, strict=True)
    bundle.model.to(device).eval()
    policy = BATCH_POLICY[args.model]

    def loader(records):
        return DataLoader(
            LineDataset(records, bundle.input_size, char_to_id),
            batch_size=policy["eval"],
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_lines,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean = evaluate_clean_dev(
        args.model,
        bundle.model,
        loader(load_records(protocol, "target_dev")),
        characters,
        device,
        args.output_dir / "clean_dev",
    )
    corrupted = evaluate_clean_dev(
        args.model,
        bundle.model,
        loader(load_corrupted_dev_records(protocol)),
        characters,
        device,
        args.output_dir / "corrupted_dev",
    )
    clean_cer = float(clean["clean_dev_macro_cer"])
    corrupted_cer = float(corrupted["mean_corrupted_macro_cer"])
    summary = {
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "clean_dev_macro_cer": clean_cer,
        "mean_corrupted_macro_cer": corrupted_cer,
        "absolute_cer_increase": corrupted_cer - clean_cer,
        "relative_cer_increase": (
            (corrupted_cer - clean_cer) / clean_cer if clean_cer else None
        ),
        "conditions": corrupted["conditions"],
        "languages_clean": clean["languages"],
        "languages_corrupted_micro": corrupted["languages"],
        "selection_policy": "checkpoint selected on Clean Dev only",
        "corrupted_dev_role": "post-selection diagnostic only",
        "test_evaluated": False,
    }
    (args.output_dir / "final_dev_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("PRETRAINED_BASELINE_CLEAN_CORRUPTED_DEV_OK")


if __name__ == "__main__":
    main()
