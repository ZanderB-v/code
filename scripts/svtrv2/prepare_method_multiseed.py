#!/usr/bin/env python3
"""Clone frozen target-finetune configs into isolated multiseed runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


METHODS = ("b1", "m1", "m2", "m3")
CONFIG_KIND = {"b1": "full_svtrv2", "m1": "dual_order", "m2": "dual_order", "m3": "dual_order"}
PREDICTION_BRANCH = {method: "ctc" for method in METHODS}
CANONICAL_SEED = 20260731
DEFAULT_NEW_SEEDS = (20260811, 20260812)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_NEW_SEEDS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, relative: str, expected: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch {relative}: expected={expected}, actual={actual}")
    return path


def replace_global_value(text: str, key: str, value: str | int) -> str:
    lines = text.splitlines()
    inside = False
    matched = 0
    rendered = str(value) if isinstance(value, int) else json.dumps(value)
    for index, line in enumerate(lines):
        if line == "Global:":
            inside = True
            continue
        if inside and line and not line.startswith(" "):
            inside = False
        if inside and re.match(rf"^  {re.escape(key)}:\s*", line):
            lines[index] = f"  {key}: {rendered}"
            matched += 1
    if matched != 1:
        raise ValueError(f"Expected one Global.{key}, found {matched}")
    return "\n".join(lines) + "\n"


def path_suffix(path: str) -> str:
    normalized = path.replace("\\", "/")
    marker = "/04_model_training/"
    if marker not in normalized:
        return normalized
    return "04_model_training/" + normalized.split(marker, 1)[1]


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    seeds = list(dict.fromkeys(args.seeds))
    if CANONICAL_SEED in seeds:
        raise ValueError("The canonical seed is already frozen and must not be retrained")
    if len(seeds) < 2:
        raise ValueError("At least two new target-finetune seeds are required")

    freeze_root = root / "00_docs" / "frozen_method_design_v1"
    verification = read_json(freeze_root / "verification_report.json")
    if verification.get("status") != "passed" or verification.get("errors") != []:
        raise ValueError("Method Design Freeze V1 verification did not pass")
    manifest = read_json(freeze_root / "method_design_v1_manifest.json")
    if float(manifest["official_m2_consistency_weight"]) != 0.30:
        raise ValueError("Official M2 must be alpha=0.30")

    config_root = root / "04_model_training" / "configs" / "multiseed"
    run_root = root / "04_model_training" / "runs" / "multiseed"
    config_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    methods: dict[str, Any] = {}
    for method in METHODS:
        entry = manifest["methods"][method]
        target = entry["target"]
        synthetic = entry["synthetic"]
        base_config = verify(root, target["config"], target["config_sha256"])
        synthetic_checkpoint = verify(
            root,
            synthetic["checkpoint"],
            synthetic["checkpoint_sha256"],
        )
        verify(root, target["summary"], target["summary_sha256"])
        base_text = base_config.read_text(encoding="utf-8-sig")
        parsed = yaml.safe_load(base_text)
        if int(parsed["Global"].get("seed", -1)) != CANONICAL_SEED:
            raise ValueError(
                f"{method} canonical target config seed drift: "
                f"expected={CANONICAL_SEED}, actual={parsed['Global'].get('seed')}"
            )
        if parsed["Global"].get("preprocess_protocol") != "P1_MSR_V3":
            raise ValueError(f"{method} does not use frozen P1_MSR_V3 preprocessing")
        pretrained = str(parsed["Global"].get("pretrained_model", ""))
        if path_suffix(pretrained) != synthetic["checkpoint"]:
            raise ValueError(
                f"{method} base target config does not use its frozen synthetic checkpoint: "
                f"{pretrained}"
            )

        new_runs = []
        for seed in seeds:
            model_name = f"{target['model']}_seed_{seed}"
            run_dir = run_root / method / f"seed_{seed}"
            config_path = config_root / f"{model_name}.yml"
            text = base_text
            text = replace_global_value(text, "output_dir", str(run_dir))
            text = replace_global_value(text, "save_res_path", str(run_dir / "predicts.txt"))
            text = replace_global_value(text, "project_name", model_name)
            text = replace_global_value(text, "seed", seed)
            config_path.write_text(text, encoding="utf-8")
            check = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            expected = copy.deepcopy(parsed)
            expected["Global"]["output_dir"] = str(run_dir)
            expected["Global"]["save_res_path"] = str(run_dir / "predicts.txt")
            expected["Global"]["project_name"] = model_name
            expected["Global"]["seed"] = seed
            if check != expected:
                raise ValueError(
                    f"Generated config changed fields outside the approved "
                    f"Global output/seed allowlist: {config_path}"
                )
            if int(check["Global"]["seed"]) != seed:
                raise ValueError(f"Seed write failed: {config_path}")
            if path_suffix(str(check["Global"]["pretrained_model"])) != synthetic["checkpoint"]:
                raise ValueError(f"Pretrained checkpoint drift: {config_path}")
            new_runs.append(
                {
                    "seed": seed,
                    "config": str(config_path),
                    "config_sha256": sha256(config_path),
                    "run_dir": str(run_dir),
                    "eval_prefix": f"multiseed_{method}_seed_{seed}",
                    "final_summary": str(
                        root
                        / "04_model_training"
                        / "eval_reports"
                        / f"multiseed_{method}_seed_{seed}_final_summary.json"
                    ),
                }
            )
        methods[method] = {
            "config_kind": CONFIG_KIND[method],
            "prediction_branch": PREDICTION_BRANCH[method],
            "fixed_synthetic_checkpoint": str(synthetic_checkpoint),
            "fixed_synthetic_checkpoint_sha256": synthetic["checkpoint_sha256"],
            "canonical": {
                "seed": CANONICAL_SEED,
                "config": str(base_config),
                "config_sha256": target["config_sha256"],
                "final_summary": str(root / target["summary"]),
                "checkpoint": str(root / target["checkpoint"]),
                "checkpoint_sha256": target["checkpoint_sha256"],
            },
            "new_runs": new_runs,
        }

    plan = {
        "status": "passed",
        "protocol_id": "METHOD_TARGET_FINETUNE_MULTISEED_V1",
        "training_scope": "fixed_synthetic_pretrain_checkpoint_target_finetune_only",
        "canonical_seed": CANONICAL_SEED,
        "new_seeds": seeds,
        "selection_metric": "Clean Dev Macro CER",
        "test_policy": "not_evaluated",
        "config_change_allowlist": [
            "Global.output_dir",
            "Global.save_res_path",
            "Global.project_name",
            "Global.seed",
        ],
        "randomness_scope": {
            "model_initialization": "seeded",
            "augmentation_and_worker_rng": "seeded",
            "lmdb_traversal": "seeded_before_dataset_construction",
            "epoch_batch_shuffle": "deterministic_by_epoch_for_resume_equivalence",
        },
        "methods": methods,
    }
    output = (
        args.output.resolve()
        if args.output
        else root / "04_model_training" / "multiseed" / "method_target_seed_plan_v1.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "METHOD_MULTISEED_CONFIGS_READY", "plan": str(output), "new_runs": sum(len(v["new_runs"]) for v in methods.values())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
