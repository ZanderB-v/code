#!/usr/bin/env python3
"""Regression checks for the frozen Clean Dev V2 evaluator contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_evaluator(root: Path):
    path = root / "scripts" / "svtrv2" / "evaluate_d2_checkpoint.py"
    spec = importlib.util.spec_from_file_location("evaluate_d2_checkpoint", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    protocol = root / "01_data_preparation" / "clean_dev_v2_verified"
    evaluator = load_evaluator(root)
    args = argparse.Namespace(
        root=root,
        metric_normalization="normalization_v2",
        frozen_clean_dev_manifest=protocol / "frozen_clean_dev_v2_manifest.json",
        split="dev",
        label_template="{lang}.txt",
        label_prefix="unused",
    )
    verified = evaluator.verify_frozen_clean_dev_v2(args, protocol / "labels")
    assert verified["protocol_id"] == "CLEAN_DEV_V2_VERIFIED"
    assert verified["rows"] == 951
    assert verified["patch_rows"] == 15

    missing_manifest = argparse.Namespace(**vars(args))
    missing_manifest.frozen_clean_dev_manifest = None
    try:
        evaluator.verify_frozen_clean_dev_v2(missing_manifest, protocol / "labels")
    except ValueError as error:
        assert "requires --frozen-clean-dev-manifest" in str(error)
    else:
        raise AssertionError("V2 evaluation must reject a missing frozen manifest")

    wrong_template = argparse.Namespace(**vars(args))
    wrong_template.label_template = "{prefix}_{split}_{lang}_logical.txt"
    try:
        evaluator.verify_frozen_clean_dev_v2(wrong_template, protocol / "labels")
    except ValueError as error:
        assert "requires --label-template" in str(error)
    else:
        raise AssertionError("V2 evaluation must reject legacy label naming")

    print(
        json.dumps(
            {
                "status": "CLEAN_DEV_V2_EVALUATOR_CONTRACT_OK",
                "rows": verified["rows"],
                "patch_rows": verified["patch_rows"],
                "test_evaluated": False,
                "errors": [],
            }
        )
    )


if __name__ == "__main__":
    main()
