#!/usr/bin/env python3
"""Regression checks for the closed M3 alpha verification grid."""

import argparse
import copy
import json
from pathlib import Path

import yaml

from dual_order_protocol import make_method_config
from run_m3_alpha_verification import normalized_config, select_alpha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    configs = {}
    for method in ("m3", "m3_alpha020", "m3_alpha025", "m3_alpha030"):
        text = make_method_config(
            method=method,
            root=root,
            run_dir=root / "unused_run",
            project_name=method,
            train_lmdbs=[root / "unused_train"],
            eval_lmdbs=[root / "unused_dev"],
            pretrained_model=root / "unused.pth",
            max_epoch=50,
            first_batch_size=16,
            num_workers=4,
            max_ratio=40,
            lr=2.5e-5,
            internal_eval_every=100000,
            seed=20260731,
        )
        path = root / "04_model_training/configs" / f".test_{method}.yml"
        path.write_text(text, encoding="utf-8")
        try:
            configs[method] = normalized_config(path)
        finally:
            path.unlink()
    assert len({json.dumps(value, sort_keys=True) for value in configs.values()}) == 1

    rows = [
        {"alpha": .15, "macro_cer": .0230},
        {"alpha": .20, "macro_cer": .0229},
        {"alpha": .25, "macro_cer": .0229},
        {"alpha": .30, "macro_cer": .0233},
    ]
    assert select_alpha(copy.deepcopy(rows))["alpha"] == .20
    try:
        select_alpha(rows[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("An incomplete alpha grid was accepted")
    print(json.dumps({
        "status": "M3_ALPHA_VERIFICATION_COMPONENTS_OK",
        "candidate_alphas": [0.15, 0.20, 0.25, 0.30],
        "new_training_alphas": [0.20, 0.25],
        "config_equivalence": True,
        "primary_metric_selection": True,
        "test_evaluated": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
