#!/usr/bin/env python3
"""CPU protocol checks for the controlled M3+SLDR pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "scripts/svtrv2"))
    sys.path.insert(0, str(root / "third_party/OpenOCR"))

    import torch

    from dual_order_protocol import METHODS, make_method_config
    from openrec.modeling.decoders.dual_order_gtc_decoder import (
        ScriptAwareLocalDetailRefinement,
        ScriptConditionedResidualAdapter,
    )

    spec = METHODS["m3_sldr_pilot"]
    assert spec["consistency_weight"] == 0.15
    assert spec["script_weight"] == 0.10
    assert spec["direction_weight"] == 0.0
    assert spec["use_script_adapter"] is True
    assert spec["use_sldr"] is True

    generated = make_method_config(
        method="m3_sldr_pilot",
        root=root,
        run_dir=root / "tmp_sldr_run",
        project_name="m3_sldr_protocol_test",
        train_lmdbs=[root / "train_lmdb"],
        eval_lmdbs=[root / "dev_lmdb"],
        pretrained_model=root / "m3_s50.pth",
        max_epoch=50,
        first_batch_size=32,
        num_workers=8,
        max_ratio=40,
        lr=2.5e-5,
        internal_eval_every=100000,
        seed=20260731,
        control_world_size=2,
        control_first_batch_size=16,
    )
    for token in (
        "method_variant: m3_sldr_pilot",
        "use_sldr: True",
        "sldr_reduction: 4",
        "sldr_gamma_init: 0.001",
        "consistency_weight: 0.15",
        "direction_weight: 0.0",
    ):
        assert token in generated, token

    torch.manual_seed(20260918)
    features = torch.randn(2, 16, 2, 13, requires_grad=True)
    adapter = ScriptConditionedResidualAdapter(16, num_scripts=4)
    logits = adapter.predict_scripts(features)
    sldr = ScriptAwareLocalDetailRefinement(
        16, num_scripts=4, reduction=4, gamma_init=1e-3
    )
    refined = sldr(features, logits)
    assert refined.shape == features.shape
    assert logits.shape == (2, 13, 4)
    assert torch.allclose(
        sldr.script_scales,
        torch.full_like(sldr.script_scales, 1e-3),
    )
    refined.mean().backward()
    assert all(parameter.grad is not None for parameter in sldr.parameters())
    print("M3_SLDR_PILOT_PROTOCOL_TESTS_OK")


if __name__ == "__main__":
    main()
