#!/usr/bin/env python3
"""CPU protocol and CTC-alignment tests for the M3+SCDL pilot."""

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
    from openrec.losses.dual_order_gtc_loss import DualOrderGTCLoss
    from openrec.modeling.decoders.dual_order_gtc_decoder import (
        ScriptAwareConfusionDiscriminator,
    )

    spec = METHODS["m3_scdl_pilot"]
    assert spec["consistency_weight"] == 0.15
    assert spec["script_weight"] == 0.10
    assert spec["scdl_weight"] == 0.05
    assert spec["scdl_topk"] == 5
    assert spec["scdl_warmup_fraction"] == 0.2

    generated = make_method_config(
        method="m3_scdl_pilot",
        root=root,
        run_dir=root / "tmp_scdl_run",
        project_name="m3_scdl_protocol_test",
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
        "method_variant: m3_scdl_pilot",
        "use_scdl: True",
        "scdl_use_space_char: True",
        "scdl_weight: 0.05",
        "scdl_topk: 5",
        "scdl_temperature: 0.1",
        "scdl_warmup_fraction: 0.2",
        "consistency_weight: 0.15",
    ):
        assert token in generated, token

    dictionary = root / "04_model_training/character_dict_hz_ug_kk_v1/character_dict.txt"
    character_count = len(dictionary.read_text(encoding="utf-8-sig").splitlines()) + 1
    discriminator = ScriptAwareConfusionDiscriminator(
        channels=8,
        num_characters=character_count,
        character_dict_path=dictionary,
        use_space_char=True,
    )
    assert torch.allclose(
        discriminator.projection.weight, torch.eye(8)
    )
    assert discriminator.prototypes.shape == (character_count, 8)
    assert set(discriminator.prototype_script_ids.tolist()).issubset({0, 1, 2, 3})

    loss_fn = DualOrderGTCLoss(
        scdl_weight=0.05,
        scdl_topk=5,
        scdl_temperature=0.1,
        scdl_warmup_fraction=0.2,
    )
    loss_fn.set_training_progress(19, 100)
    assert loss_fn._scdl_active_weight() == 0.0
    loss_fn.set_training_progress(20, 100)
    assert loss_fn._scdl_active_weight() == 0.05
    logits = torch.full((2, 5, 4), -4.0)
    logits[:, :, 0] = 1.0
    logits[0, 1:4, 1] = 5.0
    logits[1, 1:4, 2] = 5.0
    visual_ids = torch.tensor([[1, 0], [2, 0]])
    lengths = torch.tensor([1, 1])
    gamma, valid = loss_fn._ctc_token_posteriors(logits, visual_ids, lengths)
    assert gamma.shape == (2, 1, 5)
    assert valid.tolist() == [True, True]
    assert torch.all(gamma.sum(dim=-1) > 0)

    frame_features = torch.zeros(2, 5, 8)
    frame_features[0, :, 0] = 1.0
    frame_features[1, :, 1] = 1.0
    frame_features.requires_grad_(True)
    prototypes = torch.zeros(3, 8)
    counts = torch.zeros(3)
    prototype_scripts = torch.tensor([0, 0, 1])
    batch = [None] * 15
    batch[8] = lengths
    batch[9] = visual_ids
    batch[13] = torch.tensor([[0, -1], [0, -1]])
    predicts = {
        "ctc_pred": logits,
        "scdl_frame_features": frame_features,
        "scdl_prototypes": prototypes,
        "scdl_prototype_counts": counts,
        "scdl_prototype_script_ids": prototype_scripts,
    }
    warmup_loss, _, _, _ = loss_fn._scdl_loss(predicts, batch)
    assert float(warmup_loss) == 0.0
    assert counts[:2].tolist() == [1.0, 1.0]
    active_loss, valid_tokens, _, _ = loss_fn._scdl_loss(predicts, batch)
    assert torch.isfinite(active_loss) and float(active_loss) > 0.0
    assert valid_tokens == 2
    active_loss.backward()
    assert frame_features.grad is not None
    assert torch.isfinite(frame_features.grad).all()
    print("M3_SCDL_PILOT_PROTOCOL_TESTS_OK")


if __name__ == "__main__":
    main()
