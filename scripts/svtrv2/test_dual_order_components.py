#!/usr/bin/env python3
"""Deterministic CPU invariants for the dual-order method components."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from unittest.mock import patch
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "scripts" / "svtrv2"))
    sys.path.insert(0, str(root / "third_party" / "OpenOCR"))

    import torch
    import numpy as np

    from dual_order_protocol import (
        bidi_layout,
        dual_order_payload,
        normalize_logical_label,
    )
    from openrec.losses import GTCLoss
    from openrec.losses.dual_order_gtc_loss import DualOrderGTCLoss
    from openrec.modeling.decoders.dual_order_gtc_decoder import (
        LocalDirectionConditioner,
        ScriptAwareLocalDetailRefinement,
        ScriptConditionedResidualAdapter,
    )
    from tools.data.ratio_sampler import (
        SAMPLER_PROTOCOL,
        RatioSampler,
        partition_indices_by_ratio,
    )

    class _ExactDivisionDataset:
        ds_width = True
        seed = 20260731
        wh_ratio = np.asarray([1, 1, 2, 2, 5, 5], dtype=np.float64)
        wh_ratio_sort = np.argsort(wh_ratio)

        def __len__(self):
            return len(self.wh_ratio)

    exact_division_sampler = RatioSampler(
        _ExactDivisionDataset(),
        scales=[[128, 32]],
        first_bs=2,
        fix_bs=False,
        divided_factor=[4, 16],
        is_training=True,
        max_ratio=40,
        seed=20260731,
    )
    assert len(exact_division_sampler) == 4
    assert sum(len(batch) for batch in exact_division_sampler.batch_list) == 6

    ddp_dataset = _ExactDivisionDataset()
    ddp_dataset.wh_ratio = np.asarray(
        [1] * 6 + [2] * 5 + [5] * 3,
        dtype=np.float64,
    )
    ddp_dataset.wh_ratio_sort = np.argsort(ddp_dataset.wh_ratio)
    rank_indices = [
        partition_indices_by_ratio(ddp_dataset.wh_ratio, 2, rank)[0]
        for rank in (0, 1)
    ]
    assert set(rank_indices[0]) | set(rank_indices[1]) == set(range(14))
    ddp_samplers = []
    for rank in (0, 1):
        with (
            patch(
                "tools.data.ratio_sampler.torch.distributed.is_available",
                return_value=True,
            ),
            patch(
                "tools.data.ratio_sampler.torch.distributed.is_initialized",
                return_value=True,
            ),
            patch(
                "tools.data.ratio_sampler.torch.distributed.get_world_size",
                return_value=2,
            ),
            patch(
                "tools.data.ratio_sampler.torch.distributed.get_rank",
                return_value=rank,
            ),
        ):
            ddp_samplers.append(
                RatioSampler(
                    ddp_dataset,
                    scales=[[128, 32]],
                    first_bs=2,
                    fix_bs=False,
                    divided_factor=[4, 16],
                    is_training=True,
                    max_ratio=20,
                    seed=20260731,
                )
            )
    batch_specs = [
        [
            (int(batch[0][0]), int(batch[0][1]), len(batch))
            for batch in sampler.batch_list
        ]
        for sampler in ddp_samplers
    ]
    assert len(ddp_samplers[0]) == len(ddp_samplers[1])
    assert batch_specs[0] == batch_specs[1]
    sampled_union = {
        int(sample[2])
        for sampler in ddp_samplers
        for batch in sampler.batch_list
        for sample in batch
    }
    assert sampled_union == set(range(14))
    single_gpu_emulation = RatioSampler(
        ddp_dataset,
        scales=[[128, 32]],
        first_bs=4,
        fix_bs=False,
        divided_factor=[4, 16],
        is_training=True,
        max_ratio=20,
        seed=20260731,
        control_world_size=2,
        control_first_bs=2,
    )
    emulated_specs = Counter(
        (int(batch[0][0]), int(batch[0][1]), len(batch))
        for batch in single_gpu_emulation.batch_list
    )
    merged_ddp_specs = Counter(
        (int(batch[0][0]), int(batch[0][1]), len(batch) * 2)
        for batch in ddp_samplers[0].batch_list
    )
    assert emulated_specs == merged_ddp_specs
    assert len(single_gpu_emulation) == len(ddp_samplers[0])
    assert {
        int(sample[2])
        for batch in single_gpu_emulation.batch_list
        for sample in batch
    } == set(range(14))
    for epoch in (1, 2):
        yielded_specs = []
        for sampler in ddp_samplers:
            sampler.set_epoch(epoch)
            yielded_specs.append(
                [
                    (int(batch[0][0]), int(batch[0][1]), len(batch))
                    for batch in sampler
                ]
            )
        assert yielded_specs[0] == yielded_specs[1]

    examples = [
        "ئۇيغۇرچە 123 ABC",
        "(ئۇيغۇرچە)",
        "abc 12 ئۇيغۇرچە 34",
        "ئاا 11 ئاا",
    ]
    layouts = []
    for text in examples:
        layout = bidi_layout(text, "ug")
        logical_to_visual = layout["logical_to_visual"]
        visual_to_logical = layout["visual_to_logical"]
        assert sorted(logical_to_visual) == list(range(len(text)))
        assert sorted(visual_to_logical) == list(range(len(text)))
        for logical_index, visual_index in enumerate(logical_to_visual):
            assert visual_to_logical[visual_index] == logical_index
        layouts.append(
            {
                "logical": text,
                "visual": layout["visual_text"],
                "permutation": logical_to_visual,
            }
        )

    mirrored = bidi_layout("(ئۇيغۇرچە)", "ug")
    mirrored_positions = [
        logical_index
        for logical_index, visual_index in enumerate(
            mirrored["logical_to_visual"]
        )
        if "(ئۇيغۇرچە)"[logical_index]
        != mirrored["visual_text"][visual_index]
    ]
    assert mirrored_positions, "Mirrored bidi symbols must be detectable"

    source_with_controls = "\u0626\u0627\u06be \u0626\u200d\u0627\u06be \u0626\u200c\u0627\u06be"
    normalized, removed = normalize_logical_label(source_with_controls)
    normalized_layout = bidi_layout(normalized, "ug")
    normalized_payload = dual_order_payload(
        {
            "id": "zero_width_control_invariant",
            "language": "ug",
            "logical_text": source_with_controls,
            "ctc_text": normalized_layout["visual_text"],
        }
    )
    assert len(normalized_payload["logical_to_visual"]) == len(normalized)
    assert sorted(normalized_payload["logical_to_visual"]) == list(
        range(len(normalized))
    )
    assert removed == ["\u200d", "\u200c"]
    assert normalized_payload["removed_format_controls"] == [
        "U+200D",
        "U+200C",
    ]

    loss = DualOrderGTCLoss()
    ctc_logits = torch.tensor(
        [[[12.0, 8.0, 0.0], [0.0, 4.0, 0.0]]], dtype=torch.float32
    )
    character_mass, conditional, nonblank = (
        loss._ctc_character_distributions(ctc_logits, 1.0)
    )
    scores = loss._token_frame_scores(
        character_mass[0], torch.tensor([0], dtype=torch.long)
    )
    alignment = torch.softmax(scores, dim=-1)
    assert alignment[0, 1] > alignment[0, 0] * 10
    assert nonblank[0, 1] > nonblank[0, 0] * 10
    assert torch.allclose(conditional.sum(dim=-1), torch.ones_like(nonblank))

    uniform = torch.full((2, 7), 1.0 / 7.0)
    concentrated = torch.full((2, 7), 1e-7)
    concentrated[:, 0] = 1.0 - 6e-7
    assert loss._inverse_entropy_confidence(uniform).max() < 1e-6
    assert loss._inverse_entropy_confidence(concentrated).min() > 0.99

    torch.manual_seed(20260731)
    ctc_prediction = torch.randn(2, 7, 6)
    ctc_label = torch.tensor([[1, 2, 0], [2, 3, 0]], dtype=torch.long)
    ctc_length = torch.tensor([2, 2], dtype=torch.long)
    semantic_loss = torch.tensor(1.75)
    predictions = {
        "ctc_pred": ctc_prediction,
        "gtc_pred": [{"loss": semantic_loss}, torch.randn(2, 3, 7)],
    }
    # Match the 17 tensors emitted by DualOrderGTCLabelEncode + KeepKeys.
    batch = [None] * 15 + [ctc_label, ctc_length]
    baseline_loss = GTCLoss(
        gtc_loss={"name": "SMTRLoss"}, ctc_weight=0.1, gtc_weight=1.0
    )(predictions, batch)["loss"]
    dual_base_loss = DualOrderGTCLoss(
        gtc_loss={"name": "SMTRLoss"},
        ctc_weight=0.1,
        gtc_weight=1.0,
        consistency_weight=0.0,
        script_weight=0.0,
        direction_weight=0.0,
    )(predictions, batch)["loss"]
    assert torch.allclose(baseline_loss, dual_base_loss, atol=1e-7, rtol=0)

    # Exercise every proposed loss jointly with both identity and reversed
    # logical-to-visual permutations. This catches silent tensor-index drift.
    torch.manual_seed(20260801)
    method_ctc = torch.randn(2, 6, 6, requires_grad=True)
    method_sgm = torch.randn(2, 3, 7, requires_grad=True)
    method_script = torch.randn(2, 6, 4, requires_grad=True)
    method_direction = torch.randn(2, 6, 2, requires_grad=True)
    method_semantic_loss = method_sgm.square().mean()
    method_predictions = {
        "ctc_pred": method_ctc,
        "gtc_pred": [{"loss": method_semantic_loss}, method_sgm],
        "script_logits": method_script,
        "direction_logits": method_direction,
    }
    method_batch = [None] * 17
    method_batch[8] = torch.tensor([3, 3], dtype=torch.long)
    method_batch[9] = torch.tensor(
        [[1, 2, 3], [1, 2, 3]], dtype=torch.long
    )
    method_batch[10] = torch.tensor(
        [[1, 2, 3], [3, 2, 1]], dtype=torch.long
    )
    method_batch[11] = torch.tensor(
        [[0, 1, 2], [2, 1, 0]], dtype=torch.long
    )
    method_batch[12] = torch.tensor(
        [[0, 0, 0], [1, 1, 0]], dtype=torch.long
    )
    method_batch[13] = torch.tensor(
        [[0, 0, 3], [1, 1, 3]], dtype=torch.long
    )
    method_batch[14] = torch.ones(2, 3, dtype=torch.long)
    method_batch[15] = torch.tensor(
        [[1, 2, 3], [1, 2, 3]], dtype=torch.long
    )
    method_batch[16] = torch.tensor([3, 3], dtype=torch.long)
    method_loss = DualOrderGTCLoss(
        gtc_loss={"name": "SMTRLoss"},
        ctc_weight=0.1,
        gtc_weight=1.0,
        consistency_weight=0.15,
        script_weight=0.10,
        direction_weight=0.05,
    )(method_predictions, method_batch)
    for name, value in method_loss.items():
        assert torch.isfinite(value).all(), f"non-finite {name}: {value}"
    method_loss["loss"].backward()
    for name, tensor in (
        ("ctc", method_ctc),
        ("sgm", method_sgm),
        ("script", method_script),
        ("direction", method_direction),
    ):
        assert tensor.grad is not None, f"missing {name} gradient"
        assert torch.isfinite(tensor.grad).all(), f"non-finite {name} gradient"

    torch.manual_seed(20260731)
    adapter = ScriptConditionedResidualAdapter(channels=8, num_scripts=4)
    features = torch.randn(2, 8, 2, 11)
    adapted, script_logits = adapter(features)
    assert torch.equal(adapted, features)
    assert tuple(script_logits.shape) == (2, 11, 4)
    direction = LocalDirectionConditioner(channels=8)
    conditioned, direction_logits = direction(features)
    assert torch.equal(conditioned, features)
    assert tuple(direction_logits.shape) == (2, 11, 2)

    sldr = ScriptAwareLocalDetailRefinement(
        channels=8, num_scripts=4, reduction=4, gamma_init=1e-3
    )
    sldr_input = features.detach().clone().requires_grad_(True)
    refined = sldr(sldr_input, script_logits.detach())
    assert tuple(refined.shape) == tuple(features.shape)
    assert torch.allclose(
        sldr.script_scales,
        torch.full_like(sldr.script_scales, 1e-3),
    )
    assert not torch.equal(refined, sldr_input)
    refined.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in sldr.parameters()
    )

    result = {
        "status": "DUAL_ORDER_COMPONENT_INVARIANTS_OK",
        "bidi_examples": layouts,
        "mirrored_positions_masked": mirrored_positions,
        "zero_width_control_normalization": {
            "source_length": len(source_with_controls),
            "normalized_length": len(normalized),
            "removed": normalized_payload["removed_format_controls"],
            "bijection_preserved": True,
        },
        "blank_aware_alignment": alignment.tolist(),
        "nonblank_mass": nonblank.tolist(),
        "uniform_reliability": loss._inverse_entropy_confidence(
            uniform
        ).tolist(),
        "official_base_loss_equivalence": True,
        "full_method_loss_finite_backward": True,
        "adapter_identity_initialization": True,
        "script_adapter_identity_initialization": True,
        "direction_conditioner_identity_initialization": True,
        "sldr_finite_forward_backward": True,
        "sldr_gamma_init": 1e-3,
        "ratio_sampler_exact_division_retained": True,
        "ratio_sampler_protocol": SAMPLER_PROTOCOL,
        "ratio_sampler_ddp_equal_steps": True,
        "ratio_sampler_ddp_equal_batch_shapes": True,
        "ratio_sampler_ddp_equal_yielded_shapes_epochs": [1, 2],
        "ratio_sampler_ddp_full_union_coverage": True,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
