#!/usr/bin/env python3
"""Numerical branch-isolation tests; optional two-device reducer smoke."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / "third_party/OpenOCR"))
    sys.path.insert(0, str(args.root / "scripts/svtrv2"))
    import torch
    import yaml
    from torch import nn
    from openrec.modeling.decoders import dual_order_gtc_decoder as module
    from run_semantic_direction_study import metric_row, select_structure
    from dual_order_protocol import make_method_config
    from smoke_dual_order_method import audit_initialization

    torch.set_num_threads(1)
    torch.manual_seed(317)
    device = torch.device("cpu")
    if args.ddp:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group("nccl")

    class Head(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.linear = nn.Linear(8, 5)

        def forward(self, x, data=None):
            if x.ndim == 4:
                x = x.mean(dim=2).transpose(1, 2)
            return self.linear(x)

    def build(mode):
        with patch.object(module, "build_decoder", side_effect=Head):
            return module.DualOrderGTCDecoder(
                in_channels=8, out_channels=[5, 5], gtc_decoder={},
                ctc_decoder={}, use_script_adapter=True,
                semantic_direction=mode,
            ).to(device)

    reference = build("none")
    source_state = {"decoder." + k: v for k, v in reference.state_dict().items()}
    target_state = {"decoder." + k: v for k, v in build("sample_script_gated").state_dict().items()}
    fresh = audit_initialization(target_state, source_state, True, True)
    assert "decoder.direction_gate_logits" in fresh
    damaged = dict(source_state)
    damaged.pop("decoder.gtc_decoder.linear.weight")
    try:
        audit_initialization(target_state, damaged, True, True)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing shared SGM weights were silently accepted")
    x = torch.randn(2, 8, 2, 9, device=device, requires_grad=True)
    results = {}
    for mode in ("sgm_only", "script_gated", "sample_script_gated"):
        model = build(mode)
        model.load_state_dict(reference.state_dict(), strict=False)
        model.train()
        before = model(x)
        base = reference(x)
        assert torch.equal(before["ctc_pred"], base["ctc_pred"])
        assert torch.equal(before["gtc_pred"], base["gtc_pred"])
        if mode in ("script_gated", "sample_script_gated"):
            fresh = build(mode).train()
            opt = torch.optim.SGD(fresh.parameters(), lr=.1)
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                pred = fresh(x.detach())
                pred["gtc_pred"].square().mean().backward()
                gate_grad = fresh.direction_gate_logits.grad
                assert gate_grad is not None and torch.isfinite(gate_grad).all()
                if step == 1:
                    assert gate_grad.abs().sum() > 0, "Gate is stuck after zero-init"
                opt.step()
        # Break the zero residual deliberately, then verify the CTC graph is isolated.
        with torch.no_grad():
            model.semantic_direction_conditioner.bias_embed.normal_()
        after = model(x)
        assert torch.equal(before["ctc_pred"], after["ctc_pred"])
        assert not torch.equal(before["gtc_pred"], after["gtc_pred"])
        direction_params = list(model.semantic_direction_conditioner.parameters())
        if mode in ("script_gated", "sample_script_gated"):
            direction_params.append(model.direction_gate_logits)
            initial_gate = .01 if mode == "script_gated" else 1 / (1 + __import__("math").exp(4))
            assert torch.allclose(model.direction_gate_logits.sigmoid(),
                                  torch.full((4,), initial_gate, device=device))
        grads = torch.autograd.grad(after["ctc_pred"].square().mean(), direction_params,
                                    allow_unused=True, retain_graph=True)
        assert all(grad is None for grad in grads)
        after["gtc_pred"].square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        if mode in ("script_gated", "sample_script_gated"):
            grad = model.direction_gate_logits.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.semantic_direction_conditioner.parameters())
        model.eval()
        model.infer_gtc = False
        with patch.object(model.semantic_direction_conditioner, "forward",
                          side_effect=AssertionError("Direction executed at CTC inference")):
            prediction = model(x)
        assert torch.equal(prediction, after["ctc_pred"])

        if args.ddp:
            model = build(mode).train()
            ddp = nn.parallel.DistributedDataParallel(model, device_ids=[device.index],
                                                     find_unused_parameters=False)
            optimizer = torch.optim.SGD(ddp.parameters(), lr=.05)
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                out = ddp(x.detach())
                # V2 has no independent direction loss, including in this reducer test.
                loss = sum(out[key].square().mean() for key in
                           ("ctc_pred", "gtc_pred", "script_logits"))
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all()
                           for p in ddp.parameters())
                optimizer.step()
            torch.distributed.barrier()
        results[mode] = {"initial_identity": True, "ctc_direction_gradient_absent": True,
                         "ctc_output_invariant": True, "semantic_gradient_finite": True,
                         "ctc_inference_skips_direction": True,
                         "zero_init_gate_learns_after_first_update": True if mode in ("script_gated", "sample_script_gated") else None,
                         "three_step_ddp": args.ddp}

    rows = [dict(model="B1", macro_cer=.03, macro_line_accuracy=.85, candidate=False),
            dict(model="M3", macro_cer=.025, macro_line_accuracy=.87, candidate=True),
            dict(model="D1", macro_cer=.026, macro_line_accuracy=.90, candidate=True),
            dict(model="D2", macro_cer=.024, macro_line_accuracy=.84, candidate=True)]
    assert select_structure(rows)["selected_model"] == "M3"
    rows[-1]["macro_line_accuracy"] = .86
    assert select_structure(rows)["selected_model"] == "D2"
    rows[-1]["macro_cer"] = .025
    assert select_structure(rows)["selected_model"] == "M3"
    for row in rows:
        row["candidate"] = False
    assert select_structure(rows)["selected_model"] is None
    results["selection_primary_and_guardrail"] = True
    results["shared_SGM_initialization_required"] = True
    configs = {}
    for variant in (
        "m3", "full", "m3_alpha020", "m3_alpha025", "m3_alpha030",
        "dir_d1", "dir_d2", "dir_d1_v2", "dir_d2_v2",
    ):
        configs[variant] = yaml.safe_load(make_method_config(
            method=variant, root=args.root, run_dir=args.root / "unused_run",
            project_name="unit_config", train_lmdbs=[args.root / "unused_train"],
            eval_lmdbs=[args.root / "unused_dev"], pretrained_model=args.root / "unused.pth",
            max_epoch=50, first_batch_size=16, num_workers=4, max_ratio=40,
            lr=2.5e-5, internal_eval_every=100000, seed=20260731))
    assert configs["m3"]["Loss"]["consistency_weight"] == .15
    assert configs["m3_alpha020"]["Loss"]["consistency_weight"] == .20
    assert configs["m3_alpha025"]["Loss"]["consistency_weight"] == .25
    assert configs["full"]["Architecture"]["Decoder"]["use_local_direction"] is True
    for variant in ("m3_alpha030", "dir_d1", "dir_d2"):
        assert configs[variant]["Loss"]["consistency_weight"] == .30
        assert not configs[variant]["Architecture"]["Decoder"]["use_local_direction"]
        assert configs[variant]["Train"] == configs["m3"]["Train"]
        assert configs[variant]["Eval"] == configs["m3"]["Eval"]
    for variant in ("dir_d1", "dir_d2"):
        architecture = copy.deepcopy(configs[variant]["Architecture"])
        del architecture["Decoder"]["semantic_direction"]
        del architecture["Decoder"]["direction_gate_initial"]
        assert architecture == configs["m3_alpha030"]["Architecture"]
    results["locked_alpha_and_legacy_configs_preserved"] = True
    for variant in ("dir_d1_v2", "dir_d2_v2"):
        assert configs[variant]["Loss"] == configs["m3_alpha030"]["Loss"]
    assert "direction_gate_init_logit" not in configs["dir_d1_v2"]["Architecture"]["Decoder"]
    assert configs["dir_d2_v2"]["Architecture"]["Decoder"]["direction_gate_init_logit"] == -4.0
    gated = build("sample_script_gated")
    with torch.no_grad():
        gated.direction_gate_logits.copy_(torch.tensor([-4., -3., -2., -1.], device=device))
    fake_script = torch.zeros(2, 9, 4, device=device)
    fake_script[0, :, 0] = 10
    fake_script[1, :, 1] = 10
    gate = gated.sample_script_gate(fake_script)
    assert gate.shape == (2,), "Sample scalar gate must not have a width/token axis"
    assert torch.equal(gate, gated.direction_gate_logits.sigmoid()[:2])
    results["sample_scalar_routing_no_new_loss"] = True
    fixture = {"dev": {"split": "dev", "prediction_branch": "ctc",
        "macro_cer": .1, "macro_line_accuracy": .8,
        "languages": {lang: dict(cer=.1, wer=.2, one_minus_ned_macro=.9,
                                  line_accuracy=.8) for lang in ("zh", "ug", "kk")}},
        "checkpoint_selection": "clean_target_dev_macro_CER", "best_epoch": 2,
        "best_clean_dev_macro_cer": .1, "best_checkpoint_sha256": "fixture"}
    metric_row("fixture", fixture)
    fixture["best_clean_dev_macro_cer"] = .09
    try:
        metric_row("fixture", fixture)
    except ValueError:
        pass
    else:
        raise AssertionError("Mixed selected-epoch and report metrics were accepted")
    results["inconsistent_metrics_rejected"] = True
    print(json.dumps({"status": "SEMANTIC_DIRECTION_COMPONENTS_OK", "checks": results}, indent=2))
    if args.ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
