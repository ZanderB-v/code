#!/usr/bin/env python3
"""Static protocol checks for the parallel full-S50 SLDR/SCDL study."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "scripts/svtrv2"))

    from dual_order_protocol import METHODS, make_method_config
    from run_m3_full_refinement import TRACKS, names

    expected = {
        "sldr": {
            "method": "m3_sldr_full_v1",
            "tokens": (
                "use_sldr: True",
                "sldr_reduction: 4",
                "sldr_gamma_init: 0.001",
            ),
        },
        "scdl": {
            "method": "m3_scdl_full_v1",
            "tokens": (
                "use_scdl: True",
                "scdl_weight: 0.05",
                "scdl_topk: 5",
                "scdl_temperature: 0.1",
                "scdl_warmup_fraction: 0.2",
            ),
        },
    }
    with tempfile.TemporaryDirectory(prefix="m3_full_protocol_") as temporary:
        temporary_root = Path(temporary)
        checkpoint = temporary_root / "rctc_init.pth"
        checkpoint.write_bytes(b"protocol-test-checkpoint")
        for track, contract in expected.items():
            method = contract["method"]
            assert TRACKS[track]["method"] == method
            spec = METHODS[method]
            assert spec["consistency_weight"] == 0.15
            assert spec["script_weight"] == 0.10
            assert spec["direction_weight"] == 0.0
            assert spec["use_script_adapter"] is True
            generated = make_method_config(
                method=method,
                root=root,
                run_dir=temporary_root / f"tmp_{track}_full_s50",
                project_name=f"m3_{track}_full_s50_protocol_test",
                train_lmdbs=[temporary_root / "train_lmdb"],
                eval_lmdbs=[temporary_root / "dev_lmdb"],
                pretrained_model=checkpoint,
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
                f"method_variant: {method}",
                "consistency_weight: 0.15",
                "use_script_adapter: True",
                *contract["tokens"],
            ):
                assert token in generated, (track, token)
            config = temporary_root / f"{method}.yml"
            config.write_text(generated, encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/svtrv2/validate_dual_order_method_config.py"),
                    "--config", str(config),
                    "--expected-initialization", "checkpoint",
                    "--allow-missing-lmdb",
                ],
                cwd=str(root),
                check=True,
                stdout=subprocess.DEVNULL,
            )
            synthetic, target = names(method)
            assert synthetic == f"svtrv2_s_{method}_dual_order_s50"
            assert target == f"{synthetic}_to_target"

    launcher = (
        root / "scripts/svtrv2/run_m3_full_s50_parallel_tmux.sh"
    ).read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=\"$GPU_ID\"" in launcher
    assert "GPU_ID=\"${GPU_ID:-0}\"" in launcher
    assert "tmux new-window" in launcher
    assert "BATCH_SIZE=32" in launcher
    assert "preflight_complete.json" in launcher
    assert "Preflight is stale" in launcher
    assert "Test are forbidden" in launcher

    runner = (
        root / "scripts/svtrv2/run_m3_full_refinement.py"
    ).read_text(encoding="utf-8")
    assert "reset_scdl_prototypes" in runner
    assert "target_init_prototypes_reset.pth" in runner
    assert "preflight_complete.json" in runner
    assert "Macro CER improvement >= 0.02pp" in runner
    assert '"test_evaluated": False' in runner
    print("M3_FULL_REFINEMENT_PROTOCOL_TESTS_OK")


if __name__ == "__main__":
    main()
