#!/usr/bin/env python3
"""Fast synchronous preflight for formal P1/MSR experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from audit_formal_research_protocol import audit as audit_research_protocol
from p1_msr_protocol import EVAL_INFERENCE_BATCH_SIZE, PROTOCOL_ID
from validate_p1_msr_config import UNION14M, UNION14M_SHA256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--minimum-gpus", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    modules = ("bidi.algorithm", "lmdb", "PIL", "yaml", "torch")
    imported = {}
    for name in modules:
        module = importlib.import_module(name)
        imported[name] = getattr(module, "__version__", "available")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the active environment")
    gpu_count = torch.cuda.device_count()
    if gpu_count < args.minimum_gpus:
        raise RuntimeError(
            f"Expected at least {args.minimum_gpus} GPUs, found {gpu_count}"
        )

    required = {
        "protocol_manifest": (
            root / "00_docs" / "frozen_protocol_v2" / "protocol_v2_manifest.json"
        ),
        "protocol_verification": (
            root / "00_docs" / "frozen_protocol_v2" / "verification_report.json"
        ),
        "target_metadata": (
            root
            / "01_data_preparation"
            / "real_line_dataset_eval_reviewed"
            / "metadata.jsonl"
        ),
        "character_dictionary": (
            root
            / "04_model_training"
            / "character_dict_hz_ug_kk_v1"
            / "character_dict.txt"
        ),
    }
    for scale in ("s10", "s25", "s50"):
        required[f"synthetic_{scale}"] = (
            root
            / "03_synthetic_generation"
            / "synthetic_formal_v2"
            / "subsets"
            / scale
            / "metadata.jsonl"
        )
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Formal input files are missing:\n" + "\n".join(missing)
        )

    verification = json.loads(
        required["protocol_verification"].read_text(encoding="utf-8-sig")
    )
    if verification.get("status") not in ("passed", "verified", "ok", "frozen"):
        raise ValueError(
            "Protocol V2 verification report is not successful: "
            f"{verification}"
        )
    if verification.get("errors"):
        raise ValueError(
            "Protocol V2 verification still contains errors: "
            f"{verification['errors']}"
        )

    official = Path(UNION14M)
    if not official.is_file():
        raise FileNotFoundError(f"Union14M checkpoint is missing: {official}")
    official_sha = sha256(official)
    if official_sha != UNION14M_SHA256:
        raise ValueError(
            f"Union14M SHA-256 mismatch: expected {UNION14M_SHA256}, "
            f"got {official_sha}"
        )

    research_audit = audit_research_protocol(root)
    if research_audit["status"] != "passed":
        print(json.dumps(research_audit, ensure_ascii=False, indent=2))
        raise RuntimeError(
            "Formal research audit failed. Repair every listed leakage or "
            "direction-protocol issue before allocating GPUs."
        )

    critical_code = (
        "00_docs/FORMAL_RCTC_RUNBOOK_V2.md",
        "scripts/protocol/freeze_protocol_v2.py",
        "scripts/svtrv2/audit_formal_research_protocol.py",
        "scripts/svtrv2/p1_msr_protocol.py",
        "scripts/svtrv2/audit_p1_ratio_sampler_ddp.py",
        "scripts/svtrv2/test_dual_order_components.py",
        "scripts/svtrv2/metrics_v1.py",
        "scripts/svtrv2/run_p1_rctc_stage.py",
        "scripts/svtrv2/run_p1_rctc_baseline_chain.py",
        "scripts/svtrv2/run_formal_prelaunch_audit.sh",
        "scripts/svtrv2/evaluate_d2_checkpoint.py",
        "scripts/svtrv2/infer_label_file_metrics.py",
        "scripts/svtrv2/u2_visual_predictions_to_logical_metrics.py",
        "scripts/svtrv2/audit_b1_controlled_comparison.py",
        "scripts/svtrv2/convert_rctc_checkpoint_to_full_svtrv2.py",
        "scripts/svtrv2/dual_order_protocol.py",
        "scripts/svtrv2/prepare_b1_full_svtrv2.py",
        "scripts/svtrv2/prepare_dual_order_method.py",
        "scripts/svtrv2/run_b1_full_svtrv2.py",
        "scripts/svtrv2/run_dual_order_method_suite.py",
        "scripts/svtrv2/smoke_dual_order_method.py",
        "scripts/svtrv2/smoke_p1_full_svtrv2_config.py",
        "scripts/svtrv2/test_dual_order_components.py",
        "scripts/svtrv2/validate_dual_order_method_config.py",
        "scripts/svtrv2/validate_p1_full_svtrv2_config.py",
        "third_party/OpenOCR/openrec/losses/dual_order_gtc_loss.py",
        "third_party/OpenOCR/openrec/modeling/decoders/dual_order_gtc_decoder.py",
        "third_party/OpenOCR/openrec/preprocess/dual_order_gtc_label_encode.py",
        "third_party/OpenOCR/tools/data/ratio_dataset_tvresize.py",
        "third_party/OpenOCR/tools/data/__init__.py",
        "third_party/OpenOCR/tools/data/ratio_sampler.py",
        "third_party/OpenOCR/tools/engine/trainer.py",
        "third_party/OpenOCR/tools/infer_rec.py",
        "third_party/OpenOCR/tools/utils/ckpt.py",
    )
    code_hashes = {}
    for relative in critical_code:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Critical formal code is missing: {path}")
        code_hashes[relative] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }

    git = {
        "available": False,
        "repository_present": (root / ".git").exists(),
        "executable": shutil.which("git"),
        "commit": None,
        "dirty": None,
    }
    if git["repository_present"] and git["executable"]:
        commit = subprocess.run(
            [git["executable"], "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            [git["executable"], "status", "--porcelain"],
            cwd=str(root),
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        git = {
            "available": True,
            "repository_present": True,
            "executable": git["executable"],
            "commit": commit,
            "dirty": bool(status),
            "status": status.splitlines(),
        }

    result = {
        "status": "P1_FORMAL_PREFLIGHT_OK",
        "root": str(root),
        "preprocess_protocol": PROTOCOL_ID,
        "external_eval_batch_size": EVAL_INFERENCE_BATCH_SIZE,
        "python_modules": imported,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "cuda": {
            "available": True,
            "gpu_count": gpu_count,
            "devices": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
        },
        "required_files": {name: str(path) for name, path in required.items()},
        "union14m": {
            "path": str(official),
            "sha256": official_sha,
            "bytes": official.stat().st_size,
        },
        "research_audit": research_audit,
        "code_provenance": {
            "git": git,
            "critical_file_hashes": code_hashes,
        },
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
