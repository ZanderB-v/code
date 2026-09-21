#!/usr/bin/env python3
"""Fail unless all four public-baseline audits and smoke tests passed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretrained_models import PROJECT_ROOT, sha256_file
from formal_baseline_data import (
    formal_implementation_hashes,
    load_protocol,
    preflight_implementation_hashes,
    protocol_artifact_hashes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Comparison" / "pretrained_baselines_v1" / "outputs",
    )
    args = parser.parse_args()
    models = ("crnn", "svtr", "parseq", "abinet")
    protocol_path = (
        PROJECT_ROOT / "Comparison/pretrained_baselines_v1/protocol.json"
    )
    protocol_sha256 = sha256_file(protocol_path)
    protocol = load_protocol(protocol_path)
    artifact_sha256 = protocol_artifact_hashes(protocol)
    result = {
        "status": "passed",
        "protocol_id": "pretrained_baselines_v1",
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": formal_implementation_hashes(),
        "preflight_implementation_sha256": preflight_implementation_hashes(),
        "artifact_sha256": artifact_sha256,
        "models": {},
        "formal_training_allowed": True,
        "test_evaluated": False,
    }
    errors = []
    formal_audit_path = args.output_dir / "formal_protocol_audit.json"
    if not formal_audit_path.is_file():
        errors.append("missing full formal protocol audit")
        formal_audit = None
    else:
        formal_audit = json.loads(formal_audit_path.read_text(encoding="utf-8"))
        if formal_audit.get("status") != "passed":
            errors.append("full formal protocol audit did not pass")
        if formal_audit.get("protocol_sha256") != protocol_sha256:
            errors.append("full formal protocol audit is stale")
        if formal_audit.get("artifact_sha256") != artifact_sha256:
            errors.append("full formal protocol audit belongs to stale data or weights")
        if formal_audit.get("test_accessed") is not False:
            errors.append("full formal protocol audit accessed Test")
    result["formal_protocol_audit"] = (
        formal_audit.get("status") if formal_audit else "missing"
    )
    for model in models:
        audit_path = args.output_dir / f"{model}_loading_audit.json"
        preflight_path = args.output_dir / f"{model}_single_batch_preflight.json"
        ddp_path = args.output_dir / f"{model}_formal_ddp_preflight.json"
        if (
            not audit_path.is_file()
            or not preflight_path.is_file()
            or not ddp_path.is_file()
        ):
            errors.append(f"missing reports for {model}")
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        ddp = json.loads(ddp_path.read_text(encoding="utf-8"))
        result["models"][model] = {
            "loading_audit": audit["status"],
            "backbone_loaded_ratio": audit["backbone_loaded_ratio"],
            "loaded_parameter_ratio": audit["loaded_parameter_ratio"],
            "single_batch_preflight": preflight["status"],
            "loss": preflight["loss"],
            "formal_ddp_preflight": ddp["status"],
            "formal_ddp_long_label_lengths": ddp[
                "long_label_lengths_per_rank0_batch"
            ],
            "test_evaluated": preflight["test_accessed"],
        }
        if audit["status"] != "passed" or preflight["status"] != "passed":
            errors.append(f"{model} did not pass")
        for label, report in (
            ("loading audit", audit),
            ("single batch preflight", preflight),
            ("formal DDP preflight", ddp),
        ):
            if report.get("protocol_sha256") != protocol_sha256:
                errors.append(f"{model} {label} belongs to a stale protocol")
        if ddp.get("status") != "passed":
            errors.append(f"{model} formal DDP preflight did not pass")
        if ddp.get("test_evaluated") is not False:
            errors.append(f"{model} DDP preflight accessed Test")
        if preflight.get("test_accessed") is not False:
            errors.append(f"{model} accessed Test during preflight")
        if abs(float(audit["backbone_loaded_ratio"]) - 1.0) > 1e-12:
            errors.append(f"{model} backbone loading is not complete")
    if errors:
        result["status"] = "failed"
        result["formal_training_allowed"] = False
    result["errors"] = errors
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "pretrained_baselines_v1_preflight_summary.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)
    print("ALL_PRETRAINED_BASELINES_V1_PREFLIGHTS_OK")


if __name__ == "__main__":
    main()
