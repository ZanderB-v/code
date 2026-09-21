#!/usr/bin/env python3
"""Fail-fast validator for formal P1/MSR SVTRv2-S + RCTC configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from p1_msr_protocol import (
    BASE_HEIGHT,
    BASE_SHAPES,
    DEFAULT_MAX_RATIO,
    DIVIDED_FACTOR,
    MAX_TEXT_LENGTH,
    PROTOCOL_ID,
    SAMPLER_PROTOCOL_ID,
    SAMPLER_SCALES,
    assert_p1_config_text,
)


UNION14M = (
    "/home/wudayu/models/openocr_svtrv2/official/"
    "svtrv2_s_union14m/best.pth"
)
UNION14M_SHA256 = (
    "4c4e21c8eabd7a38519d9f10d6ffa286d470a07b1a8976109fe2b995a9414970"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--expected-initialization",
        choices=("random", "union14m", "checkpoint"),
        required=True,
    )
    parser.add_argument(
        "--allow-missing-lmdb",
        action="store_true",
        help="Only for local static tests; formal server runs must not use it.",
    )
    return parser.parse_args()


def require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(f"{field}: expected {expected!r}, got {actual!r}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_dataset(
    cfg: dict[str, Any],
    section: str,
    allow_missing_lmdb: bool,
) -> list[dict[str, Any]]:
    block = cfg[section]
    dataset = block["dataset"]
    sampler = block["sampler"]
    loader = block["loader"]

    require_equal(
        dataset.get("name"),
        "RatioDataSetTVResize",
        f"{section}.dataset.name",
    )
    require_equal(dataset.get("ds_width"), True, f"{section}.dataset.ds_width")
    require_equal(dataset.get("padding"), False, f"{section}.dataset.padding")
    require_equal(
        dataset.get("base_shape"),
        BASE_SHAPES,
        f"{section}.dataset.base_shape",
    )
    require_equal(
        dataset.get("base_h"),
        BASE_HEIGHT,
        f"{section}.dataset.base_h",
    )
    require_equal(sampler.get("name"), "RatioSampler", f"{section}.sampler.name")
    require_equal(
        sampler.get("scales"),
        SAMPLER_SCALES,
        f"{section}.sampler.scales",
    )
    require_equal(
        sampler.get("divided_factor"),
        DIVIDED_FACTOR,
        f"{section}.sampler.divided_factor",
    )
    require_equal(
        sampler.get("max_ratio"),
        DEFAULT_MAX_RATIO,
        f"{section}.sampler.max_ratio",
    )
    require_equal(
        loader.get("max_ratio"),
        DEFAULT_MAX_RATIO,
        f"{section}.loader.max_ratio",
    )
    if not isinstance(sampler.get("first_bs"), int):
        raise TypeError(f"{section}.sampler.first_bs must be an integer")

    try:
        import lmdb  # type: ignore
    except Exception as exc:
        if not allow_missing_lmdb:
            raise RuntimeError(
                "Install lmdb in openocr_svtrv2: python -m pip install lmdb"
            ) from exc
        lmdb = None

    manifests = []
    for raw_path in dataset.get("data_dir_list") or []:
        lmdb_path = Path(raw_path)
        manifest_path = lmdb_path / "msr_lmdb_manifest.json"
        if not manifest_path.is_file():
            if allow_missing_lmdb:
                manifests.append(
                    {"path": str(lmdb_path), "status": "missing_allowed"}
                )
                continue
            raise FileNotFoundError(
                f"{section} LMDB manifest is missing: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        require_equal(
            manifest.get("preprocess_protocol"),
            PROTOCOL_ID,
            f"{manifest_path}.preprocess_protocol",
        )
        if not (lmdb_path / "data.mdb").is_file():
            raise FileNotFoundError(f"LMDB data file is missing: {lmdb_path}")
        if lmdb is not None:
            environment = lmdb.open(
                str(lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            try:
                with environment.begin(write=False) as transaction:
                    raw_count = transaction.get(b"num-samples")
                    if raw_count is None:
                        raise ValueError(
                            f"LMDB num-samples is missing: {lmdb_path}"
                        )
                    lmdb_rows = int(raw_count)
                    require_equal(
                        lmdb_rows,
                        manifest.get("rows"),
                        f"{lmdb_path}.num-samples",
                    )
                    for index in (1, lmdb_rows):
                        for prefix in ("image", "label", "wh"):
                            key = f"{prefix}-{index:09d}".encode("ascii")
                            if transaction.get(key) is None:
                                raise ValueError(
                                    f"LMDB boundary key is missing: "
                                    f"{lmdb_path}/{key.decode('ascii')}"
                                )
            finally:
                environment.close()
        manifests.append(manifest)
    if not manifests:
        raise ValueError(f"{section} has no LMDB data_dir_list entries")
    return manifests


def validate_initialization(
    cfg: dict[str, Any],
    expected: str,
) -> dict[str, Any] | None:
    global_cfg = cfg["Global"]
    pretrained = global_cfg.get("pretrained_model")
    checkpoint = global_cfg.get("checkpoints")
    if checkpoint not in (None, ""):
        raise ValueError(
            "Formal baseline initialization must use pretrained_model, not checkpoints"
        )
    if expected == "random":
        if pretrained not in (None, ""):
            raise ValueError(
                f"Random initialization has non-empty pretrained_model: {pretrained}"
            )
        return None
    if not isinstance(pretrained, str) or not pretrained.strip():
        raise ValueError(f"{expected} initialization requires pretrained_model")
    if expected == "union14m":
        require_equal(pretrained, UNION14M, "Global.pretrained_model")
    elif pretrained == UNION14M:
        raise ValueError(
            "Checkpoint initialization unexpectedly points directly to Union14M"
        )
    pretrained_path = Path(pretrained)
    if not pretrained_path.is_file():
        raise FileNotFoundError(
            f"Initialization checkpoint is missing: {pretrained_path}"
        )
    checkpoint_sha256 = sha256(pretrained_path)
    if expected == "union14m" and checkpoint_sha256 != UNION14M_SHA256:
        raise ValueError(
            "Union14M checkpoint SHA-256 mismatch: "
            f"expected {UNION14M_SHA256}, got {checkpoint_sha256}"
        )
    return {
        "path": str(pretrained_path),
        "sha256": checkpoint_sha256,
        "bytes": pretrained_path.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    text = config_path.read_text(encoding="utf-8-sig")
    assert_p1_config_text(text)
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Install PyYAML in openocr_svtrv2: python -m pip install pyyaml"
        ) from exc
    cfg = yaml.safe_load(text)

    require_equal(
        cfg["Global"].get("preprocess_protocol"),
        PROTOCOL_ID,
        "Global.preprocess_protocol",
    )
    require_equal(
        cfg["Global"].get("sampler_protocol"),
        SAMPLER_PROTOCOL_ID,
        "Global.sampler_protocol",
    )
    require_equal(
        cfg["Global"].get("max_text_length"),
        MAX_TEXT_LENGTH,
        "Global.max_text_length",
    )
    require_equal(cfg["Architecture"].get("algorithm"), "SVTRv2", "algorithm")
    require_equal(
        cfg["Architecture"]["Encoder"].get("name"),
        "SVTRv2LNConvTwo33",
        "Architecture.Encoder.name",
    )
    require_equal(
        cfg["Architecture"]["Decoder"].get("name"),
        "RCTCDecoder",
        "Architecture.Decoder.name",
    )
    require_equal(cfg["Loss"].get("name"), "CTCLoss", "Loss.name")
    if not isinstance(cfg["Optimizer"].get("lr"), (float, int)):
        raise TypeError(
            "Optimizer.lr must be numeric, not a quoted string. "
            f"Got {cfg['Optimizer'].get('lr')!r}"
        )
    if "test" in text.lower():
        raise ValueError(
            "Formal development config unexpectedly contains a test reference"
        )

    initialization_checkpoint = validate_initialization(
        cfg,
        args.expected_initialization,
    )
    train_manifests = validate_dataset(
        cfg,
        "Train",
        args.allow_missing_lmdb,
    )
    eval_manifests = validate_dataset(
        cfg,
        "Eval",
        args.allow_missing_lmdb,
    )
    result = {
        "status": "P1_MSR_CONFIG_OK",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "preprocess_protocol": PROTOCOL_ID,
        "sampler_protocol": SAMPLER_PROTOCOL_ID,
        "expected_initialization": args.expected_initialization,
        "initialization_checkpoint": initialization_checkpoint,
        "train_rows": sum(item.get("rows", 0) for item in train_manifests),
        "eval_rows": sum(item.get("rows", 0) for item in eval_manifests),
        "train_lmdb_manifests": train_manifests,
        "eval_lmdb_manifests": eval_manifests,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
