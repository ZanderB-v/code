#!/usr/bin/env python3
"""Map a Stage-1 RCTC checkpoint into the nested B1 GTCDecoder namespace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not args.replace:
        raise FileExistsError(f"Output exists; use --replace: {output}")

    import torch

    payload = torch.load(source, map_location=torch.device("cpu"))
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint state_dict is not a mapping")
    if any(key.startswith("decoder.ctc_decoder.") for key in state):
        raise ValueError("Input already uses the full-model nested decoder")

    converted = {}
    encoder_count = 0
    decoder_count = 0
    for key, value in state.items():
        if key.startswith("decoder."):
            new_key = "decoder.ctc_decoder." + key[len("decoder.") :]
            decoder_count += 1
        else:
            new_key = key
            if key.startswith("encoder."):
                encoder_count += 1
        if new_key in converted:
            raise ValueError(f"Duplicate mapped checkpoint key: {new_key}")
        converted[new_key] = value
    if encoder_count == 0 or decoder_count == 0:
        raise ValueError(
            f"Unexpected Stage-1 checkpoint: encoder={encoder_count}, "
            f"decoder={decoder_count}"
        )

    if "state_dict" in payload:
        payload = dict(payload)
        payload["state_dict"] = converted
    else:
        payload = converted
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    report = {
        "status": "RCTC_TO_FULL_SVTRV2_CHECKPOINT_OK",
        "input": str(source),
        "input_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "state_tensors": len(converted),
        "encoder_tensors": encoder_count,
        "rctc_tensors_remapped": decoder_count,
        "mapping": "decoder.* -> decoder.ctc_decoder.*",
        "sgm_initialization": "random",
    }
    output.with_suffix(".conversion.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
