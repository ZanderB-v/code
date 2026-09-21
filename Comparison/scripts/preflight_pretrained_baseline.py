#!/usr/bin/env python3
"""Run one real S50 batch through forward, loss, backward and gradient checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pretrained_models import PROJECT_ROOT, build_model_bundle, sha256_file
from formal_baseline_data import formal_implementation_hashes


def read_dictionary(path: Path) -> list[str]:
    characters = path.read_text(encoding="utf-8").splitlines()
    if not characters or any(len(character) != 1 for character in characters):
        raise ValueError("Every dictionary line must contain exactly one character")
    if len(set(characters)) != len(characters):
        raise ValueError("Character dictionary contains duplicates")
    return characters


def select_s50_records(manifest: Path, characters: set[str]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            language = row.get("language")
            if language not in {"zh", "ug", "kk"} or language in selected:
                continue
            text = row.get("ctc_text_u2") if language == "ug" else row.get("ctc_text")
            text = text or row.get("logical_text") or ""
            if 3 <= len(text) <= 20 and all(character in characters for character in text):
                row["preflight_text"] = text
                selected[language] = row
            if len(selected) == 3:
                break
    missing = sorted({"zh", "ug", "kk"} - set(selected))
    if missing:
        raise ValueError(f"Could not select S50 preflight rows for: {missing}")
    return [selected[language] for language in ("zh", "ug", "kk")]


def resolve_image(synthetic_root: Path, row: dict[str, Any]) -> Path:
    candidates = []
    for key in ("image", "formal_image"):
        value = row.get(key)
        if value:
            path = Path(value)
            candidates.append(path if path.is_absolute() else synthetic_root / path)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No image for S50 sample {row.get('id')}; checked: "
        + ", ".join(map(str, candidates))
    )


def image_tensor(path: Path, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    with Image.open(path) as image:
        image = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor.div(127.5).sub(1.0)


def encode_rows(rows: list[dict[str, Any]], characters: list[str]) -> list[list[int]]:
    char_to_id = {character: index + 1 for index, character in enumerate(characters)}
    encoded = []
    for row in rows:
        text = row["preflight_text"]
        unknown = sorted(set(text) - set(char_to_id))
        if unknown:
            raise ValueError(f"Unknown characters in {row['id']}: {unknown}")
        encoded.append([char_to_id[character] for character in text])
    return encoded


def ctc_objective(logits: torch.Tensor, token_rows: list[list[int]]) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"Expected B,T,C logits, got {tuple(logits.shape)}")
    input_lengths = torch.full(
        (logits.shape[0],), logits.shape[1], dtype=torch.long, device=logits.device
    )
    target_lengths = torch.tensor(
        [len(row) for row in token_rows], dtype=torch.long, device=logits.device
    )
    if int(target_lengths.max()) > logits.shape[1]:
        raise ValueError(
            f"CTC target length {int(target_lengths.max())} exceeds T={logits.shape[1]}"
        )
    targets = torch.tensor(
        [token for row in token_rows for token in row],
        dtype=torch.long,
        device=logits.device,
    )
    return F.ctc_loss(
        logits.log_softmax(-1).transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        zero_infinity=True,
    )


def abinet_objective(
    model: torch.nn.Module,
    images: torch.Tensor,
    token_rows: list[list[int]],
) -> tuple[torch.Tensor, list[int]]:
    result = model(images)
    all_alignment, all_language, vision = result
    result_dicts = [vision] + list(all_language) + list(all_alignment)
    output_steps = vision["logits"].shape[1]
    targets = torch.full(
        (len(token_rows), output_steps), -100, dtype=torch.long, device=images.device
    )
    for index, row in enumerate(token_rows):
        row = row[: output_steps - 1]
        if row:
            targets[index, : len(row)] = torch.tensor(row, device=images.device)
        targets[index, len(row)] = 0
    losses = [
        F.cross_entropy(
            item["logits"].reshape(-1, item["logits"].shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )
        for item in result_dicts
    ]
    return sum(losses) / len(losses), list(vision["logits"].shape)


def gradient_summary(model: torch.nn.Module, predicate) -> dict[str, Any]:
    names = []
    finite = True
    nonzero_numel = 0
    total_numel = 0
    for name, parameter in model.named_parameters():
        if not predicate(name):
            continue
        names.append(name)
        if parameter.grad is None:
            finite = False
            continue
        finite = finite and bool(torch.isfinite(parameter.grad).all().item())
        nonzero_numel += int(torch.count_nonzero(parameter.grad).item())
        total_numel += parameter.grad.numel()
    return {
        "parameter_tensors": len(names),
        "finite": finite,
        "nonzero_gradient_numel": nonzero_numel,
        "gradient_numel": total_numel,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("crnn", "svtr", "parseq", "abinet"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "Comparison" / "pretrained_baselines_v1" / "protocol.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Comparison" / "pretrained_baselines_v1" / "outputs",
    )
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    model_spec = protocol["models"][args.model]
    dictionary_entries = read_dictionary(PROJECT_ROOT / protocol["dictionary"])
    if len(dictionary_entries) != int(protocol["expected_dictionary_size"]):
        raise ValueError("Dictionary size changed after the loading audit")
    characters = list(dictionary_entries)
    if protocol.get("use_space_char"):
        characters.append(" ")
    if len(characters) != int(protocol["expected_recognition_symbol_count"]):
        raise ValueError("Runtime recognition-symbol count changed")

    audit_path = args.output_dir / f"{args.model}_loading_audit.json"
    initialization_path = args.output_dir / f"{args.model}_4891_initialization.pth"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise ValueError(f"Loading audit did not pass: {audit_path}")
    protocol_sha256 = sha256_file(args.protocol)
    implementation_sha256 = formal_implementation_hashes()
    if audit.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Loading audit belongs to a stale protocol file")
    if audit.get("implementation_sha256") != implementation_sha256:
        raise ValueError("Loading audit belongs to stale comparison code")

    checkpoint = PROJECT_ROOT / model_spec["checkpoint"]
    bundle = build_model_bundle(
        args.model,
        checkpoint,
        len(characters),
        int(protocol["max_label_length"]),
    )
    initialization = torch.load(initialization_path, map_location="cpu")
    if initialization.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Adapted initialization belongs to a stale protocol")
    if initialization.get("implementation_sha256") != implementation_sha256:
        raise ValueError("Adapted initialization belongs to stale comparison code")
    bundle.model.load_state_dict(initialization["state_dict"], strict=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested but CUDA is unavailable")
    rows = select_s50_records(
        PROJECT_ROOT / protocol["synthetic_manifest"], set(characters)
    )
    synthetic_root = PROJECT_ROOT / protocol["synthetic_root"]
    paths = [resolve_image(synthetic_root, row) for row in rows]
    images = torch.stack([image_tensor(path, bundle.input_size) for path in paths]).to(device)
    token_rows = encode_rows(rows, characters)

    torch.manual_seed(20260815)
    bundle.model.to(device).train()
    bundle.model.zero_grad(set_to_none=True)
    if bundle.objective == "ctc":
        logits = bundle.model(images)
        loss = ctc_objective(logits, token_rows)
        output_shape = list(logits.shape)
    elif bundle.objective == "parseq":
        loss, output_shape = bundle.model.training_loss(images, token_rows)
    elif bundle.objective == "abinet":
        loss, output_shape = abinet_objective(bundle.model, images, token_rows)
    else:
        raise ValueError(bundle.objective)

    if not bool(torch.isfinite(loss).item()):
        raise ValueError(f"Non-finite loss for {args.model}: {loss.item()}")
    loss.backward()
    backbone_gradients = gradient_summary(bundle.model, bundle.backbone_key)
    reset_gradients = gradient_summary(bundle.model, bundle.reset_key)
    errors = []
    for label, summary in (
        ("backbone", backbone_gradients),
        ("reset", reset_gradients),
    ):
        if not summary["finite"]:
            errors.append(f"{label} gradients are missing or non-finite")
        if summary["nonzero_gradient_numel"] == 0:
            errors.append(f"{label} gradients are all zero")

    report = {
        "status": "passed" if not errors else "failed",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "model": args.model,
        "checkpoint_sha256": sha256_file(checkpoint),
        "initialization": str(initialization_path),
        "device": str(device),
        "input_shape": list(images.shape),
        "output_shape": output_shape,
        "loss": float(loss.detach().cpu()),
        "backbone_gradients": backbone_gradients,
        "reset_module_gradients": reset_gradients,
        "runtime_compatibility_notes": bundle.runtime_compatibility_notes,
        "samples": [
            {
                "id": row["id"],
                "language": row["language"],
                "image": str(path),
                "label_field": "ctc_text_u2" if row["language"] == "ug" else "ctc_text",
                "label_length": len(row["preflight_text"]),
            }
            for row, path in zip(rows, paths)
        ],
        "test_accessed": False,
        "errors": errors,
    }
    report_path = args.output_dir / f"{args.model}_single_batch_preflight.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)
    print(f"SINGLE_BATCH_PREFLIGHT_OK model={args.model} report={report_path}")


if __name__ == "__main__":
    main()
