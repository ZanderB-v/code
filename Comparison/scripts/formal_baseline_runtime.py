#!/usr/bin/env python3
"""Shared training, decoding and metric runtime for controlled baselines."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from formal_baseline_data import PROJECT_ROOT, TextRecord, encode_text
from scripts.svtrv2.metrics_v1 import (  # type: ignore
    finalize_metric_bucket,
    update_metric_bucket,
)


def amp_enabled_for_model(model_name: str) -> bool:
    """Return the numerical precision policy for a comparison model.

    The public CRNN recurrent CTC path is not stable in CUDA FP16 after the
    vocabulary is expanded to 4891 classes. Keep CRNN in FP32 while retaining
    AMP for the other baselines. This changes precision only, not the model,
    data, optimizer, or checkpoint-selection protocol.
    """

    return model_name != "crnn"


def empty_metric_bucket() -> dict[str, float]:
    return {
        "samples": 0,
        "chars": 0,
        "pred_chars": 0,
        "edit_distance": 0,
        "ned_sum": 0.0,
        "line_correct": 0,
        "word_edit_distance": 0,
        "words": 0,
    }


def image_tensor(path: Path, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (width, height), Image.Resampling.BICUBIC
        )
        array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1).div(127.5).sub(1.0)


class LineDataset(Dataset):
    def __init__(
        self,
        records: Sequence[TextRecord],
        input_size: tuple[int, int],
        char_to_id: dict[str, int],
    ) -> None:
        self.records = list(records)
        self.input_size = input_size
        self.char_to_id = char_to_id
        self.encoded = [
            encode_text(record.train_text, char_to_id) for record in self.records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "image": image_tensor(record.image, self.input_size),
            "tokens": self.encoded[index],
            "sample_id": record.sample_id,
            "language": record.language,
            "logical_text": record.logical_text,
            "train_text": record.train_text,
            "image_path": str(record.image),
            "corruption": record.corruption,
            "severity": record.severity,
        }


def collate_lines(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([row["image"] for row in rows]),
        "token_rows": [row["tokens"] for row in rows],
        "sample_ids": [row["sample_id"] for row in rows],
        "languages": [row["language"] for row in rows],
        "logical_texts": [row["logical_text"] for row in rows],
        "train_texts": [row["train_text"] for row in rows],
        "image_paths": [row["image_path"] for row in rows],
        "corruptions": [row["corruption"] for row in rows],
        "severities": [row["severity"] for row in rows],
    }


def ctc_objective(logits: torch.Tensor, token_rows: list[list[int]]) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"Expected B,T,C CTC logits, got {tuple(logits.shape)}")
    required = [
        len(row) + sum(a == b for a, b in zip(row, row[1:]))
        for row in token_rows
    ]
    if max(required) > logits.shape[1]:
        raise ValueError(
            f"CTC alignment requires {max(required)} steps but model emits "
            f"T={logits.shape[1]}; zero_infinity must not hide this error"
        )
    target_lengths = torch.tensor(
        [len(row) for row in token_rows], dtype=torch.long, device=logits.device
    )
    input_lengths = torch.full_like(target_lengths, logits.shape[1])
    targets = torch.tensor(
        [token for row in token_rows for token in row],
        dtype=torch.long,
        device=logits.device,
    )
    # Keep CTC log-probabilities and its backward in FP32 even when the model
    # forward is executed under AMP. This avoids half-precision underflow with
    # the 4891-class vocabulary.
    log_probs = logits.float().log_softmax(-1)
    return F.ctc_loss(
        log_probs.transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=False,
    )


def abinet_objective(
    model: torch.nn.Module,
    images: torch.Tensor,
    token_rows: list[list[int]],
    max_label_length: int,
) -> torch.Tensor:
    longest = max(len(row) for row in token_rows)
    if longest > max_label_length:
        raise ValueError(
            f"ABINet label length {longest} exceeds capacity {max_label_length}"
        )
    all_alignment, all_language, vision = model(images)
    outputs = [vision] + list(all_language) + list(all_alignment)
    steps = int(vision["logits"].shape[1])
    targets = torch.full(
        (len(token_rows), steps), -100, dtype=torch.long, device=images.device
    )
    for index, row in enumerate(token_rows):
        if row:
            targets[index, : len(row)] = torch.tensor(
                row, dtype=torch.long, device=images.device
            )
        targets[index, len(row)] = 0
    losses = [
        F.cross_entropy(
            item["logits"].reshape(-1, item["logits"].shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )
        for item in outputs
    ]
    return sum(losses) / len(losses)


def training_objective(
    model_name: str,
    model: torch.nn.Module,
    images: torch.Tensor,
    token_rows: list[list[int]],
    max_label_length: int,
) -> torch.Tensor:
    if model_name in {"crnn", "svtr"}:
        return ctc_objective(model(images), token_rows)
    if model_name == "parseq":
        return model(images, token_rows)
    if model_name == "abinet":
        return abinet_objective(model, images, token_rows, max_label_length)
    raise ValueError(model_name)


def _decode_ctc(logits: torch.Tensor, characters: list[str]) -> list[str]:
    predictions = logits.argmax(-1).detach().cpu().tolist()
    output = []
    for row in predictions:
        text = []
        previous = None
        for token in row:
            if token != 0 and token != previous:
                if not 1 <= token <= len(characters):
                    raise ValueError(f"CTC emitted invalid token ID: {token}")
                text.append(characters[token - 1])
            previous = token
        output.append("".join(text))
    return output


def _decode_eos(logits: torch.Tensor, characters: list[str]) -> list[str]:
    predictions = logits.argmax(-1).detach().cpu().tolist()
    output = []
    for row in predictions:
        text = []
        for token in row:
            if token == 0:
                break
            if not 1 <= token <= len(characters):
                raise ValueError(f"Autoregressive model emitted invalid token ID: {token}")
            text.append(characters[token - 1])
        output.append("".join(text))
    return output


def inference_visual_texts(
    model_name: str,
    model: torch.nn.Module,
    images: torch.Tensor,
    characters: list[str],
) -> list[str]:
    if model_name in {"crnn", "svtr"}:
        return _decode_ctc(model(images), characters)
    if model_name == "parseq":
        return _decode_eos(model.forward_inference(images), characters)
    if model_name == "abinet":
        alignment, _, _ = model(images)
        return _decode_eos(alignment["logits"], characters)
    raise ValueError(model_name)


def visual_to_logical(text: str, language: str) -> str:
    if language != "ug":
        return text
    from bidi.algorithm import get_display

    return get_display(text, base_dir="R")


@torch.inference_mode()
def evaluate_clean_dev(
    model_name: str,
    model: torch.nn.Module,
    loader,
    characters: list[str],
    device: torch.device,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    buckets = defaultdict(empty_metric_bucket)
    condition_buckets = defaultdict(lambda: defaultdict(empty_metric_bucket))
    predictions = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        visual_predictions = inference_visual_texts(
            model_name, model, images, characters
        )
        for index, visual_prediction in enumerate(visual_predictions):
            language = batch["languages"][index]
            logical_prediction = visual_to_logical(visual_prediction, language)
            logical_gt = batch["logical_texts"][index]
            update_metric_bucket(buckets[language], logical_gt, logical_prediction)
            update_metric_bucket(buckets["all"], logical_gt, logical_prediction)
            corruption = batch["corruptions"][index]
            severity = int(batch["severities"][index])
            if corruption:
                condition = f"{corruption}/level_{severity}"
                update_metric_bucket(
                    condition_buckets[condition][language],
                    logical_gt,
                    logical_prediction,
                )
            predictions.append(
                {
                    "id": batch["sample_ids"][index],
                    "language": language,
                    "image": batch["image_paths"][index],
                    "gt_logical_text": logical_gt,
                    "gt_train_visual_text": batch["train_texts"][index],
                    "pred_visual_text": visual_prediction,
                    "pred_logical_text": logical_prediction,
                    "corruption": corruption or None,
                    "severity": severity or None,
                }
            )
    languages = {
        language: finalize_metric_bucket(buckets[language])
        for language in ("zh", "ug", "kk")
    }
    macro_cer = sum(item["cer"] for item in languages.values()) / 3
    result = {
        "selection_metric": "clean_dev_macro_cer",
        "clean_dev_macro_cer": macro_cer,
        "languages": languages,
        "all_micro": finalize_metric_bucket(buckets["all"]),
        "samples": len(predictions),
        "test_evaluated": False,
    }
    if condition_buckets:
        conditions = {}
        for condition in sorted(condition_buckets):
            per_language = {
                language: finalize_metric_bucket(
                    condition_buckets[condition][language]
                )
                for language in ("zh", "ug", "kk")
            }
            conditions[condition] = {
                "macro_cer": sum(
                    item["cer"] for item in per_language.values()
                )
                / 3,
                "languages": per_language,
            }
        result["conditions"] = conditions
        result["mean_corrupted_macro_cer"] = sum(
            item["macro_cer"] for item in conditions.values()
        ) / len(conditions)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as out:
            for row in predictions:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        (output_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result
