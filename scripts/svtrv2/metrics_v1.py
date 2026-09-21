#!/usr/bin/env python3
"""Frozen v1 metric definitions for multilingual meme line recognition."""

from __future__ import annotations


METRIC_PROTOCOL_VERSION = "multilingual_meme_line_metrics_v1"


def edit_distance(a, b) -> int:
    """Compute unit-cost Levenshtein distance for character or token sequences."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, item_a in enumerate(a, 1):
        cur = [i]
        for j, item_b in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (item_a != item_b),
                )
            )
        prev = cur
    return prev[-1]


def word_error_counts(gt: str, pred: str) -> tuple[int, int]:
    """Return word-level edit distance and the number of ground-truth words."""
    gt_words = gt.split()
    pred_words = pred.split()
    return edit_distance(gt_words, pred_words), len(gt_words)


def update_metric_bucket(bucket: dict, gt: str, pred: str) -> None:
    """Accumulate the frozen v1 micro CER/WER and macro 1-NED statistics."""
    distance = edit_distance(gt, pred)
    gt_len = len(gt)
    pred_len = len(pred)
    ned_denominator = max(gt_len, pred_len, 1)
    word_distance, gt_word_count = word_error_counts(gt, pred)

    bucket["samples"] += 1
    bucket["chars"] += gt_len
    bucket["pred_chars"] += pred_len
    bucket["edit_distance"] += distance
    bucket["ned_sum"] += 1.0 - distance / ned_denominator
    bucket["line_correct"] += int(gt == pred)
    bucket["word_edit_distance"] += word_distance
    bucket["words"] += gt_word_count


def finalize_metric_bucket(bucket: dict) -> dict:
    """Finalize metrics using the protocol frozen for all subsequent experiments."""
    samples = bucket["samples"]
    chars = bucket["chars"]
    words = bucket["words"]
    return {
        "metric_protocol": METRIC_PROTOCOL_VERSION,
        "samples": samples,
        "chars": chars,
        "pred_chars": bucket["pred_chars"],
        "edit_distance": bucket["edit_distance"],
        "cer": bucket["edit_distance"] / chars if chars else None,
        "one_minus_ned_macro": bucket["ned_sum"] / samples if samples else None,
        "line_accuracy": bucket["line_correct"] / samples if samples else None,
        "word_edit_distance": bucket["word_edit_distance"],
        "wer": bucket["word_edit_distance"] / words if words else None,
    }
