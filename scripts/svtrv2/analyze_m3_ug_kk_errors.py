#!/usr/bin/env python3
"""Build the frozen M3 ZH/UG/KK error profile and evidence gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metrics_v1 import finalize_metric_bucket, update_metric_bucket


PROTOCOL_ID = "S50_M3_TARGETED_ERROR_DIAGNOSTICS_V1"
LANGUAGES = ("zh", "ug", "kk")
# This is an audit candidate, not an active metric normalization. It becomes
# immutable only after the task-level punctuation policy is approved.
PUNCTUATION_CANDIDATE = frozenset(
    "，。！？；：、（）【】《》“”‘’\"'…-,.!?;:()[]{}<>،؛؟۔"
)
UG_REVIEW_CATEGORIES = {
    "local_visual_detail",
    "joining_or_contextual_form",
    "semantic_or_order",
    "annotation_or_ambiguous",
    "other",
}
ZH_REVIEW_CATEGORIES = {
    "local_glyph_confusion",
    "degraded_visual_detail",
    "punctuation_annotation_mismatch",
    "gt_annotation_error",
    "normalization_issue",
    "ambiguous_image",
    "semantic_or_other",
}
PUNCTUATION_REVIEW_CATEGORIES = {
    "visible_unlabeled_punctuation",
    "model_punctuation_error",
    "gt_punctuation_error",
    "not_punctuation_issue",
    "ambiguous_image",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--m3-report", type=Path, required=True)
    parser.add_argument("--raw-path-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--high-confidence", type=float, default=0.90)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def indexed_alignment(reference: str, hypothesis: str) -> list[dict[str, Any]]:
    n, m = len(reference), len(hypothesis)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], back[i][0] = i, "del"
    for j in range(1, m + 1):
        cost[0][j], back[0][j] = j, "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost[i][j], back[i][j] = cost[i - 1][j - 1], "match"
            else:
                choices = (
                    (cost[i - 1][j - 1] + 1, 0, "sub"),
                    (cost[i - 1][j] + 1, 1, "del"),
                    (cost[i][j - 1] + 1, 2, "ins"),
                )
                cost[i][j], _, back[i][j] = min(choices)
    aligned = []
    i, j = n, m
    while i or j:
        operation = back[i][j]
        if operation in ("match", "sub"):
            aligned.append(
                {
                    "operation": operation,
                    "gt": reference[i - 1],
                    "pred": hypothesis[j - 1],
                    "gt_index": i - 1,
                    "pred_index": j - 1,
                }
            )
            i -= 1
            j -= 1
        elif operation == "del":
            aligned.append(
                {
                    "operation": operation,
                    "gt": reference[i - 1],
                    "pred": "",
                    "gt_index": i - 1,
                    "pred_index": None,
                }
            )
            i -= 1
        elif operation == "ins":
            aligned.append(
                {
                    "operation": operation,
                    "gt": "",
                    "pred": hypothesis[j - 1],
                    "gt_index": None,
                    "pred_index": j - 1,
                }
            )
            j -= 1
        else:
            raise RuntimeError(f"Broken alignment at {i},{j}")
    return list(reversed(aligned))


def is_arabic(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x0600 <= codepoint <= 0x06FF
        or 0x0750 <= codepoint <= 0x077F
        or 0x08A0 <= codepoint <= 0x08FF
    ) and unicodedata.category(character).startswith("L")


def ug_run_position(text: str, index: int | None) -> str:
    if index is None or not 0 <= index < len(text) or not is_arabic(text[index]):
        return "non_arabic_or_insertion"
    start = index
    while start > 0 and is_arabic(text[start - 1]):
        start -= 1
    end = index
    while end + 1 < len(text) and is_arabic(text[end + 1]):
        end += 1
    if start == end:
        return "isolated"
    if index == start:
        return "initial"
    if index == end:
        return "final"
    return "medial"


def duplicate_kind(prediction: str, index: int) -> str:
    character = prediction[index]
    left = index > 0 and prediction[index - 1] == character
    right = index + 1 < len(prediction) and prediction[index + 1] == character
    if left and right:
        return "duplicate_both"
    if left:
        return "duplicate_left"
    if right:
        return "duplicate_right"
    return "other_insertion"


def strip_candidate_punctuation(text: str) -> str:
    return "".join(character for character in text if character not in PUNCTUATION_CANDIDATE)


def blank_separated_evidence(raw: dict[str, Any], pred_index: int, kind: str) -> bool:
    segments = raw["collapsed_segments"]
    if not 0 <= pred_index < len(segments):
        return False
    neighbors = []
    if kind in ("duplicate_left", "duplicate_both") and pred_index > 0:
        neighbors.append(pred_index - 1)
    if kind in ("duplicate_right", "duplicate_both") and pred_index + 1 < len(segments):
        neighbors.append(pred_index + 1)
    for neighbor in neighbors:
        first, second = sorted(
            (segments[pred_index]["raw_run_index"], segments[neighbor]["raw_run_index"])
        )
        gap = raw["runs"][first + 1 : second]
        if gap and any(item["is_blank"] for item in gap):
            return True
    return False


def prediction_sources(report: Path) -> dict[str, Path]:
    return {
        "zh": report / "zh/predictions.jsonl",
        "ug": report / "ug_logical/predictions_logical.jsonl",
        "kk": report / "kk/predictions.jsonl",
    }


def make_review_html(
    root: Path,
    output: Path,
    rows: list[dict[str, Any]],
    categories: set[str],
    title: str,
    download_name: str,
) -> None:
    cards = []
    for row in rows:
        absolute = root / "01_data_preparation/real_line_dataset_eval_reviewed" / row["image"]
        try:
            relative = os.path.relpath(absolute, output.parent).replace("\\", "/")
        except ValueError:
            # A synced review directory may live on another Windows drive.
            relative = absolute.resolve().as_uri()
        options = "".join(
            f'<option value="{html.escape(value)}"'
            f'{" selected" if row.get("manual_category") == value else ""}>'
            f'{html.escape(value)}</option>'
            for value in sorted(categories)
        )
        cards.append(
            f"""
<article data-row="{row['review_index']}">
  <img src="{html.escape(relative)}" loading="lazy">
  <div><b>{html.escape(row['sample_id'])}</b> | score={row['score']}</div>
  <div>GT: <span class="transcript">{html.escape(row['gt'])}</span></div>
  <div>Pred: <span class="transcript">{html.escape(row['prediction'])}</span></div>
  <div>Ops: {html.escape(row['operation_signature'])}</div>
  <label>Decision <select><option value=""></option>{options}</select></label>
  <label>Note <input type="text" value="{html.escape(row.get('review_note', ''))}"></label>
</article>"""
        )
    serialized_rows = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    page = """<!doctype html><meta charset="utf-8"><title>__TITLE__</title>
<style>
body{font:14px Arial,sans-serif;margin:20px;background:#f5f6f7;color:#18202a}
header{position:sticky;top:0;background:white;padding:12px;border-bottom:1px solid #bbb;z-index:2}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:10px;margin-top:12px}
article{background:white;border:1px solid #ccd2d8;padding:10px;border-radius:6px}
img{width:100%;height:100px;object-fit:contain;background:#eee} label{display:block;margin-top:8px}
select,input{width:100%;box-sizing:border-box;padding:5px}.transcript{font-size:20px}
button{padding:7px 12px;margin-left:12px}
</style>
<header><b>__TITLE__</b><button id="export">Export reviewed CSV</button></header>
<main>""" + "\n".join(cards) + "</main>" + f"""
<script>
const rows = {serialized_rows};
document.getElementById('export').addEventListener('click', () => {{
  if (!rows.length) return;
  document.querySelectorAll('article').forEach(card => {{
    const row = rows[Number(card.dataset.row) - 1];
    row.manual_category = card.querySelector('select').value;
    row.review_note = card.querySelector('input').value;
  }});
  const fields = Object.keys(rows[0]);
  const quote = value => '"' + String(value ?? '').replaceAll('"', '""') + '"';
  const csv = '\\ufeff' + [fields.map(quote).join(','), ...rows.map(row => fields.map(key => quote(row[key])).join(','))].join('\\r\\n');
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([csv], {{type:'text/csv;charset=utf-8'}}));
  link.download = {json.dumps(download_name)};
  link.click();
  URL.revokeObjectURL(link.href);
}});
</script>"""
    page = page.replace("__TITLE__", html.escape(title))
    output.write_text(page, encoding="utf-8")


def build_review_rows(
    output: Path,
    line_rows: list[dict[str, Any]],
    language: str,
    filename: str,
    categories: set[str],
) -> tuple[Path, list[dict[str, Any]]]:
    path = output / filename
    previous_rows = {}
    if path.is_file():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            previous_rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    rows = []
    for index, row in enumerate(
        (item for item in line_rows if item["language"] == language), 1
    ):
        previous = previous_rows.get(row["sample_id"], {})
        rows.append(
            {
                "review_index": index,
                **row,
                "manual_category": previous.get("manual_category", ""),
                "review_note": previous.get("review_note", ""),
            }
        )
    for row in rows:
        decision = row["manual_category"]
        if decision and decision not in categories:
            raise ValueError(f"Invalid {language} review category: {decision}")
    return path, rows


def macro_metrics(per_language: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        "macro_cer": sum(per_language[language]["cer"] for language in LANGUAGES) / len(LANGUAGES),
        "macro_wer": sum(per_language[language]["wer"] for language in LANGUAGES) / len(LANGUAGES),
        "macro_one_minus_ned": sum(
            per_language[language]["one_minus_ned_macro"] for language in LANGUAGES
        ) / len(LANGUAGES),
        "macro_line_accuracy": sum(
            per_language[language]["line_accuracy"] for language in LANGUAGES
        ) / len(LANGUAGES),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    report = args.m3_report.resolve()
    raw_dir = args.raw_path_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_paths = prediction_sources(report)
    predictions = {language: read_jsonl(path) for language, path in source_paths.items()}
    raw_paths = {
        language: {
            row["image"]: row
            for row in read_jsonl(raw_dir / f"{language}_ctc_raw_paths.jsonl")
        }
        for language in ("ug", "kk")
    }
    operations = []
    line_rows = []
    duplicate_rows = []
    confusion_counts: Counter[tuple[str, str, str]] = Counter()
    operation_counts: Counter[tuple[str, str]] = Counter()
    position_counts: Counter[tuple[str, str]] = Counter()
    punctuation_counts: Counter[tuple[str, str]] = Counter()
    raw_metric_buckets = {language: defaultdict(float) for language in LANGUAGES}
    normalized_metric_buckets = {language: defaultdict(float) for language in LANGUAGES}

    for language in LANGUAGES:
        for row in predictions[language]:
            image = str(row["image"]).replace("\\", "/")
            gt = row["gt_logical_text"] if language == "ug" else row["eval_gt_text"]
            prediction = row["pred_logical_text"] if language == "ug" else row["eval_pred_text"]
            score = float(row.get("score") or 0.0)
            update_metric_bucket(raw_metric_buckets[language], gt, prediction)
            update_metric_bucket(
                normalized_metric_buckets[language],
                strip_candidate_punctuation(gt),
                strip_candidate_punctuation(prediction),
            )
            alignment = indexed_alignment(gt, prediction)
            edits = [item for item in alignment if item["operation"] != "match"]
            punctuation_edit_count = sum(
                item["gt"] in PUNCTUATION_CANDIDATE
                or item["pred"] in PUNCTUATION_CANDIDATE
                for item in edits
            )
            if edits:
                line_rows.append(
                    {
                        "sample_id": Path(image).stem,
                        "language": language,
                        "image": image,
                        "gt": gt,
                        "prediction": prediction,
                        "score": score,
                        "high_confidence_error": int(score >= args.high_confidence),
                        "edit_distance": len(edits),
                        "punctuation_mismatch_count": punctuation_edit_count,
                        "operation_signature": "; ".join(
                            f"{item['operation']}:{item['gt']}->{item['pred']}" for item in edits
                        ),
                    }
                )
            raw = raw_paths.get(language, {}).get(image)
            if language in ("ug", "kk") and raw is None:
                raise ValueError(f"Missing raw CTC path for {language}:{image}")
            if language == "kk" and raw["decoded_visual_text"] != prediction:
                raise ValueError(f"KK raw path does not reproduce prediction: {image}")
            for item in edits:
                operation = item["operation"]
                operation_counts[(language, operation)] += 1
                confusion_counts[(language, item["gt"] or "<eps>", item["pred"] or "<eps>")] += 1
                position = ug_run_position(gt, item["gt_index"]) if language == "ug" else "not_applicable"
                if language == "ug":
                    position_counts[(position, operation)] += 1
                punctuation_mismatch = bool(
                    item["gt"] in PUNCTUATION_CANDIDATE
                    or item["pred"] in PUNCTUATION_CANDIDATE
                )
                punctuation_counts[(language, "punctuation_mismatch" if punctuation_mismatch else "non_punctuation_edit")] += 1
                if operation == "ins" and item["pred"] in PUNCTUATION_CANDIDATE:
                    punctuation_counts[(language, "punctuation_insertion")] += 1
                public = {
                    "sample_id": Path(image).stem,
                    "language": language,
                    "image": image,
                    "score": score,
                    "gt": gt,
                    "prediction": prediction,
                    "operation": operation,
                    "gt_character": item["gt"],
                    "pred_character": item["pred"],
                    "gt_index": item["gt_index"],
                    "pred_index": item["pred_index"],
                    "ug_arabic_run_position": position,
                    "punctuation_mismatch": int(punctuation_mismatch),
                }
                operations.append(public)
                if operation == "ins":
                    kind = duplicate_kind(prediction, int(item["pred_index"]))
                    raw_supported = (
                        language == "kk"
                        and blank_separated_evidence(raw, int(item["pred_index"]), kind)
                    )
                    duplicate_rows.append(
                        {
                            **public,
                            "insertion_class": kind,
                            "blank_separated_duplicate_peak": int(raw_supported),
                            "gt_contains_adjacent_same_character": int(
                                any(gt[index] == gt[index + 1] for index in range(len(gt) - 1))
                            ),
                        }
                    )

    operation_summary = []
    for language in LANGUAGES:
        for operation in ("sub", "del", "ins"):
            operation_summary.append(
                {
                    "language": language,
                    "operation": operation,
                    "count": operation_counts[(language, operation)],
                }
            )
    confusion_rows = [
        {"language": language, "gt_character": gt, "pred_character": pred, "count": count}
        for (language, gt, pred), count in sorted(
            confusion_counts.items(), key=lambda pair: (-pair[1], pair[0])
        )
    ]
    top_confusion_rows = []
    for language in LANGUAGES:
        candidates = [row for row in confusion_rows if row["language"] == language]
        for rank, row in enumerate(candidates[:20], 1):
            top_confusion_rows.append({"rank": rank, **row})
    zh_substitutions = [
        row
        for row in confusion_rows
        if row["language"] == "zh"
        and row["gt_character"] != "<eps>"
        and row["pred_character"] != "<eps>"
    ][:30]
    zh_top30_rows = [
        {"rank": rank, **row} for rank, row in enumerate(zh_substitutions, 1)
    ]
    ed_bucket_rows = []
    for language in LANGUAGES:
        subset = [row for row in line_rows if row["language"] == language]
        buckets = Counter(
            "ED>=3" if int(row["edit_distance"]) >= 3 else f"ED={row['edit_distance']}"
            for row in subset
        )
        for bucket in ("ED=1", "ED=2", "ED>=3"):
            ed_bucket_rows.append(
                {
                    "language": language,
                    "bucket": bucket,
                    "error_lines": buckets[bucket],
                    "ratio_among_error_lines": buckets[bucket] / len(subset) if subset else None,
                }
            )
    position_rows = [
        {"position": position, "operation": operation, "count": count}
        for (position, operation), count in sorted(position_counts.items())
    ]
    ug_review_path, ug_review_rows = build_review_rows(
        output,
        line_rows,
        "ug",
        "ug_local_detail_review.csv",
        UG_REVIEW_CATEGORIES,
    )
    zh_review_path, zh_review_rows = build_review_rows(
        output,
        line_rows,
        "zh",
        "zh_error_review.csv",
        ZH_REVIEW_CATEGORIES,
    )
    punctuation_review_path = output / "punctuation_mismatch_review.csv"
    previous_punctuation = {}
    if punctuation_review_path.is_file():
        with punctuation_review_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            previous_punctuation = {
                row["review_id"]: row for row in csv.DictReader(handle)
            }
    punctuation_review_rows = []
    for index, row in enumerate(
        (item for item in line_rows if item["punctuation_mismatch_count"]), 1
    ):
        review_id = f"{row['language']}:{row['sample_id']}"
        previous = previous_punctuation.get(review_id, {})
        punctuation_review_rows.append(
            {
                "review_index": index,
                "review_id": review_id,
                **row,
                "manual_category": previous.get("manual_category", ""),
                "review_note": previous.get("review_note", ""),
            }
        )
    for row in punctuation_review_rows:
        decision = row["manual_category"]
        if decision and decision not in PUNCTUATION_REVIEW_CATEGORIES:
            raise ValueError(f"Invalid punctuation review category: {decision}")

    raw_metrics = {
        language: finalize_metric_bucket(raw_metric_buckets[language])
        for language in LANGUAGES
    }
    normalized_metrics = {
        language: finalize_metric_bucket(normalized_metric_buckets[language])
        for language in LANGUAGES
    }
    raw_macro = macro_metrics(raw_metrics)
    normalized_macro = macro_metrics(normalized_metrics)
    punctuation_rows = []
    for language in LANGUAGES:
        total = sum(
            operation_counts[(language, operation)]
            for operation in ("sub", "del", "ins")
        )
        mismatches = punctuation_counts[(language, "punctuation_mismatch")]
        punctuation_rows.append(
            {
                "language": language,
                "all_edit_operations": total,
                "punctuation_mismatch_operations": mismatches,
                "punctuation_mismatch_ratio": mismatches / total if total else None,
                "punctuation_insertions": punctuation_counts[(language, "punctuation_insertion")],
                "non_punctuation_edits": punctuation_counts[(language, "non_punctuation_edit")],
            }
        )
    punctuation_reviewed = [
        row for row in punctuation_review_rows if row["manual_category"]
    ]
    visible_unlabeled = [
        row
        for row in punctuation_reviewed
        if row["manual_category"] == "visible_unlabeled_punctuation"
    ]
    punctuation_audit = {
        "status": (
            "PUNCTUATION_POLICY_DECISION_REQUIRED"
            if len(punctuation_reviewed) == len(punctuation_review_rows)
            else "PUNCTUATION_MANUAL_REVIEW_REQUIRED"
        ),
        "policy_is_active": False,
        "candidate_policy": "remove the exact frozen set from both GT and prediction",
        "candidate_characters": sorted(PUNCTUATION_CANDIDATE),
        "candidate_codepoints": [
            f"U+{ord(character):04X}" for character in sorted(PUNCTUATION_CANDIDATE)
        ],
        "manual_review": {
            "rows": len(punctuation_review_rows),
            "reviewed_rows": len(punctuation_reviewed),
            "visible_unlabeled_punctuation_rows": len(visible_unlabeled),
            "visible_unlabeled_ratio": (
                len(visible_unlabeled) / len(punctuation_reviewed)
                if punctuation_reviewed
                else None
            ),
        },
        "raw_metrics": {"macro": raw_macro, "languages": raw_metrics},
        "candidate_normalized_metrics": {
            "macro": normalized_macro,
            "languages": normalized_metrics,
        },
        "candidate_minus_raw_macro": {
            key: normalized_macro[key] - raw_macro[key] for key in raw_macro
        },
        "decision_options": {
            "exclude_punctuation": (
                "Freeze this exact set, then recompute every model and every split "
                "from raw predictions under the same rule."
            ),
            "full_transcription": (
                "Repair missing GT punctuation before any new training; do not "
                "suppress correct model punctuation."
            ),
        },
        "official_frozen_metrics_changed": False,
        "test_evaluated": False,
    }

    write_csv(output / "error_lines.csv", line_rows, list(line_rows[0]) if line_rows else [])
    write_csv(output / "error_operations.csv", operations, list(operations[0]) if operations else [])
    write_csv(output / "operation_summary.csv", operation_summary, ["language", "operation", "count"])
    write_csv(output / "all_character_confusions.csv", confusion_rows, ["language", "gt_character", "pred_character", "count"])
    write_csv(output / "top20_confusion_pairs.csv", top_confusion_rows, ["rank", "language", "gt_character", "pred_character", "count"])
    write_csv(output / "zh_top30_substitutions.csv", zh_top30_rows, ["rank", "language", "gt_character", "pred_character", "count"])
    write_csv(output / "edit_distance_error_lines.csv", ed_bucket_rows, ["language", "bucket", "error_lines", "ratio_among_error_lines"])
    write_csv(output / "ug_context_position_summary.csv", position_rows, ["position", "operation", "count"])
    write_csv(output / "duplicate_insertion_events.csv", duplicate_rows, list(duplicate_rows[0]) if duplicate_rows else [])
    write_csv(output / "punctuation_mismatch_summary.csv", punctuation_rows, list(punctuation_rows[0]))
    write_csv(ug_review_path, ug_review_rows, list(ug_review_rows[0]) if ug_review_rows else [])
    write_csv(zh_review_path, zh_review_rows, list(zh_review_rows[0]) if zh_review_rows else [])
    write_csv(
        punctuation_review_path,
        punctuation_review_rows,
        list(punctuation_review_rows[0]) if punctuation_review_rows else [],
    )
    make_review_html(
        root,
        output / "ug_local_detail_review.html",
        ug_review_rows,
        UG_REVIEW_CATEGORIES,
        "Uyghur local and joining-form error review",
        "ug_local_detail_review.csv",
    )
    make_review_html(
        root,
        output / "zh_error_review.html",
        zh_review_rows,
        ZH_REVIEW_CATEGORIES,
        "Chinese glyph and punctuation error review",
        "zh_error_review.csv",
    )
    make_review_html(
        root,
        output / "punctuation_mismatch_review.html",
        punctuation_review_rows,
        PUNCTUATION_REVIEW_CATEGORIES,
        "Three-language punctuation mismatch review",
        "punctuation_mismatch_review.csv",
    )
    (output / "punctuation_protocol_candidate_v1.json").write_text(
        json.dumps(punctuation_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    ug_reviewed = [row for row in ug_review_rows if row["manual_category"]]
    ug_local = [
        row for row in ug_reviewed
        if row["manual_category"] in {"local_visual_detail", "joining_or_contextual_form"}
    ]
    ug_review_complete = bool(ug_review_rows) and len(ug_reviewed) == len(ug_review_rows)
    ug_local_ratio = len(ug_local) / len(ug_reviewed) if ug_reviewed else None
    zh_reviewed = [row for row in zh_review_rows if row["manual_category"]]
    zh_local = [
        row for row in zh_reviewed
        if row["manual_category"] in {"local_glyph_confusion", "degraded_visual_detail"}
    ]
    zh_review_complete = bool(zh_review_rows) and len(zh_reviewed) == len(zh_review_rows)
    zh_local_ratio = len(zh_local) / len(zh_reviewed) if zh_reviewed else None
    sldr_authorized = bool(
        ug_review_complete
        and zh_review_complete
        and len(ug_reviewed) >= 20
        and len(zh_reviewed) >= 20
        and ug_local_ratio is not None
        and zh_local_ratio is not None
        and ug_local_ratio >= 0.50
        and zh_local_ratio >= 0.50
    )

    kk_insertions = [row for row in duplicate_rows if row["language"] == "kk"]
    kk_duplicates = [row for row in kk_insertions if row["insertion_class"] != "other_insertion"]
    kk_raw_supported = [row for row in kk_duplicates if row["blank_separated_duplicate_peak"]]
    kk_duplicate_ratio = len(kk_duplicates) / len(kk_insertions) if kk_insertions else None
    raw_support_ratio = len(kk_raw_supported) / len(kk_duplicates) if kk_duplicates else None
    count_authorized = bool(
        len(kk_insertions) >= 10
        and kk_duplicate_ratio is not None
        and kk_duplicate_ratio >= 0.50
        and raw_support_ratio is not None
        and raw_support_ratio >= 0.90
    )

    gate = {
        "protocol_id": PROTOCOL_ID,
        "status": (
            "DIAGNOSTIC_REVIEW_COMPLETE"
            if ug_review_complete and zh_review_complete
            else "ZH_UG_MANUAL_REVIEW_REQUIRED"
        ),
        "punctuation_protocol": {
            "authorized": False,
            "status": punctuation_audit["status"],
            "note": "Candidate metrics are diagnostic and have not replaced frozen metrics.",
        },
        "sldr": {
            "authorized": sldr_authorized,
            "rule": (
                "all ZH and UG error lines reviewed; each n>=20; "
                "ZH local/degraded-detail ratio>=0.50 and "
                "UG local/joining ratio>=0.50"
            ),
            "zh": {
                "review_rows": len(zh_review_rows),
                "reviewed_rows": len(zh_reviewed),
                "local_detail_rows": len(zh_local),
                "local_detail_ratio": zh_local_ratio,
            },
            "ug": {
                "review_rows": len(ug_review_rows),
                "reviewed_rows": len(ug_reviewed),
                "local_or_joining_rows": len(ug_local),
                "local_or_joining_ratio": ug_local_ratio,
            },
        },
        "segment_count_regularization": {
            "authorized": count_authorized,
            "rule": "KK insertions>=10; duplicate/INS>=0.50; raw blank-separated support>=0.90",
            "kk_insertions": len(kk_insertions),
            "kk_duplicate_insertions": len(kk_duplicates),
            "duplicate_ratio": kk_duplicate_ratio,
            "raw_supported_duplicates": len(kk_raw_supported),
            "raw_support_ratio": raw_support_ratio,
        },
        "corrupted_dev_used": False,
        "test_evaluated": False,
    }
    (output / "decision_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "M3_MULTILINGUAL_ERROR_PROFILE_OK",
        "protocol_id": PROTOCOL_ID,
        "split": "clean_dev",
        "high_confidence_threshold": args.high_confidence,
        "error_lines": dict(Counter(row["language"] for row in line_rows)),
        "high_confidence_error_lines": dict(
            Counter(row["language"] for row in line_rows if row["high_confidence_error"])
        ),
        "source_sha256": {
            str(path): sha256(path) for path in source_paths.values()
        },
        "raw_path_summary_sha256": sha256(raw_dir / "raw_path_export_summary.json"),
        "decision_gate": gate,
        "punctuation_audit": punctuation_audit,
        "outputs": {
            "ug_review_csv": str(ug_review_path),
            "ug_review_html": str(output / "ug_local_detail_review.html"),
            "zh_review_csv": str(zh_review_path),
            "zh_review_html": str(output / "zh_error_review.html"),
            "punctuation_candidate": str(output / "punctuation_protocol_candidate_v1.json"),
            "punctuation_review_csv": str(punctuation_review_path),
            "punctuation_review_html": str(output / "punctuation_mismatch_review.html"),
            "duplicate_events": str(output / "duplicate_insertion_events.csv"),
        },
        "test_evaluated": False,
    }
    (output / "error_profile_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
