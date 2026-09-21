#!/usr/bin/env python3
"""Build a frozen Clean Dev error profile for the selected final M3 checkpoint."""

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

from analyze_m3_ug_kk_errors import (
    PUNCTUATION_CANDIDATE,
    duplicate_kind,
    indexed_alignment,
    ug_run_position,
)
from metrics_v1 import finalize_metric_bucket, update_metric_bucket


LANGUAGES = ("zh", "ug", "kk")
SLDR_LOCAL_RATIO_THRESHOLD = 0.40

MANUAL_CATEGORIES = {
    "zh": (
        "similar_han_character",
        "local_stroke_or_component",
        "small_or_blurred",
        "outline_shadow_or_art_font",
        "other_substitution",
        "deletion",
        "insertion",
        "punctuation_or_normalization",
        "ambiguous_or_crop",
        "other",
    ),
    "ug": (
        "local_glyph_confusion",
        "joining_form_confusion",
        "order_related",
        "other_substitution",
        "deletion",
        "insertion",
        "punctuation_or_normalization",
        "ambiguous_or_crop",
        "other",
    ),
    "kk": (
        "glyph_confusion",
        "other_substitution",
        "deletion",
        "insertion",
        "duplicate_error",
        "punctuation_or_normalization",
        "ambiguous_or_crop",
        "other",
    ),
}

FAMILY_MAP = {
    "similar_han_character": "local_glyph_confusion",
    "local_stroke_or_component": "local_glyph_confusion",
    "small_or_blurred": "local_glyph_confusion",
    "outline_shadow_or_art_font": "local_glyph_confusion",
    "local_glyph_confusion": "local_glyph_confusion",
    "glyph_confusion": "local_glyph_confusion",
    "joining_form_confusion": "joining_form",
    "order_related": "order_related",
    "other_substitution": "genuine_substitution",
    "deletion": "genuine_deletion",
    "insertion": "genuine_insertion",
    "duplicate_error": "genuine_insertion",
    "punctuation_or_normalization": "punctuation_or_normalization",
    "ambiguous_or_crop": "ambiguous_or_crop",
    "other": "other",
}

LINE_FIELDS = (
    "review_id",
    "sample_id",
    "language",
    "image_path",
    "gt",
    "pred",
    "confidence",
    "ed",
    "substitutions",
    "deletions",
    "insertions",
    "auto_category",
    "punctuation_edit",
    "adjacent_duplicate_candidate",
    "operation_signature",
    "manual_category",
    "review_note",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("05_evaluation/clean_dev_v2_m3_error_profile_v1"),
    )
    parser.add_argument(
        "--reselection-dir",
        type=Path,
        default=Path("05_evaluation/clean_dev_v2_checkpoint_reselection"),
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("01_data_preparation/clean_dev_v2_verified"),
    )
    parser.add_argument(
        "--frozen-manifest",
        default="frozen_clean_dev_v2_manifest.json",
    )
    parser.add_argument("--protocol-label", default="Clean Dev V2")
    parser.add_argument(
        "--model",
        default="auto",
        help="M3 alpha-grid model, or 'auto' to use final_decisions.json.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Expected alpha when --model is explicit.",
    )
    parser.add_argument(
        "--expected-epoch",
        type=int,
        default=None,
        help="Optional assertion for a pre-frozen checkpoint epoch.",
    )
    parser.add_argument(
        "--previous-review-dir",
        type=Path,
        default=None,
        help="Optional prior review directory; categories are inherited by review_id.",
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_previous(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["review_id"]: row for row in csv.DictReader(handle)}


def strip_punctuation(text: str) -> str:
    return "".join(char for char in text if char not in PUNCTUATION_CANDIDATE)


def auto_category(gt: str, pred: str, edits: list[dict[str, Any]]) -> str:
    counts = Counter(item["operation"] for item in edits)
    punctuation = any(
        item["gt"] in PUNCTUATION_CANDIDATE
        or item["pred"] in PUNCTUATION_CANDIDATE
        for item in edits
    )
    if strip_punctuation(gt) == strip_punctuation(pred):
        return "punctuation_only_candidate"
    if punctuation:
        return "punctuation_plus_other_candidate"
    active = [name for name in ("sub", "del", "ins") if counts[name]]
    if active == ["sub"]:
        return "substitution_candidate"
    if active == ["del"]:
        return "deletion_candidate"
    if active == ["ins"]:
        return "insertion_candidate"
    return "mixed_error_candidate"


def image_href(root: Path, page: Path, image_path: str) -> str:
    source = root / "01_data_preparation/real_line_dataset_eval_reviewed" / image_path
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        return os.path.relpath(source, page.parent).replace("\\", "/")
    except ValueError:
        return source.resolve().as_uri()


def review_page(
    root: Path,
    path: Path,
    language: str,
    rows: list[dict[str, Any]],
    protocol_label: str,
) -> None:
    choices = MANUAL_CATEGORIES[language]
    cards = []
    for index, row in enumerate(rows):
        options = ['<option value=""></option>']
        for choice in choices:
            selected = " selected" if row["manual_category"] == choice else ""
            options.append(
                f'<option value="{html.escape(choice)}"{selected}>'
                f"{html.escape(choice)}</option>"
            )
        cards.append(
            f"""
<article data-index="{index}" data-reviewed="{int(bool(row['manual_category']))}">
  <img src="{html.escape(image_href(root, path, row['image_path']))}" loading="lazy" alt="{html.escape(row['sample_id'])}">
  <div class="meta"><b>{html.escape(row['sample_id'])}</b><span>ED {row['ed']}</span><span>conf {float(row['confidence']):.4f}</span></div>
  <div class="auto">{html.escape(row['auto_category'])}</div>
  <div class="text"><b>GT</b>{html.escape(row['gt'])}</div>
  <div class="text"><b>Pred</b>{html.escape(row['pred'])}</div>
  <div class="ops">{html.escape(row['operation_signature'])}</div>
  <label>Manual category<select class="category">{''.join(options)}</select></label>
  <label>Review note<input class="note" value="{html.escape(row['review_note'])}"></label>
</article>"""
        )
    serialized = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    fields = json.dumps(list(LINE_FIELDS))
    output_name = f"{language}_review.csv"
    page_html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(protocol_label)} M3 {language.upper()} error review</title>
<style>
:root{{--ink:#182027;--muted:#65727b;--line:#c8d0d6;--paper:#fff;--bg:#eef1f3;--accent:#155d4f}}
*{{box-sizing:border-box}}body{{margin:0;font:14px Arial,sans-serif;color:var(--ink);background:var(--bg)}}
header{{position:sticky;top:0;z-index:3;display:flex;gap:10px;align-items:center;padding:10px 16px;background:var(--paper);border-bottom:1px solid var(--line)}}
header b{{font-size:16px}}header span{{color:var(--muted)}}button,select,input{{height:34px;border:1px solid #9ba8b1;border-radius:4px;background:#fff;padding:0 8px}}button{{margin-left:auto;color:#fff;background:var(--accent);border-color:var(--accent);cursor:pointer}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(390px,1fr));gap:10px;padding:12px}}article{{min-width:0;padding:10px;background:var(--paper);border:1px solid var(--line);border-radius:6px}}article.reviewed{{border-color:#3f8f78}}
img{{display:block;width:100%;height:118px;object-fit:contain;background:#f7f8f9;border:1px solid #e3e7e9}}.meta{{display:flex;gap:9px;margin-top:8px}}.meta b{{margin-right:auto;overflow-wrap:anywhere}}.auto{{margin:5px 0;color:var(--muted)}}.text{{font-size:19px;line-height:1.45;overflow-wrap:anywhere}}.text b{{display:inline-block;width:48px;font-size:12px;color:var(--muted)}}.ops{{margin:6px 0;color:#7a3e22;overflow-wrap:anywhere}}label{{display:block;margin-top:6px;color:var(--muted)}}label select,label input{{display:block;width:100%;margin-top:3px;color:var(--ink)}}
</style></head><body>
<header><b>{html.escape(protocol_label)} M3 {language.upper()} errors</b><span id="progress"></span><select id="filter"><option value="all">All rows</option><option value="pending">Pending</option><option value="reviewed">Reviewed</option></select><button id="export">Export CSV</button></header>
<main>{''.join(cards)}</main>
<script>
const rows={serialized};const fields={fields};const cards=[...document.querySelectorAll('article')];
function sync(){{cards.forEach(card=>{{const row=rows[Number(card.dataset.index)];row.manual_category=card.querySelector('.category').value;row.review_note=card.querySelector('.note').value;card.dataset.reviewed=row.manual_category?'1':'0';card.classList.toggle('reviewed',Boolean(row.manual_category));}})}}
function refresh(){{sync();const mode=document.getElementById('filter').value;let visible=0,reviewed=0;cards.forEach(card=>{{const done=card.dataset.reviewed==='1';if(done)reviewed++;const show=mode==='all'||(mode==='reviewed'&&done)||(mode==='pending'&&!done);card.hidden=!show;if(show)visible++;}});document.getElementById('progress').textContent=`${{reviewed}} / ${{rows.length}} reviewed; ${{visible}} shown`;}}
document.querySelectorAll('select,input').forEach(node=>node.addEventListener('change',refresh));
document.getElementById('export').addEventListener('click',()=>{{sync();const q=v=>'"'+String(v??'').replaceAll('"','""')+'"';const csv='\\ufeff'+[fields.map(q).join(','),...rows.map(row=>fields.map(key=>q(row[key])).join(','))].join('\\r\\n');const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));link.download='{output_name}';link.click();URL.revokeObjectURL(link.href);}});refresh();
</script></body></html>"""
    path.write_text(page_html, encoding="utf-8")


def verify_selection(
    root: Path,
    reselection_dir: Path,
    expected_model: str,
    expected_alpha: float | None,
    expected_epoch: int | None,
) -> tuple[dict[str, Any], dict[str, Any], str, float]:
    decisions_path = reselection_dir / "final_decisions.json"
    selection_path = reselection_dir / "selection_summary.csv"
    decisions = read_json(decisions_path)
    alpha = decisions["alpha_selection"]
    if expected_model == "auto":
        expected_model = str(alpha["selected_model"])
        expected_alpha = float(alpha["selected_alpha"])
    if expected_alpha is None:
        raise ValueError("--alpha is required when --model is explicit")
    if alpha["selected_model"] != expected_model or float(alpha["selected_alpha"]) != expected_alpha:
        raise ValueError(
            f"The frozen final M3 alpha selection is not {expected_model} alpha={expected_alpha}"
        )
    with selection_path.open("r", encoding="utf-8-sig", newline="") as handle:
        selected = next(
            row for row in csv.DictReader(handle) if row["model"] == expected_model
        )
    if expected_epoch is not None and int(selected["cer_selected_epoch"]) != expected_epoch:
        raise ValueError(f"The frozen final M3 checkpoint is not epoch {expected_epoch}")
    return {
        "final_decisions": str(decisions_path),
        "final_decisions_sha256": sha256(decisions_path),
        "selection_summary": str(selection_path),
        "selection_summary_sha256": sha256(selection_path),
    }, selected, expected_model, expected_alpha


def summarize_manual(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    families = (
        "genuine_substitution",
        "genuine_deletion",
        "genuine_insertion",
        "local_glyph_confusion",
        "joining_form",
        "order_related",
        "punctuation_or_normalization",
        "ambiguous_or_crop",
        "other",
    )
    table = []
    reviewed = [row for row in rows if row["manual_category"]]
    for family in families:
        item = {"error_category": family}
        for language in LANGUAGES:
            item[language] = sum(
                row["language"] == language
                and FAMILY_MAP.get(row["manual_category"]) == family
                for row in reviewed
            )
        table.append(item)

    zh_ug_genuine = [
        row
        for row in reviewed
        if row["language"] in {"zh", "ug"}
        and FAMILY_MAP.get(row["manual_category"])
        not in {"punctuation_or_normalization", "ambiguous_or_crop"}
    ]
    local_categories = {
        "similar_han_character",
        "local_stroke_or_component",
        "small_or_blurred",
        "outline_shadow_or_art_font",
        "local_glyph_confusion",
        "joining_form_confusion",
    }
    local_rows = [
        row for row in zh_ug_genuine if row["manual_category"] in local_categories
    ]
    family_counts = Counter(
        FAMILY_MAP.get(row["manual_category"], "other") for row in zh_ug_genuine
    )
    local_ratio = len(local_rows) / len(zh_ug_genuine) if zh_ug_genuine else None
    largest_family = family_counts.most_common(1)[0][0] if family_counts else None
    review_complete = len(reviewed) == len(rows)
    sldr_authorized = bool(
        review_complete
        and len(zh_ug_genuine) >= 40
        and local_ratio is not None
        and local_ratio >= SLDR_LOCAL_RATIO_THRESHOLD
        and largest_family in {"local_glyph_confusion", "joining_form"}
    )
    gate = {
        "review_complete": review_complete,
        "reviewed_rows": len(reviewed),
        "total_error_rows": len(rows),
        "zh_ug_genuine_error_rows": len(zh_ug_genuine),
        "zh_ug_local_or_joining_rows": len(local_rows),
        "zh_ug_local_or_joining_ratio": local_ratio,
        "largest_zh_ug_error_family": largest_family,
        "sldr_authorized": sldr_authorized,
        "sldr_rule": (
            "all error lines reviewed; at least 40 genuine ZH+UG errors; "
            "local glyph/joining errors >=40%; local glyph or joining is the "
            "largest error family"
        ),
    }
    return table, gate


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    reselection_dir = args.reselection_dir
    if not reselection_dir.is_absolute():
        reselection_dir = root / reselection_dir
    protocol = args.protocol_dir
    if not protocol.is_absolute():
        protocol = root / protocol
    frozen_path = protocol / args.frozen_manifest
    frozen = read_json(frozen_path)
    language_rows = frozen.get("language_evaluated_counts") or frozen.get("language_counts")
    if language_rows is None:
        raise ValueError("Frozen protocol has no evaluated language counts")
    expected_rows = {language: int(language_rows[language]) for language in LANGUAGES}
    protocol_id = f"{args.protocol_label.upper().replace(' ', '_')}_M3_ERROR_PROFILE_V1"
    output = args.output
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)

    inherited_reviews: dict[str, dict[str, dict[str, str]]] = {
        language: {} for language in LANGUAGES
    }
    if args.previous_review_dir is not None:
        previous_dir = args.previous_review_dir
        if not previous_dir.is_absolute():
            previous_dir = root / previous_dir
        for language in LANGUAGES:
            inherited_reviews[language] = load_previous(
                previous_dir / f"{language}_review.csv"
            )

    provenance, selected, selected_model, selected_alpha = verify_selection(
        root,
        reselection_dir,
        args.model,
        args.alpha,
        args.expected_epoch,
    )
    predictions = args.predictions
    if predictions is None:
        predictions = reselection_dir / "selected_predictions" / selected_model
    elif not predictions.is_absolute():
        predictions = root / predictions
    metrics = {}
    source_hashes = {}
    line_rows: list[dict[str, Any]] = []
    operation_rows: list[dict[str, Any]] = []
    confusion_counts: Counter[tuple[str, str, str]] = Counter()
    ug_positions: Counter[tuple[str, str]] = Counter()
    seen_images: set[str] = set()

    for language in LANGUAGES:
        source_path = predictions / f"{language}.jsonl"
        source_hashes[language] = sha256(source_path)
        source_rows = read_jsonl(source_path)
        if len(source_rows) != expected_rows[language]:
            raise ValueError(
                f"{language}: expected {expected_rows[language]} predictions, "
                f"got {len(source_rows)}"
            )
        bucket: defaultdict[str, float] = defaultdict(float)
        review_path = output / "review" / f"{language}_review.csv"
        previous = {
            **inherited_reviews[language],
            **load_previous(review_path),
        }
        for source in source_rows:
            image = str(source["image"]).replace("\\", "/")
            if image in seen_images:
                raise ValueError(f"Duplicate image across selected predictions: {image}")
            seen_images.add(image)
            gt = str(source["gt_text_v2"])
            pred = str(source["pred_text"])
            update_metric_bucket(bucket, gt, pred)
            alignment = indexed_alignment(gt, pred)
            edits = [item for item in alignment if item["operation"] != "match"]
            if not edits:
                continue
            counts = Counter(item["operation"] for item in edits)
            sample_id = Path(image).stem
            review_id = f"{language}:{sample_id}"
            old = previous.get(review_id, {})
            manual_category = old.get("manual_category", "")
            if manual_category and manual_category not in MANUAL_CATEGORIES[language]:
                raise ValueError(f"Invalid category for {review_id}: {manual_category}")
            punctuation_edit = any(
                item["gt"] in PUNCTUATION_CANDIDATE
                or item["pred"] in PUNCTUATION_CANDIDATE
                for item in edits
            )
            adjacent_duplicate = False
            for item in edits:
                if item["operation"] == "ins":
                    kind = duplicate_kind(pred, int(item["pred_index"]))
                    adjacent_duplicate = adjacent_duplicate or kind != "other_insertion"
                else:
                    kind = "not_insertion"
                if item["operation"] == "sub":
                    confusion_counts[(language, item["gt"], item["pred"])] += 1
                if language == "ug":
                    ug_positions[(ug_run_position(gt, item["gt_index"]), item["operation"])] += 1
                operation_rows.append(
                    {
                        "review_id": review_id,
                        "sample_id": sample_id,
                        "language": language,
                        "operation": item["operation"],
                        "gt_character": item["gt"],
                        "pred_character": item["pred"],
                        "gt_index": item["gt_index"],
                        "pred_index": item["pred_index"],
                        "ug_context_position": (
                            ug_run_position(gt, item["gt_index"])
                            if language == "ug"
                            else "not_applicable"
                        ),
                        "adjacent_duplicate_kind": kind,
                    }
                )
            line_rows.append(
                {
                    "review_id": review_id,
                    "sample_id": sample_id,
                    "language": language,
                    "image_path": image,
                    "gt": gt,
                    "pred": pred,
                    "confidence": float(source.get("score") or 0.0),
                    "ed": len(edits),
                    "substitutions": counts["sub"],
                    "deletions": counts["del"],
                    "insertions": counts["ins"],
                    "auto_category": auto_category(gt, pred, edits),
                    "punctuation_edit": int(punctuation_edit),
                    "adjacent_duplicate_candidate": int(adjacent_duplicate),
                    "operation_signature": "; ".join(
                        f"{item['operation']}:{item['gt']}->{item['pred']}"
                        for item in edits
                    ),
                    "manual_category": manual_category,
                    "review_note": old.get("review_note", ""),
                }
            )
        metrics[language] = finalize_metric_bucket(bucket)

    macro = {
        key: sum(metrics[lang][key] for lang in LANGUAGES) / len(LANGUAGES)
        for key in ("cer", "wer", "one_minus_ned_macro", "line_accuracy")
    }
    expected = {
        "cer": float(selected["macro_cer"]),
        "wer": float(selected["macro_wer"]),
        "one_minus_ned_macro": float(selected["macro_one_minus_ned"]),
        "line_accuracy": float(selected["macro_line_accuracy"]),
    }
    for key, value in expected.items():
        if abs(macro[key] - value) > 1e-12:
            raise ValueError(f"Macro {key} mismatch: {macro[key]} != {value}")

    line_rows.sort(key=lambda row: (LANGUAGES.index(row["language"]), row["sample_id"]))
    operation_rows.sort(
        key=lambda row: (LANGUAGES.index(row["language"]), row["sample_id"], str(row["gt_index"]), str(row["pred_index"]))
    )
    write_csv(output / "all_error_lines.csv", line_rows, list(LINE_FIELDS))
    write_csv(output / "error_operations.csv", operation_rows, list(operation_rows[0]))

    automatic_rows = []
    for language in LANGUAGES:
        subset = [row for row in line_rows if row["language"] == language]
        counts = Counter(row["auto_category"] for row in subset)
        for category in sorted(counts):
            automatic_rows.append(
                {
                    "language": language,
                    "auto_category": category,
                    "error_lines": counts[category],
                    "ratio_among_language_error_lines": counts[category] / len(subset),
                }
            )
        review_path = output / "review" / f"{language}_review.csv"
        write_csv(review_path, subset, list(LINE_FIELDS))
        review_page(
            root,
            output / "review" / f"{language}_review.html",
            language,
            subset,
            args.protocol_label,
        )
    write_csv(
        output / "automatic_error_summary.csv",
        automatic_rows,
        ["language", "auto_category", "error_lines", "ratio_among_language_error_lines"],
    )

    confusion_rows = []
    for language in LANGUAGES:
        pairs = sorted(
            (
                (gt, pred, count)
                for (lang, gt, pred), count in confusion_counts.items()
                if lang == language
            ),
            key=lambda item: (-item[2], item[0], item[1]),
        )
        for rank, (gt, pred, count) in enumerate(pairs[:30], 1):
            confusion_rows.append(
                {
                    "language": language,
                    "rank": rank,
                    "gt_character": gt,
                    "pred_character": pred,
                    "count": count,
                }
            )
    write_csv(
        output / "top30_substitution_pairs.csv",
        confusion_rows,
        ["language", "rank", "gt_character", "pred_character", "count"],
    )
    write_csv(
        output / "ug_context_position_summary.csv",
        [
            {"context_position": position, "operation": operation, "count": count}
            for (position, operation), count in sorted(ug_positions.items())
        ],
        ["context_position", "operation", "count"],
    )

    manual_table, manual_gate = summarize_manual(line_rows)
    write_csv(
        output / "manual_error_category_table.csv",
        manual_table,
        ["error_category", "zh", "ug", "kk"],
    )
    kk_duplicate_candidates = sum(
        row["language"] == "kk" and int(row["adjacent_duplicate_candidate"])
        for row in line_rows
    )
    kk_confirmed_duplicates = sum(
        row["language"] == "kk" and row["manual_category"] == "duplicate_error"
        for row in line_rows
    )
    gate = {
        "protocol_id": protocol_id,
        "status": (
            "MANUAL_REVIEW_COMPLETE"
            if manual_gate["review_complete"]
            else "MANUAL_REVIEW_REQUIRED"
        ),
        "sldr": manual_gate,
        "segment_count_or_duplicate_alignment": {
            "authorized": False,
            "status": "STOPPED_UNLESS_NEW_MANUAL_AND_RAW_PATH_EVIDENCE",
            "kk_adjacent_duplicate_candidates": kk_duplicate_candidates,
            "kk_manual_confirmed_duplicate_errors": kk_confirmed_duplicates,
            "note": (
                "An adjacent duplicate prediction is only an automatic candidate. "
                "This branch remains stopped unless manual review confirms genuine "
                "duplicate errors and a separate CTC raw-path audit confirms "
                "blank-separated repeated peaks."
            ),
        },
        "m3_nococ_allowed_after_profile_review": manual_gate["review_complete"],
        "test_evaluated": False,
    }
    (output / "decision_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    error_counts = Counter(row["language"] for row in line_rows)
    summary = {
        "status": f"{args.protocol_label.upper().replace(' ', '_')}_M3_ERROR_PROFILE_READY",
        "protocol_id": protocol_id,
        "model": "M3 / SOAR-SVTR",
        "alpha": selected_alpha,
        "selected_epoch": int(selected["cer_selected_epoch"]),
        "split": args.protocol_label,
        "rows": expected_rows,
        "error_lines": dict(error_counts),
        "total_error_lines": len(line_rows),
        "metrics": {"macro": macro, "languages": metrics},
        "provenance": {
            **provenance,
            "prediction_sha256": source_hashes,
            "frozen_clean_dev_manifest": str(frozen_path),
            "frozen_clean_dev_manifest_sha256": sha256(frozen_path),
        },
        "manual_review": {
            "reviewed_rows": manual_gate["reviewed_rows"],
            "review_complete": manual_gate["review_complete"],
            "review_directory": str(output / "review"),
        },
        "decision_gate": gate,
        "selected_predictions_modified": False,
        "corrupted_dev_used": False,
        "test_evaluated": False,
    }
    (output / "error_profile_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
