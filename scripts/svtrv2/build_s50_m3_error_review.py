#!/usr/bin/env python3
"""Build an auditable Clean-Dev error review package for frozen S50 M3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_m3_ug_kk_errors import PUNCTUATION_CANDIDATE, indexed_alignment


PROTOCOL_ID = "S50_M3_ERROR_REVIEW_V1"
LANGUAGES = ("zh", "ug", "kk")
PUNCTUATION = PUNCTUATION_CANDIDATE
MANUAL_DECISIONS = (
    "genuine_char_error",
    "punctuation_gt_missing",
    "gt_annotation_error",
    "normalization_issue",
    "duplicate_error",
    "image_or_crop_issue",
    "ambiguous_image",
)
EXPECTED_CONFIG_SHA256 = (
    "43ad1d8f221f17afe41d6aab1fafc9c336d6d9994d0cd4ca22cc3684ee2bc94f"
)
EXPECTED_MACRO_CER = 0.022986871482536465

FIELDS = [
    "sample_id",
    "language",
    "image_path",
    "gt",
    "pred",
    "confidence",
    "ed",
    "sub",
    "del",
    "ins",
    "gt_len",
    "auto_category",
    "operation_signature",
    "manual_decision",
    "corrected_gt",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--m3-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prediction_sources(report: Path) -> dict[str, Path]:
    return {
        "zh": report / "zh/predictions.jsonl",
        "ug": report / "ug_logical/predictions_logical.jsonl",
        "kk": report / "kk/predictions.jsonl",
    }


def strip_punctuation(text: str) -> str:
    return "".join(character for character in text if character not in PUNCTUATION)


def is_duplicate_insertion(prediction: str, index: int | None) -> bool:
    if index is None or not 0 <= index < len(prediction):
        return False
    character = prediction[index]
    return bool(
        (index > 0 and prediction[index - 1] == character)
        or (index + 1 < len(prediction) and prediction[index + 1] == character)
    )


def auto_category(gt: str, pred: str, edits: list[dict[str, Any]]) -> str:
    operations = Counter(item["operation"] for item in edits)
    punctuation_edits = [
        item
        for item in edits
        if item["gt"] in PUNCTUATION or item["pred"] in PUNCTUATION
    ]
    if strip_punctuation(gt) == strip_punctuation(pred):
        return "punctuation_only_candidate"
    if punctuation_edits:
        return "punctuation_plus_other_error"
    if operations["ins"] and not operations["sub"] and not operations["del"]:
        if any(is_duplicate_insertion(pred, item["pred_index"]) for item in edits):
            return "duplicate_insertion_candidate"
        return "insertion_candidate"
    if operations["del"] and not operations["sub"] and not operations["ins"]:
        return "deletion_candidate"
    if operations["sub"] and not operations["del"] and not operations["ins"]:
        return "genuine_substitution_candidate"
    return "mixed_error_candidate"


def load_previous(output: Path) -> dict[str, dict[str, str]]:
    previous: dict[str, dict[str, str]] = {}
    candidates = [output / "all_errors.csv"]
    candidates.extend(output / language / "all_errors.csv" for language in LANGUAGES)
    candidates.extend(output / f"{language}_reviewed_errors.csv" for language in LANGUAGES)
    for path in candidates:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                language = row.get("language", "")
                sample_id = row.get("sample_id", "")
                if language and sample_id:
                    previous[f"{language}:{sample_id}"] = row
    return previous


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def image_href(root: Path, html_path: Path, image_path: str) -> str:
    source = root / "01_data_preparation/real_line_dataset_eval_reviewed" / image_path
    try:
        return os.path.relpath(source, html_path.parent).replace("\\", "/")
    except ValueError:
        return source.resolve().as_uri()


def make_review_html(
    root: Path,
    path: Path,
    rows: list[dict[str, Any]],
    language: str,
) -> None:
    categories = sorted({row["auto_category"] for row in rows})
    decision_options = "".join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>'
        for value in MANUAL_DECISIONS
    )
    category_options = "".join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>'
        for value in categories
    )
    cards = []
    for index, row in enumerate(rows):
        selected = row["manual_decision"]
        options = decision_options.replace(
            f'value="{html.escape(selected)}"',
            f'value="{html.escape(selected)}" selected',
            1,
        ) if selected else decision_options
        cards.append(
            f"""
<article data-index="{index}" data-category="{html.escape(row['auto_category'])}" data-reviewed="{int(bool(selected))}">
  <img src="{html.escape(image_href(root, path, row['image_path']))}" loading="lazy" alt="{html.escape(row['sample_id'])}">
  <div class="meta"><b>{html.escape(row['sample_id'])}</b><span>{language.upper()}</span><span>ED {row['ed']}</span><span>conf {float(row['confidence']):.4f}</span></div>
  <div class="category">{html.escape(row['auto_category'])}</div>
  <div class="text"><b>GT</b> {html.escape(row['gt'])}</div>
  <div class="text"><b>Pred</b> {html.escape(row['pred'])}</div>
  <div class="ops">{html.escape(row['operation_signature'])}</div>
  <label>Decision<select><option value=""></option>{options}</select></label>
  <label>Corrected GT<input class="corrected" value="{html.escape(row['corrected_gt'])}"></label>
  <label>Note<input class="note" value="{html.escape(row['note'])}"></label>
</article>"""
        )
    serialized = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>S50 M3 {language.upper()} error review</title>
<style>
:root{{--ink:#182027;--muted:#66727c;--line:#c8d0d6;--paper:#fff;--bg:#eef1f3;--accent:#155d4f}}
*{{box-sizing:border-box}}body{{margin:0;font:14px Arial,sans-serif;color:var(--ink);background:var(--bg)}}
header{{position:sticky;top:0;z-index:3;display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 16px;background:var(--paper);border-bottom:1px solid var(--line)}}
header b{{font-size:16px}}header .count{{color:var(--muted)}}select,input,button{{height:34px;border:1px solid #9ba8b1;border-radius:4px;background:#fff;padding:0 8px}}button{{cursor:pointer;background:var(--accent);color:#fff;border-color:var(--accent)}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(390px,1fr));gap:10px;padding:12px}}
article{{min-width:0;background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:10px}}
article.reviewed{{border-color:#3f8f78}}article[hidden]{{display:none}}img{{display:block;width:100%;height:112px;object-fit:contain;background:#f7f8f9;border:1px solid #e3e7e9}}
.meta{{display:flex;gap:8px;align-items:center;margin-top:8px}}.meta span,.category{{color:var(--muted)}}.meta b{{margin-right:auto;overflow-wrap:anywhere}}.category{{margin:5px 0}}.text{{font-size:19px;line-height:1.45;overflow-wrap:anywhere}}.text b{{display:inline-block;width:42px;font-size:12px;color:var(--muted)}}.ops{{margin:6px 0;color:#7a3e22;overflow-wrap:anywhere}}
label{{display:block;margin-top:6px;color:var(--muted)}}label select,label input{{display:block;width:100%;margin-top:3px;color:var(--ink)}}
</style></head><body>
<header><b>S50 M3 {language.upper()} Clean Dev errors</b><span class="count" id="count"></span>
<select id="category"><option value="">All categories</option>{category_options}</select>
<select id="progress"><option value="">All rows</option><option value="pending">Pending</option><option value="reviewed">Reviewed</option></select>
<button id="export">Export reviewed CSV</button></header>
<main>{''.join(cards)}</main>
<script>
const rows={serialized};
const cards=[...document.querySelectorAll('article')];
function sync(){{cards.forEach(card=>{{const row=rows[Number(card.dataset.index)];row.manual_decision=card.querySelector('select').value;row.corrected_gt=card.querySelector('.corrected').value;row.note=card.querySelector('.note').value;card.dataset.reviewed=row.manual_decision?'1':'0';card.classList.toggle('reviewed',Boolean(row.manual_decision));}})}}
function filter(){{sync();const category=document.getElementById('category').value;const progress=document.getElementById('progress').value;let visible=0;cards.forEach(card=>{{const reviewed=card.dataset.reviewed==='1';const show=(!category||card.dataset.category===category)&&(!progress||(progress==='reviewed'&&reviewed)||(progress==='pending'&&!reviewed));card.hidden=!show;if(show)visible++;}});document.getElementById('count').textContent=`${{visible}} / ${{rows.length}}`;}}
document.querySelectorAll('select,input').forEach(node=>node.addEventListener('change',filter));
document.getElementById('export').addEventListener('click',()=>{{sync();const fields={json.dumps(FIELDS)};const quote=value=>'"'+String(value??'').replaceAll('"','""')+'"';const csv='\\ufeff'+[fields.map(quote).join(','),...rows.map(row=>fields.map(key=>quote(row[key])).join(','))].join('\\r\\n');const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));link.download='{language}_reviewed_errors.csv';link.click();URL.revokeObjectURL(link.href);}});
filter();
</script></body></html>"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    report = args.m3_report.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metric_path = report / "metrics_macro_summary.json"
    metrics = read_json(metric_path)
    if metrics.get("split") != "dev" or metrics.get("selection_metric") != "clean_target_dev_macro_CER":
        raise ValueError("M3 report is not the frozen Clean-Dev CER-selected report")
    if metrics.get("config_sha256") != EXPECTED_CONFIG_SHA256:
        raise ValueError("M3 config hash differs from the frozen alpha=0.15 configuration")
    if abs(float(metrics.get("macro_cer")) - EXPECTED_MACRO_CER) > 1e-12:
        raise ValueError("M3 Macro CER differs from the frozen alpha=0.15 result")

    sources = prediction_sources(report)
    previous = load_previous(output)
    rows: list[dict[str, Any]] = []
    for language in LANGUAGES:
        for source in read_jsonl(sources[language]):
            image_path = str(source["image"]).replace("\\", "/")
            gt = source["gt_logical_text"] if language == "ug" else source["eval_gt_text"]
            pred = source["pred_logical_text"] if language == "ug" else source["eval_pred_text"]
            edits = [
                item
                for item in indexed_alignment(gt, pred)
                if item["operation"] != "match"
            ]
            if not edits:
                continue
            counts = Counter(item["operation"] for item in edits)
            sample_id = Path(image_path).stem
            old = previous.get(f"{language}:{sample_id}", {})
            decision = old.get("manual_decision", "")
            if decision and decision not in MANUAL_DECISIONS:
                raise ValueError(f"Invalid manual_decision for {language}:{sample_id}: {decision}")
            rows.append(
                {
                    "sample_id": sample_id,
                    "language": language,
                    "image_path": image_path,
                    "gt": gt,
                    "pred": pred,
                    "confidence": float(source.get("score") or 0.0),
                    "ed": len(edits),
                    "sub": counts["sub"],
                    "del": counts["del"],
                    "ins": counts["ins"],
                    "gt_len": len(gt),
                    "auto_category": auto_category(gt, pred, edits),
                    "operation_signature": "; ".join(
                        f"{item['operation']}:{item['gt']}->{item['pred']}" for item in edits
                    ),
                    "manual_decision": decision,
                    "corrected_gt": old.get("corrected_gt", ""),
                    "note": old.get("note", ""),
                }
            )

    rows.sort(key=lambda row: (LANGUAGES.index(row["language"]), row["sample_id"]))
    write_csv(output / "all_errors.csv", rows)

    aliases = {
        "zh": {
            "punctuation_candidates.csv": lambda row: row["auto_category"].startswith("punctuation_"),
        },
        "ug": {"ed1.csv": lambda row: int(row["ed"]) == 1},
        "kk": {
            "duplicate_candidates.csv": lambda row: row["auto_category"] == "duplicate_insertion_candidate",
        },
    }
    for language in LANGUAGES:
        directory = output / language
        subset = [row for row in rows if row["language"] == language]
        write_csv(directory / "all_errors.csv", subset)
        write_csv(output / f"{language}_reviewed_errors.csv", subset)
        for category in sorted({row["auto_category"] for row in subset}):
            write_csv(
                directory / f"{category}.csv",
                [row for row in subset if row["auto_category"] == category],
            )
        for filename, predicate in aliases[language].items():
            write_csv(directory / filename, [row for row in subset if predicate(row)])
        make_review_html(root, directory / "review.html", subset, language)

    patches = []
    invalid_patch_rows = []
    for row in rows:
        if row["manual_decision"] not in {"punctuation_gt_missing", "gt_annotation_error"}:
            continue
        if not row["corrected_gt"] or row["corrected_gt"] == row["gt"]:
            invalid_patch_rows.append(f"{row['language']}:{row['sample_id']}")
            continue
        patches.append(
            {
                "sample_id": row["sample_id"],
                "language": row["language"],
                "image_path": row["image_path"],
                "old_gt": row["gt"],
                "new_gt": row["corrected_gt"],
                "reason": row["manual_decision"],
                "note": row["note"],
            }
        )
    patch_fields = ["sample_id", "language", "image_path", "old_gt", "new_gt", "reason", "note"]
    write_csv(output / "label_patch_v1.csv", patches, patch_fields)

    per_language = {}
    for language in LANGUAGES:
        subset = [row for row in rows if row["language"] == language]
        decisions = Counter(row["manual_decision"] for row in subset if row["manual_decision"])
        per_language[language] = {
            "error_lines": len(subset),
            "reviewed_lines": sum(decisions.values()),
            "manual_decisions": dict(decisions),
            "auto_categories": dict(Counter(row["auto_category"] for row in subset)),
            "punctuation_gt_missing_ratio_among_error_lines": (
                decisions["punctuation_gt_missing"] / len(subset) if subset else None
            ),
            "genuine_char_error_ratio_among_error_lines": (
                decisions["genuine_char_error"] / len(subset) if subset else None
            ),
        }
    review_complete = all(bool(row["manual_decision"]) for row in rows)
    summary = {
        "status": (
            "S50_M3_ERROR_REVIEW_V1_COMPLETE"
            if review_complete and not invalid_patch_rows
            else "S50_M3_ERROR_REVIEW_V1_READY"
        ),
        "protocol_id": PROTOCOL_ID,
        "model": "S50 M3 / SOAR-SVTR",
        "alpha": 0.15,
        "split": "clean_dev",
        "metric_selection": "minimum Clean Dev Macro CER",
        "source_metric_sha256": sha256(metric_path),
        "source_prediction_sha256": {language: sha256(path) for language, path in sources.items()},
        "error_lines": len(rows),
        "reviewed_lines": sum(bool(row["manual_decision"]) for row in rows),
        "review_complete": review_complete,
        "invalid_patch_rows": invalid_patch_rows,
        "label_patch_rows": len(patches),
        "review_csv_sha256": {
            language: sha256(output / f"{language}_reviewed_errors.csv")
            for language in LANGUAGES
        },
        "label_patch_sha256": sha256(output / "label_patch_v1.csv"),
        "original_clean_dev_modified": False,
        "per_language": per_language,
        "test_evaluated": False,
    }
    (output / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
