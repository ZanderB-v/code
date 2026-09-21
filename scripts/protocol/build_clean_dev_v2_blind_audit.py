#!/usr/bin/env python3
"""Build the model-blind, full Clean Dev audit package for Protocol V2."""

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


PROTOCOL_ID = "CLEAN_DEV_V2_MODEL_BLIND_AUDIT_V1"
LANGUAGES = ("zh", "ug", "kk")
DECISIONS = (
    "gt_correct",
    "gt_annotation_error",
    "normalization_issue",
    "ambiguous_image",
    "crop_issue",
)
REVIEW_FIELDS = (
    "review_index",
    "sample_id",
    "language",
    "image_path",
    "current_gt",
    "decision",
    "corrected_gt",
    "note",
    "source_record_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "01_data_preparation/real_line_dataset_eval_reviewed/"
            "dev_reviewed/metadata.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("01_data_preparation/clean_dev_v2_protocol/blind_audit_v1"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(row: dict[str, Any]) -> str:
    payload = {
        "sample_id": row["id"],
        "language": row["language"],
        "image": row["image"],
        "text": row["logical_text"] if row["language"] == "ug" else row["text"],
        "split": row["split"],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_review(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def image_href(root: Path, page: Path, image_path: str) -> str:
    image = root / "01_data_preparation/real_line_dataset_eval_reviewed" / image_path
    try:
        return os.path.relpath(image, page.parent).replace("\\", "/")
    except ValueError:
        return image.resolve().as_uri()


def build_html(
    root: Path,
    page: Path,
    language: str,
    rows: list[dict[str, Any]],
) -> None:
    decision_options = "".join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>'
        for value in DECISIONS
    )
    cards = []
    for index, row in enumerate(rows):
        decision = row["decision"]
        options = decision_options
        if decision:
            options = options.replace(
                f'value="{html.escape(decision)}"',
                f'value="{html.escape(decision)}" selected',
                1,
            )
        cards.append(
            f"""
<article data-index="{index}" data-reviewed="{int(bool(decision))}">
  <a class="image" href="{html.escape(image_href(root, page, row['image_path']))}" target="_blank" rel="noreferrer">
    <img src="{html.escape(image_href(root, page, row['image_path']))}" loading="lazy" alt="{html.escape(row['sample_id'])}">
  </a>
  <div class="identity"><b>{html.escape(row['sample_id'])}</b><span>{language.upper()}</span></div>
  <div class="gt" dir="auto">{html.escape(row['current_gt'])}</div>
  <label>Decision<select><option value=""></option>{options}</select></label>
  <label>Corrected GT<input class="corrected" dir="auto" value="{html.escape(row['corrected_gt'])}"></label>
  <label>Note<input class="note" value="{html.escape(row['note'])}"></label>
</article>"""
        )
    serialized = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    page_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clean Dev V2 blind audit - {language.upper()}</title>
<style>
:root{{--ink:#182225;--muted:#657276;--line:#bec9cc;--paper:#fff;--wash:#edf2f1;--accent:#126354;--warn:#9a4e22}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font-family:"Noto Sans","Microsoft YaHei UI","Segoe UI",sans-serif}}
header{{position:sticky;top:0;z-index:4;display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:11px 16px;background:#fbfcfc;border-bottom:1px solid var(--line)}}
header b{{font-family:Georgia,"Noto Serif",serif;font-size:17px;letter-spacing:0}}header .progress{{color:var(--muted);margin-right:auto}}
select,input,button{{min-height:36px;border:1px solid #93a3a7;border-radius:4px;background:#fff;padding:6px 9px;font:inherit}}button{{cursor:pointer;color:#fff;background:var(--accent);border-color:var(--accent)}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:10px;padding:12px}}
article{{min-width:0;background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:10px}}article.reviewed{{border-left:4px solid var(--accent)}}article.invalid{{border-left:4px solid var(--warn)}}article[hidden]{{display:none}}
.image{{display:block;width:100%;height:126px;background:#f6f8f8;border:1px solid #dfe5e6}}img{{width:100%;height:100%;object-fit:contain}}
.identity{{display:flex;gap:10px;align-items:center;margin-top:8px}}.identity b{{margin-right:auto;overflow-wrap:anywhere}}.identity span{{color:var(--muted)}}
.gt{{min-height:42px;margin:8px 0;padding:8px;border-left:3px solid #78a69b;font-size:20px;line-height:1.35;overflow-wrap:anywhere}}
label{{display:block;margin-top:7px;color:var(--muted);font-size:12px}}label select,label input{{display:block;width:100%;margin-top:3px;color:var(--ink);font-size:14px}}
@media(max-width:520px){{main{{grid-template-columns:1fr;padding:8px}}header{{padding:9px}}.image{{height:110px}}}}
</style></head><body>
<header><b>Clean Dev V2 · {language.upper()}</b><span class="progress" id="progressText"></span>
<select id="view"><option value="pending">Pending</option><option value="all">All</option><option value="reviewed">Reviewed</option><option value="invalid">Invalid</option></select>
<button id="export">Export review CSV</button></header>
<main>{''.join(cards)}</main>
<script>
const rows={serialized};const fields={json.dumps(REVIEW_FIELDS)};const cards=[...document.querySelectorAll('article')];
function sync(){{cards.forEach(card=>{{const row=rows[Number(card.dataset.index)];row.decision=card.querySelector('select').value;row.corrected_gt=card.querySelector('.corrected').value;row.note=card.querySelector('.note').value;const needs=row.decision==='gt_annotation_error';const invalid=needs&&(!row.corrected_gt||row.corrected_gt===row.current_gt);card.dataset.reviewed=row.decision?'1':'0';card.dataset.invalid=invalid?'1':'0';card.classList.toggle('reviewed',Boolean(row.decision)&&!invalid);card.classList.toggle('invalid',invalid);}})}}
function applyView(){{sync();const view=document.getElementById('view').value;let shown=0,reviewed=0,invalid=0;cards.forEach(card=>{{const done=card.dataset.reviewed==='1',bad=card.dataset.invalid==='1';if(done&&!bad)reviewed++;if(bad)invalid++;const visible=view==='all'||(view==='pending'&&(!done||bad))||(view==='reviewed'&&done&&!bad)||(view==='invalid'&&bad);card.hidden=!visible;if(visible)shown++;}});document.getElementById('progressText').textContent=`${{reviewed}} / ${{rows.length}} reviewed · ${{invalid}} invalid · ${{shown}} shown`;}}
document.querySelectorAll('select,input').forEach(node=>node.addEventListener('change',applyView));
document.getElementById('export').addEventListener('click',()=>{{sync();const quote=value=>'"'+String(value??'').replaceAll('"','""')+'"';const csv='\\ufeff'+[fields.map(quote).join(','),...rows.map(row=>fields.map(key=>quote(row[key])).join(','))].join('\\r\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));a.download='{language}_blind_review.csv';a.click();URL.revokeObjectURL(a.href);}});applyView();
</script></body></html>"""
    page.write_text(page_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    metadata_path = args.metadata if args.metadata.is_absolute() else root / args.metadata
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    metadata = read_jsonl(metadata_path)
    if len(metadata) != 951:
        raise ValueError(f"Expected all 951 Clean-Dev rows, got {len(metadata)}")
    ids = [row["id"] for row in metadata]
    if len(ids) != len(set(ids)):
        raise ValueError("Clean Dev contains duplicate sample IDs")
    if any(row.get("split") != "dev" for row in metadata):
        raise ValueError("Non-dev row found in Clean Dev metadata")
    language_counts = Counter(row["language"] for row in metadata)
    if set(language_counts) != set(LANGUAGES):
        raise ValueError(f"Unexpected languages: {language_counts}")

    missing_images = []
    raw_manifest_rows = []
    for row in metadata:
        image_path = str(row["image"]).replace("\\", "/")
        image = root / "01_data_preparation/real_line_dataset_eval_reviewed" / image_path
        if not image.is_file():
            missing_images.append(str(image))
        raw_manifest_rows.append(
            {
                "sample_id": row["id"],
                "language": row["language"],
                "image_path": image_path,
                "raw_gt": row["logical_text"] if row["language"] == "ug" else row["text"],
                "split": "dev",
                "source_record_sha256": record_sha256(row),
            }
        )
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} Clean-Dev images")

    raw_manifest_path = output / "clean_dev_v1_raw_manifest.jsonl"
    with raw_manifest_path.open("w", encoding="utf-8") as handle:
        for row in raw_manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    review_rows: list[dict[str, Any]] = []
    for language in LANGUAGES:
        review_path = output / f"{language}_blind_review.csv"
        previous = read_review(review_path)
        language_rows = [row for row in raw_manifest_rows if row["language"] == language]
        current_rows = []
        for index, row in enumerate(language_rows, 1):
            old = previous.get(row["sample_id"], {})
            decision = old.get("decision", "")
            if decision and decision not in DECISIONS:
                raise ValueError(f"Invalid decision for {row['sample_id']}: {decision}")
            current_rows.append(
                {
                    "review_index": index,
                    "sample_id": row["sample_id"],
                    "language": language,
                    "image_path": row["image_path"],
                    "current_gt": row["raw_gt"],
                    "decision": decision,
                    "corrected_gt": old.get("corrected_gt", ""),
                    "note": old.get("note", ""),
                    "source_record_sha256": row["source_record_sha256"],
                }
            )
        write_csv(review_path, current_rows, REVIEW_FIELDS)
        build_html(root, output / f"{language}_blind_review.html", language, current_rows)
        review_rows.extend(current_rows)

    invalid_rows = []
    patch_rows = []
    for row in review_rows:
        decision = row["decision"]
        corrected = row["corrected_gt"]
        if decision == "gt_annotation_error":
            if not corrected or corrected == row["current_gt"]:
                invalid_rows.append(f"{row['language']}:{row['sample_id']}")
            else:
                patch_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "language": row["language"],
                        "image_path": row["image_path"],
                        "old_gt": row["current_gt"],
                        "new_gt": corrected,
                        "reason": decision,
                        "note": row["note"],
                    }
                )
        elif decision and corrected and corrected != row["current_gt"]:
            invalid_rows.append(f"{row['language']}:{row['sample_id']}")

    patch_fields = (
        "sample_id",
        "language",
        "image_path",
        "old_gt",
        "new_gt",
        "reason",
        "note",
    )
    patch_path = output / "label_patch_v2_candidate.csv"
    write_csv(patch_path, patch_rows, patch_fields)

    reviewed = [row for row in review_rows if row["decision"]]
    complete = len(reviewed) == len(review_rows) and not invalid_rows
    per_language = {}
    for language in LANGUAGES:
        subset = [row for row in review_rows if row["language"] == language]
        decisions = Counter(row["decision"] for row in subset if row["decision"])
        per_language[language] = {
            "rows": len(subset),
            "reviewed": sum(decisions.values()),
            "decisions": dict(decisions),
        }
    summary = {
        "status": (
            "CLEAN_DEV_V2_BLIND_AUDIT_COMPLETE"
            if complete
            else "CLEAN_DEV_V2_BLIND_AUDIT_REVIEW_REQUIRED"
        ),
        "protocol_id": PROTOCOL_ID,
        "model_blind": True,
        "model_predictions_loaded": False,
        "source_metadata": str(metadata_path),
        "source_metadata_sha256": sha256(metadata_path),
        "raw_manifest_sha256": sha256(raw_manifest_path),
        "rows": len(review_rows),
        "language_counts": dict(language_counts),
        "reviewed_rows": len(reviewed),
        "pending_rows": len(review_rows) - len(reviewed),
        "invalid_rows": invalid_rows,
        "candidate_patch_rows": len(patch_rows),
        "candidate_patch_sha256": sha256(patch_path),
        "per_language": per_language,
        "clean_dev_v2_generated": False,
        "normalization_v2_frozen": False,
        "ambiguous_or_crop_rows_excluded": False,
        "formal_model_development_allowed": False,
        "test_evaluated": False,
    }
    (output / "blind_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
