#!/usr/bin/env python3
"""Generate a local 100-image corruption review pilot."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from build_corruption_protocol import DEFAULT_ROOT, LANGUAGES, read_metadata, safe_name
from straug12_deterministic import (
    CORRUPTIONS,
    IMPLEMENTATION_ID,
    SEVERITIES,
    apply_corruption,
    deterministic_seed,
    file_sha256,
    plan_corruption,
)


PILOT_ID = "corruption_protocol_v1_pilot100_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--target-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def build_slots(count: int) -> list[tuple[str, int, str]]:
    if count < len(CORRUPTIONS):
        raise ValueError(
            f"count must be at least {len(CORRUPTIONS)} to cover every operator"
        )
    base, remainder = divmod(count, len(CORRUPTIONS))
    pairs = [
        (severity, language)
        for severity in SEVERITIES
        for language in LANGUAGES
    ]
    slots = []
    for operator_index, corruption in enumerate(CORRUPTIONS):
        operator_count = base + int(operator_index < remainder)
        rotated = pairs[operator_index % len(pairs) :] + pairs[
            : operator_index % len(pairs)
        ]
        for index in range(operator_count):
            severity, language = rotated[index % len(rotated)]
            slots.append((corruption, severity, language))
    if len(slots) != count:
        raise AssertionError((len(slots), count))
    return slots


def copy_clean(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        shutil.copy2(source, destination)
        return
    with Image.open(source) as opened:
        opened.convert("RGB").save(destination, format="PNG")


def generate_review_html(
    output_dir: Path,
    records: list[dict[str, Any]],
) -> Path:
    review_path = output_dir / "review.html"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["corruption"]].append(record)

    sections = []
    for corruption in CORRUPTIONS:
        cards = []
        for record in grouped[corruption]:
            params = json.dumps(
                record["parameters"],
                ensure_ascii=False,
                sort_keys=True,
            )
            cards.append(
                f"""
<article class="card" data-key="{html.escape(record['pilot_key'])}"
 data-operator="{html.escape(corruption)}">
  <div class="images">
    <figure>
      <figcaption>Clean</figcaption>
      <img src="{html.escape(record['clean_image'])}" loading="lazy">
    </figure>
    <figure>
      <figcaption>Level {record['severity']}</figcaption>
      <img src="{html.escape(record['corrupted_image'])}" loading="lazy">
    </figure>
  </div>
  <div class="details">
    <div><strong>{html.escape(record['language'].upper())}</strong>
      {html.escape(record['sample_id'])} | severity {record['severity']}</div>
    <div class="text" dir="auto">{html.escape(record['logical_text'])}</div>
    <details><summary>parameters</summary><pre>{html.escape(params)}</pre></details>
    <div class="buttons">
      <button data-decision="keep">Keep</button>
      <button data-decision="too_weak">Too weak</button>
      <button data-decision="too_strong">Too strong</button>
      <button data-decision="invalid">Invalid</button>
    </div>
    <input class="note" placeholder="optional note">
  </div>
</article>
"""
            )
        sections.append(
            f'<section id="{html.escape(corruption)}">'
            f"<h2>{html.escape(corruption)}</h2>"
            + "".join(cards)
            + "</section>"
        )

    manifest_json = json.dumps(
        [
            {
                "pilot_key": record["pilot_key"],
                "sample_id": record["sample_id"],
                "language": record["language"],
                "corruption": record["corruption"],
                "severity": record["severity"],
            }
            for record in records
        ],
        ensure_ascii=False,
    ).replace("</", "<\\/")
    navigation = "".join(
        f'<a href="#{name}">{name}</a>' for name in CORRUPTIONS
    )
    document = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Corruption Pilot 100 Review</title>
<style>
* { box-sizing: border-box; }
body { margin: 0; font-family: Arial, sans-serif; background: #f3f4f1; color: #202124; }
header { position: sticky; top: 0; z-index: 5; background: #fff; border-bottom: 1px solid #bbb; padding: 12px 18px; }
h1 { margin: 0 0 8px; font-size: 22px; }
nav { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
nav a { color: #0b57d0; text-decoration: none; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.toolbar button { padding: 7px 12px; }
#summary { font-family: ui-monospace, Consolas, monospace; white-space: pre-wrap; }
section { padding: 18px; border-bottom: 2px solid #aaa; }
h2 { margin: 0 0 12px; }
.card { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(320px, 1fr); gap: 14px; background: #fff; border: 2px solid #c9cbc7; border-radius: 6px; padding: 12px; margin-bottom: 12px; }
.card[data-current="keep"] { border-color: #188038; }
.card[data-current="too_weak"] { border-color: #1a73e8; }
.card[data-current="too_strong"] { border-color: #f9ab00; }
.card[data-current="invalid"] { border-color: #d93025; }
.images { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; min-width: 0; }
figure { margin: 0; min-width: 0; }
figcaption { font-weight: 700; margin-bottom: 4px; }
img { display: block; width: 100%; height: 150px; object-fit: contain; background: #ecebe5; border: 1px solid #bbb; }
.details { min-width: 0; }
.text { font-size: 18px; margin: 8px 0; overflow-wrap: anywhere; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 10px; }
.buttons { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin: 10px 0; }
.buttons button { min-height: 36px; }
.buttons button.active { color: #fff; background: #185abc; }
.note { width: 100%; min-height: 34px; padding: 6px; }
@media (max-width: 850px) {
  .card { grid-template-columns: 1fr; }
  .buttons { grid-template-columns: 1fr 1fr; }
}
</style>
</head>
<body>
<header>
  <h1>Corruption Pilot 100 Review</h1>
  <div>Compare the corrupted image with Clean. Mark whether the operator and severity remain label-preserving and useful.</div>
  <nav>__NAV__</nav>
  <div class="toolbar">
    <button id="export">Export CSV</button>
    <button id="clear">Clear decisions</button>
    <span id="progress"></span>
  </div>
  <div id="summary"></div>
</header>
<main>__SECTIONS__</main>
<script>
const rows = __MANIFEST__;
const storageKey = "corruption_pilot100_v2_label_safe";
let state = JSON.parse(localStorage.getItem(storageKey) || "{}");

function save() {
  localStorage.setItem(storageKey, JSON.stringify(state));
  render();
}

function render() {
  const counts = {};
  let reviewed = 0;
  document.querySelectorAll(".card").forEach(card => {
    const key = card.dataset.key;
    const current = state[key] || {};
    const decision = current.decision || "";
    card.dataset.current = decision;
    card.querySelectorAll("button[data-decision]").forEach(button => {
      button.classList.toggle("active", button.dataset.decision === decision);
    });
    const note = card.querySelector(".note");
    if (document.activeElement !== note) note.value = current.note || "";
    if (decision) {
      reviewed += 1;
      const operator = card.dataset.operator;
      counts[operator] ||= {keep: 0, too_weak: 0, too_strong: 0, invalid: 0};
      counts[operator][decision] += 1;
    }
  });
  document.getElementById("progress").textContent = `reviewed ${reviewed}/${rows.length}`;
  const lines = Object.keys(counts).sort().map(operator => {
    const c = counts[operator];
    return `${operator}: keep=${c.keep} weak=${c.too_weak} strong=${c.too_strong} invalid=${c.invalid}`;
  });
  document.getElementById("summary").textContent = lines.join("\\n");
}

document.querySelectorAll("button[data-decision]").forEach(button => {
  button.addEventListener("click", () => {
    const card = button.closest(".card");
    state[card.dataset.key] ||= {};
    state[card.dataset.key].decision = button.dataset.decision;
    save();
  });
});

document.querySelectorAll(".note").forEach(input => {
  input.addEventListener("change", () => {
    const card = input.closest(".card");
    state[card.dataset.key] ||= {};
    state[card.dataset.key].note = input.value;
    save();
  });
});

document.getElementById("clear").addEventListener("click", () => {
  if (!confirm("Clear all review decisions?")) return;
  state = {};
  save();
});

function csvCell(value) {
  const text = String(value ?? "");
  return `"${text.replaceAll('"', '""')}"`;
}

document.getElementById("export").addEventListener("click", () => {
  const header = ["pilot_key", "sample_id", "language", "corruption", "severity", "decision", "note"];
  const lines = [header.map(csvCell).join(",")];
  rows.forEach(row => {
    const current = state[row.pilot_key] || {};
    lines.push([
      row.pilot_key, row.sample_id, row.language, row.corruption, row.severity,
      current.decision || "", current.note || ""
    ].map(csvCell).join(","));
  });
  const blob = new Blob(["\\ufeff" + lines.join("\\n")], {type: "text/csv;charset=utf-8"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "corruption_pilot100_v2_review.csv";
  link.click();
  URL.revokeObjectURL(link.href);
});

render();
</script>
</body>
</html>
"""
    review_path.write_text(
        document.replace("__NAV__", navigation)
        .replace("__SECTIONS__", "".join(sections))
        .replace("__MANIFEST__", manifest_json),
        encoding="utf-8",
    )
    return review_path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    target_root = (
        args.target_root.resolve()
        if args.target_root
        else root / "01_data_preparation" / "real_line_dataset_eval_reviewed"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "05_evaluation" / PILOT_ID
    )
    if output_dir.exists():
        if not args.replace:
            raise FileExistsError(f"{output_dir} exists; use --replace")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    source_rows = read_metadata(target_root / "metadata.csv", args.split)
    by_language: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        by_language[row["language"]].append(row)
    rng = random.Random(args.seed)
    for language in LANGUAGES:
        rng.shuffle(by_language[language])

    counters = Counter()
    slots = build_slots(args.count)
    records = []
    for index, (corruption, severity, language) in enumerate(slots, 1):
        source_index = counters[language]
        counters[language] += 1
        row = by_language[language][source_index]
        source_path = target_root / row["image"]
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
        sample_seed = deterministic_seed(
            args.seed,
            row["id"],
            corruption,
            severity,
        )
        params = plan_corruption(
            corruption,
            severity,
            source.size,
            sample_seed,
        )
        corrupted = apply_corruption(source, params)
        pilot_key = f"{index:03d}_{corruption}_l{severity}_{language}"
        clean_path = (
            output_dir
            / "assets"
            / "clean"
            / language
            / (safe_name(row["id"]) + source_path.suffix.lower())
        )
        corrupted_path = (
            output_dir
            / "assets"
            / "corrupted"
            / corruption
            / f"level_{severity}"
            / language
            / f"{index:03d}_{safe_name(row['id'])}.png"
        )
        copy_clean(source_path, clean_path)
        corrupted_path.parent.mkdir(parents=True, exist_ok=True)
        corrupted.save(corrupted_path, format="PNG", optimize=False)
        records.append(
            {
                "pilot_id": PILOT_ID,
                "pilot_key": pilot_key,
                "sample_id": row["id"],
                "source_id": row["source_id"],
                "language": language,
                "split": args.split,
                "logical_text": row["logical_text"],
                "source_image": row["image"],
                "source_sha256": file_sha256(source_path),
                "clean_image": clean_path.relative_to(output_dir).as_posix(),
                "corrupted_image": corrupted_path.relative_to(
                    output_dir
                ).as_posix(),
                "corrupted_sha256": file_sha256(corrupted_path),
                "corruption": corruption,
                "severity": severity,
                "sample_seed": sample_seed,
                "parameters": params,
                "decision": "pending",
            }
        )

    manifest_path = output_dir / "pilot_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    csv_path = output_dir / "pilot_manifest.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = (
            "pilot_key",
            "sample_id",
            "language",
            "corruption",
            "severity",
            "logical_text",
            "clean_image",
            "corrupted_image",
            "decision",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})

    operator_counts = Counter(record["corruption"] for record in records)
    language_counts = Counter(record["language"] for record in records)
    severity_counts = Counter(str(record["severity"]) for record in records)
    if max(operator_counts.values()) - min(operator_counts.values()) > 1:
        raise AssertionError(operator_counts)
    if max(language_counts.values()) - min(language_counts.values()) > 1:
        raise AssertionError(language_counts)
    if max(severity_counts.values()) - min(severity_counts.values()) > 1:
        raise AssertionError(severity_counts)
    review_path = generate_review_html(output_dir, records)
    summary = {
        "status": "passed",
        "errors": [],
        "pilot_id": PILOT_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "count": len(records),
        "seed": args.seed,
        "split": args.split,
        "operator_counts": dict(operator_counts),
        "language_counts": dict(language_counts),
        "severity_counts": dict(severity_counts),
        "manifest": str(manifest_path),
        "review": str(review_path),
    }
    (output_dir / "pilot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
