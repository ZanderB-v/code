#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
CONDA_ENV="${CONDA_ENV:-paddleocr_vl15}"
FORMAL_ROOT="${FORMAL_ROOT:-$ROOT/03_synthetic_generation/synthetic_formal_v2}"
SHARD_ID="${SHARD_ID:-shard_0000}"
PARTS_ROOT="${PARTS_ROOT:-$FORMAL_ROOT/${SHARD_ID}_parts}"
MERGED_OUT="${MERGED_OUT:-$FORMAL_ROOT/synthetic_${SHARD_ID}_parallel}"
CHROME_BINARY="${CHROME_BINARY:-/data_home/wudayu/local_chrome/149.0.7827.22/chrome-linux64/chrome}"
CHROME_DRIVER="${CHROME_DRIVER:-/data_home/wudayu/local_chrome/149.0.7827.22/chromedriver-linux64/chromedriver}"
PER_PART="${PER_PART:-5000}"
TEXT_SHARD_COUNT="${TEXT_SHARD_COUNT:-2}"
FORMAL_SHARD_COUNT="${FORMAL_SHARD_COUNT:-5}"
TEXT_RESERVE="${TEXT_RESERVE:-128}"
TEXT_PLAN_SEED="${TEXT_PLAN_SEED:-20260724}"
MAX_TEXT_OCCURRENCES="${MAX_TEXT_OCCURRENCES:-2}"
STYLE_PROFILE="${STYLE_PROFILE:-formal_diverse}"
DIFFICULTY_PROFILE="${DIFFICULTY_PROFILE:-formal_diverse}"
RENDER_RETRIES="${RENDER_RETRIES:-5}"
REVIEW_SAMPLES="${REVIEW_SAMPLES:-160}"
TARGET_HEIGHT="${TARGET_HEIGHT:-64}"
MAX_ASPECT="${MAX_ASPECT:-40}"

if [[ ! "$SHARD_ID" =~ ^shard_([0-9]{4})$ ]]; then
  echo "SHARD_ID must match shard_NNNN, got: $SHARD_ID" >&2
  exit 2
fi
SHARD_ORDINAL=$((10#${BASH_REMATCH[1]}))
if (( FORMAL_SHARD_COUNT < 1 )); then
  echo "FORMAL_SHARD_COUNT must be >= 1, got: $FORMAL_SHARD_COUNT" >&2
  exit 2
fi
if (( SHARD_ORDINAL >= FORMAL_SHARD_COUNT )); then
  echo "SHARD_ID=$SHARD_ID is outside the formal plan [shard_0000, shard_$(printf '%04d' $((FORMAL_SHARD_COUNT - 1)))]" >&2
  exit 2
fi
if (( TEXT_SHARD_COUNT < 1 || PER_PART < 1 || TEXT_RESERVE < 0 )); then
  echo "TEXT_SHARD_COUNT and PER_PART must be >= 1; TEXT_RESERVE must be >= 0" >&2
  exit 2
fi

GLOBAL_TEXT_SHARD_COUNT=$((FORMAL_SHARD_COUNT * TEXT_SHARD_COUNT))
TEXT_CANDIDATE_COUNT=$((PER_PART + TEXT_RESERVE))
EXPECTED_PART_DIRS=$((3 * TEXT_SHARD_COUNT))
EXPECTED_PER_LANG=$((PER_PART * TEXT_SHARD_COUNT))

LANGS=()
PARTS=()
for lang in zh ug kk; do
  for ((part = 0; part < TEXT_SHARD_COUNT; part++)); do
    LANGS+=("$lang")
    PARTS+=("$part")
  done
done

run_worker() {
  local lang="$1"
  local part="$2"
  local lang_index
  case "$lang" in
    zh) lang_index=0 ;;
    ug) lang_index=1 ;;
    kk) lang_index=2 ;;
    *) echo "Unsupported language: $lang" >&2; return 2 ;;
  esac
  local session="synth_${SHARD_ID}_${lang}_part${part}"
  local global_text_shard_index=$((SHARD_ORDINAL * TEXT_SHARD_COUNT + part))
  local start_index=$((global_text_shard_index * PER_PART + 1))
  local worker_index=$((SHARD_ORDINAL * EXPECTED_PART_DIRS + lang_index * TEXT_SHARD_COUNT + part))
  local render_seed=$((20260724 + worker_index * 1009))
  local postprocess_seed=$((20260725 + worker_index * 1009))
  local statistics_seed=$((20260726 + worker_index * 1009))
  local out="$PARTS_ROOT/synthetic_${SHARD_ID}_${lang}_part${part}"
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && SYNTH_WORKER=1 LANG='$lang' PART='$part' START_INDEX='$start_index' OUT='$out' ROOT='$ROOT' CONDA_ENV='$CONDA_ENV' CHROME_BINARY='$CHROME_BINARY' CHROME_DRIVER='$CHROME_DRIVER' PER_PART='$PER_PART' TEXT_SHARD_COUNT='$TEXT_SHARD_COUNT' FORMAL_SHARD_COUNT='$FORMAL_SHARD_COUNT' TEXT_RESERVE='$TEXT_RESERVE' GLOBAL_TEXT_SHARD_INDEX='$global_text_shard_index' GLOBAL_TEXT_SHARD_COUNT='$GLOBAL_TEXT_SHARD_COUNT' TEXT_CANDIDATE_COUNT='$TEXT_CANDIDATE_COUNT' TEXT_PLAN_SEED='$TEXT_PLAN_SEED' MAX_TEXT_OCCURRENCES='$MAX_TEXT_OCCURRENCES' STYLE_PROFILE='$STYLE_PROFILE' DIFFICULTY_PROFILE='$DIFFICULTY_PROFILE' RENDER_SEED='$render_seed' POSTPROCESS_SEED='$postprocess_seed' STATISTICS_SEED='$statistics_seed' RENDER_RETRIES='$RENDER_RETRIES' REVIEW_SAMPLES='$REVIEW_SAMPLES' TARGET_HEIGHT='$TARGET_HEIGHT' MAX_ASPECT='$MAX_ASPECT' SHARD_ID='$SHARD_ID' bash '$ROOT/scripts/synth/run_formal_parallel_shards_tmux.sh'"
  echo "$session"
}

run_watcher() {
  local sessions=("$@")
  local joined
  joined="$(IFS=,; echo "${sessions[*]}")"
  tmux new-session -d -s "synth_${SHARD_ID}_watch_merge" \
    "cd '$ROOT' && SYNTH_WATCHER=1 WORKER_SESSIONS='$joined' ROOT='$ROOT' CONDA_ENV='$CONDA_ENV' PARTS_ROOT='$PARTS_ROOT' MERGED_OUT='$MERGED_OUT' SHARD_ID='$SHARD_ID' PER_PART='$PER_PART' TEXT_SHARD_COUNT='$TEXT_SHARD_COUNT' FORMAL_SHARD_COUNT='$FORMAL_SHARD_COUNT' TEXT_RESERVE='$TEXT_RESERVE' TEXT_PLAN_SEED='$TEXT_PLAN_SEED' MAX_TEXT_OCCURRENCES='$MAX_TEXT_OCCURRENCES' STYLE_PROFILE='$STYLE_PROFILE' DIFFICULTY_PROFILE='$DIFFICULTY_PROFILE' GLOBAL_TEXT_SHARD_COUNT='$GLOBAL_TEXT_SHARD_COUNT' TEXT_CANDIDATE_COUNT='$TEXT_CANDIDATE_COUNT' EXPECTED_PART_DIRS='$EXPECTED_PART_DIRS' EXPECTED_PER_LANG='$EXPECTED_PER_LANG' SHARD_ORDINAL='$SHARD_ORDINAL' bash '$ROOT/scripts/synth/run_formal_parallel_shards_tmux.sh'"
}

if [[ "${SYNTH_WORKER:-0}" == "1" ]]; then
  mkdir -p "$OUT/logs"
  exec > >(tee -a "$OUT/logs/pipeline.log") 2>&1

  echo "===== worker start ====="
  date
  echo "LANG=$LANG PART=$PART START_INDEX=$START_INDEX OUT=$OUT"
  echo "GLOBAL_TEXT_SHARD_INDEX=$GLOBAL_TEXT_SHARD_INDEX/$GLOBAL_TEXT_SHARD_COUNT TEXT_CANDIDATE_COUNT=$TEXT_CANDIDATE_COUNT"
  echo "RENDER_SEED=$RENDER_SEED POSTPROCESS_SEED=$POSTPROCESS_SEED STATISTICS_SEED=$STATISTICS_SEED"

  source /home/wudayu/anaconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"

  python - <<'PY'
import importlib
missing = []
for name in ["selenium", "PIL", "bidi"]:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("Missing required Python packages:\n" + "\n".join(missing))
print("PYTHON_DEPENDENCY_CHECK_OK")
PY

  python "$ROOT/scripts/synth/06_generate_samples.py" \
    --text-pool-dir "$ROOT/02_corpus_preparation/mixed_text_pool_v2" \
    --font-manifest "$ROOT/03_synthetic_generation/font_library/final_check/font_manifest_final.json" \
    --background-manifest "$ROOT/03_synthetic_generation/background_library_v1/manifest.jsonl" \
    --output-dir "$OUT" \
    --per-lang "$PER_PART" \
    --languages "$LANG" \
    --seed "$RENDER_SEED" \
    --shard-id "${SHARD_ID}_${LANG}_part${PART}" \
    --start-index "$START_INDEX" \
    --text-shard-index "$GLOBAL_TEXT_SHARD_INDEX" \
    --text-shard-count "$GLOBAL_TEXT_SHARD_COUNT" \
    --text-shard-mode global_window \
    --text-shard-size "$TEXT_CANDIDATE_COUNT" \
    --text-plan-seed "$TEXT_PLAN_SEED" \
    --max-text-occurrences "$MAX_TEXT_OCCURRENCES" \
    --style-profile "$STYLE_PROFILE" \
    --max-text-len-zh 35 \
    --max-text-len-ugkk 60 \
    --raw-height-min 72 \
    --raw-height-max 128 \
    --max-raw-width 1200 \
    --min-font-size 28 \
    --crop-pad-x 28 \
    --crop-pad-y 14 \
    --render-retries "$RENDER_RETRIES" \
    --review-samples "$REVIEW_SAMPLES" \
    --chrome-binary "$CHROME_BINARY" \
    --driver-path "$CHROME_DRIVER" \
    --no-sandbox \
    --require-python-bidi \
    --overwrite

  python "$ROOT/scripts/synth/07_postprocess_images.py" \
    --dataset-dir "$OUT" \
    --metadata-raw "$OUT/metadata_raw.jsonl" \
    --target-height "$TARGET_HEIGHT" \
    --max-aspect "$MAX_ASPECT" \
    --seed "$POSTPROCESS_SEED" \
    --allow-hard \
    --difficulty-profile "$DIFFICULTY_PROFILE" \
    --overwrite-final

  python "$ROOT/scripts/synth/08_validate_dataset.py" \
    --dataset-dir "$OUT" \
    --metadata "$OUT/metadata.jsonl" \
    --character-dict "$ROOT/04_model_training/character_dict_hz_ug_kk_v1/character_dict.txt" \
    --min-width 8 \
    --min-height 24 \
    --max-aspect "$MAX_ASPECT" \
    --blank-std-threshold 1.2 \
    --copy-failures

  python "$ROOT/scripts/synth/10_dataset_statistics.py" \
    --dataset-dir "$OUT" \
    --metadata "$OUT/metadata.jsonl" \
    --output-dir "$OUT/statistics" \
    --seed "$STATISTICS_SEED" \
    --samples-per-sheet 32

  echo "===== worker done ====="
  date
  exit 0
fi

if [[ "${SYNTH_WATCHER:-0}" == "1" ]]; then
  mkdir -p "$PARTS_ROOT/logs"
  exec > >(tee -a "$PARTS_ROOT/logs/merge.log") 2>&1
  echo "===== watcher start ====="
  date
  echo "WORKER_SESSIONS=$WORKER_SESSIONS"
  IFS=',' read -r -a sessions <<< "$WORKER_SESSIONS"
  while true; do
    alive=0
    for s in "${sessions[@]}"; do
      if tmux has-session -t "$s" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done
    echo "$(date '+%F %T') alive_workers=$alive"
    [[ "$alive" -eq 0 ]] && break
    sleep 60
  done

  source /home/wudayu/anaconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"

  python - <<'PY'
import html
import json
import os
import shutil
from collections import Counter
from pathlib import Path

parts_root = Path(os.environ["PARTS_ROOT"])
merged = Path(os.environ["MERGED_OUT"])
shard_id = os.environ["SHARD_ID"]
if merged.exists():
    shutil.rmtree(merged)
merged.mkdir(parents=True, exist_ok=True)
(merged / "synthetic_raw").mkdir(parents=True, exist_ok=True)
(merged / "synthetic_final").mkdir(parents=True, exist_ok=True)
(merged / "labels").mkdir(parents=True, exist_ok=True)

part_dirs = sorted(p for p in parts_root.glob(f"synthetic_{shard_id}_*_part*") if p.is_dir())
expected_part_dirs = int(os.environ["EXPECTED_PART_DIRS"])
if len(part_dirs) != expected_part_dirs:
    raise SystemExit(
        f"Expected {expected_part_dirs} part dirs, got {len(part_dirs)}: {[str(p) for p in part_dirs]}"
    )

raw_rows = []
final_rows = []
render_failures = []
postprocess_failures = []
validation_failures = []
duplicate_hashes = []
part_summaries = []

def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

for part in part_dirs:
    for sub in ["synthetic_raw", "synthetic_final"]:
        src = part / sub
        if src.exists():
            for file in src.rglob("*"):
                if file.is_file():
                    rel = file.relative_to(src)
                    dst = merged / sub / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        raise SystemExit(f"Duplicate image path in merge: {sub}/{rel}")
                    shutil.copy2(file, dst)
    raw_rows.extend(read_jsonl(part / "metadata_raw.jsonl"))
    final_rows.extend(read_jsonl(part / "metadata.jsonl"))
    render_failures.extend(read_jsonl(part / "render_failures.jsonl"))
    postprocess_failures.extend(read_jsonl(part / "postprocess_failures.jsonl"))
    validation_failures.extend(read_jsonl(part / "validation_failures.jsonl"))
    duplicate_hashes.extend(read_jsonl(part / "duplicate_hashes.jsonl"))
    for name in ["summary_raw.json", "summary_postprocess.json", "validation_summary.json"]:
        p = part / name
        if p.exists():
            part_summaries.append({"part": part.name, "file": name, "summary": json.loads(p.read_text(encoding="utf-8-sig"))})

ids = [row["id"] for row in final_rows]
if len(ids) != len(set(ids)):
    dup = [item for item, count in Counter(ids).items() if count > 1][:20]
    raise SystemExit(f"Duplicate sample IDs: {dup}")

for name, rows in [
    ("metadata_raw.jsonl", raw_rows),
    ("metadata.jsonl", final_rows),
    ("render_failures.jsonl", render_failures),
    ("postprocess_failures.jsonl", postprocess_failures),
    ("validation_failures.jsonl", validation_failures),
    ("duplicate_hashes.jsonl", duplicate_hashes),
]:
    (merged / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")

all_lines = []
for lang in ["zh", "ug", "kk"]:
    lines = [f"{row['image']}\t{row['ctc_text']}" for row in final_rows if row.get("language") == lang]
    (merged / "labels" / f"train_{lang}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    all_lines.extend(lines)
(merged / "labels" / "train_all.txt").write_text("\n".join(all_lines) + ("\n" if all_lines else ""), encoding="utf-8")

sample_rows = []
for lang in ["zh", "ug", "kk"]:
    sample_rows.extend([r for r in final_rows if r.get("language") == lang][:80])
cards = []
for row in sample_rows:
    cards.append(
        "<section class='card'>"
        f"<img src='{html.escape(row['image'])}' alt='{html.escape(row['id'])}'>"
        f"<div><b>{html.escape(row['language'].upper())}</b> {html.escape(row['id'])} | {html.escape(row.get('background_type',''))} | {html.escape(row.get('font_name',''))}</div>"
        f"<pre>{html.escape(row.get('logical_text',''))}</pre>"
        "</section>"
    )
page = """<!doctype html><html><head><meta charset="utf-8"><title>Merged Synthetic Review</title>
<style>body{margin:0;background:#f5f3ee;font-family:Arial,sans-serif;color:#20242a}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px;padding:16px}.card{background:white;border:1px solid #d8d8d0;border-radius:6px;padding:10px}img{width:100%;max-height:180px;object-fit:contain;background:#e9e5dc;border:1px solid #ded9cc}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0 0;font-size:16px}header{position:sticky;top:0;background:#fffdf7;border-bottom:1px solid #ddd5c7;padding:12px 18px}</style></head><body><header><h1>Merged Synthetic Review</h1></header><main class="grid">"""
page += "\n".join(cards) + "\n</main></body></html>\n"
(merged / "review_merged.html").write_text(page, encoding="utf-8")

kk_specific = set("\u04d8\u04d9\u0492\u0493\u049a\u049b\u04a2\u04a3\u04e8\u04e9\u04b0\u04b1\u04ae\u04af\u04ba\u04bb\u0406\u0456")
rows_by_lang = {
    language: [row for row in final_rows if row.get("language") == language]
    for language in ["zh", "ug", "kk"]
}
text_counts_by_lang = {
    language: Counter(
        row.get("logical_text", "")
        for row in rows
    )
    for language, rows in rows_by_lang.items()
}
ug_rows = rows_by_lang["ug"]
kk_rows = rows_by_lang["kk"]
summary = {
    "dataset_dir": str(merged),
    "parts_root": str(parts_root),
    "part_dirs": [str(p) for p in part_dirs],
    "formal_shard_id": shard_id,
    "formal_shard_ordinal": int(os.environ["SHARD_ORDINAL"]),
    "formal_shard_count": int(os.environ["FORMAL_SHARD_COUNT"]),
    "parts_per_language": int(os.environ["TEXT_SHARD_COUNT"]),
    "global_text_shard_count": int(os.environ["GLOBAL_TEXT_SHARD_COUNT"]),
    "text_candidates_per_worker": int(os.environ["TEXT_CANDIDATE_COUNT"]),
    "text_plan_seed": int(os.environ["TEXT_PLAN_SEED"]),
    "max_text_occurrences": int(os.environ["MAX_TEXT_OCCURRENCES"]),
    "style_profile": os.environ["STYLE_PROFILE"],
    "difficulty_profile": os.environ["DIFFICULTY_PROFILE"],
    "unique_texts_per_lang": {
        language: len(counts) for language, counts in text_counts_by_lang.items()
    },
    "max_text_occurrence_per_lang": {
        language: max(counts.values(), default=0) for language, counts in text_counts_by_lang.items()
    },
    "samples": len(final_rows),
    "raw_samples": len(raw_rows),
    "per_lang": dict(Counter(r.get("language") for r in final_rows)),
    "render_failures": len(render_failures),
    "postprocess_failures": len(postprocess_failures),
    "validation_failures": len(validation_failures),
    "duplicate_hashes": len(duplicate_hashes),
    "ug_u2_status": dict(Counter(r.get("ctc_text_u2_status") for r in ug_rows)),
    "ug_ctc_equals_u2_ratio": sum(1 for r in ug_rows if r.get("ctc_text") == r.get("ctc_text_u2")) / max(1, len(ug_rows)),
    "kk_specific_ratio": sum(1 for r in kk_rows if any(ch in kk_specific for ch in r.get("logical_text", ""))) / max(1, len(kk_rows)),
    "background_type_counts": dict(Counter(r.get("background_type") for r in final_rows)),
    "meme_style_counts": dict(Counter(r.get("meme_style") for r in final_rows)),
    "contrast_counts": dict(Counter(r.get("contrast_bucket") for r in final_rows)),
    "style_profile_counts": dict(Counter(r.get("style_profile") for r in final_rows)),
    "difficulty_profile_counts": dict(Counter(r.get("difficulty_profile") for r in final_rows)),
    "difficulty_counts": dict(Counter(r.get("difficulty") for r in final_rows)),
    "part_summaries": part_summaries,
    "outputs": {
        "metadata": str(merged / "metadata.jsonl"),
        "metadata_raw": str(merged / "metadata_raw.jsonl"),
        "labels": str(merged / "labels"),
        "review_merged": str(merged / "review_merged.html"),
    },
}
(merged / "merged_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
check_errors = []
expected_per_lang = int(os.environ["EXPECTED_PER_LANG"])
expected_counts = {language: expected_per_lang for language in ["zh", "ug", "kk"]}
if summary["per_lang"] != expected_counts:
    check_errors.append(f"Unexpected merged per-language counts: expected {expected_counts}")
if summary["unique_texts_per_lang"] != expected_counts:
    check_errors.append(
        f"A formal shard must contain unique texts within each language: "
        f"expected {expected_counts}, got {summary['unique_texts_per_lang']}"
    )
expected_total = expected_per_lang * 3
if summary["style_profile_counts"] != {os.environ["STYLE_PROFILE"]: expected_total}:
    check_errors.append("Not all rows use the requested style profile")
if summary["difficulty_profile_counts"] != {os.environ["DIFFICULTY_PROFILE"]: expected_total}:
    check_errors.append("Not all rows use the requested difficulty profile")
required_styles = {"plain", "outline", "shadow", "outline_shadow", "double_shadow", "glow", "translucent", "caption_box"}
if os.environ["STYLE_PROFILE"] == "formal_diverse" and not required_styles.issubset(summary["meme_style_counts"]):
    check_errors.append(f"Missing formal meme styles: {sorted(required_styles - set(summary['meme_style_counts']))}")
if os.environ["STYLE_PROFILE"] == "formal_diverse" and not {"high", "medium", "low"}.issubset(summary["contrast_counts"]):
    check_errors.append("Formal diverse shard is missing one or more contrast buckets")
if os.environ["DIFFICULTY_PROFILE"] == "formal_diverse" and not {"clear", "medium", "hard"}.issubset(summary["difficulty_counts"]):
    check_errors.append("Formal diverse shard is missing one or more difficulty buckets")
if summary["postprocess_failures"] > 0:
    check_errors.append("Merged dataset has postprocess failures")
if summary["validation_failures"] > 0:
    check_errors.append("Merged dataset has validation failures")
if summary["duplicate_hashes"] > 0:
    check_errors.append("Merged dataset has duplicate image hashes")
if summary["ug_u2_status"] != {"python_bidi": expected_per_lang}:
    check_errors.append("Merged Uyghur labels are not all python-bidi U2 labels")
if summary["ug_ctc_equals_u2_ratio"] != 1.0:
    check_errors.append("Merged Uyghur ctc_text is not fully U2 visual label")
if summary["kk_specific_ratio"] < 0.30:
    check_errors.append("Merged Kazakh specific character coverage below 30%")
(merged / "merge_check_errors.json").write_text(json.dumps(check_errors, ensure_ascii=False, indent=2), encoding="utf-8")
if check_errors:
    print(json.dumps({"merge_check_errors": check_errors}, ensure_ascii=False, indent=2))
PY

  python "$ROOT/scripts/synth/10_dataset_statistics.py" \
    --dataset-dir "$MERGED_OUT" \
    --metadata "$MERGED_OUT/metadata.jsonl" \
    --output-dir "$MERGED_OUT/statistics" \
    --seed "$((20260730 + SHARD_ORDINAL * 1009))" \
    --samples-per-sheet 48

  python - <<'PY'
import json
import os
from pathlib import Path
errors = json.loads((Path(os.environ["MERGED_OUT"]) / "merge_check_errors.json").read_text(encoding="utf-8"))
if errors:
    raise SystemExit("Merge checks failed after statistics generation:\n" + "\n".join(errors))
print("MERGE_CHECKS_OK")
PY

  echo "===== merge done ====="
  date
  echo "merged output: $MERGED_OUT"
  exit 0
fi

echo "formal shard plan:"
echo "  shard=$SHARD_ID ordinal=$SHARD_ORDINAL/$FORMAL_SHARD_COUNT"
echo "  parts_per_language=$TEXT_SHARD_COUNT per_part=$PER_PART reserve=$TEXT_RESERVE"
echo "  global_text_shards=$GLOBAL_TEXT_SHARD_COUNT text_plan_seed=$TEXT_PLAN_SEED"
echo "  style_profile=$STYLE_PROFILE difficulty_profile=$DIFFICULTY_PROFILE"
for i in "${!LANGS[@]}"; do
  lang="${LANGS[$i]}"
  part="${PARTS[$i]}"
  global_index=$((SHARD_ORDINAL * TEXT_SHARD_COUNT + part))
  start_index=$((global_index * PER_PART + 1))
  echo "  $lang part$part: text_slice=$global_index/$GLOBAL_TEXT_SHARD_COUNT ids=$start_index-$((start_index + PER_PART - 1))"
done
if [[ "${SYNTH_PLAN_ONLY:-0}" == "1" ]]; then
  exit 0
fi
if tmux has-session -t "synth_${SHARD_ID}_watch_merge" 2>/dev/null; then
  echo "watcher session already exists: synth_${SHARD_ID}_watch_merge"
  exit 1
fi

mkdir -p "$PARTS_ROOT"
started=()
for i in "${!LANGS[@]}"; do
  lang="${LANGS[$i]}"
  part="${PARTS[$i]}"
  session="synth_${SHARD_ID}_${lang}_part${part}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "worker session already exists: $session"
    exit 1
  fi
done

for i in "${!LANGS[@]}"; do
  started+=("$(run_worker "${LANGS[$i]}" "${PARTS[$i]}")")
done
run_watcher "${started[@]}"

echo "started workers:"
printf '  %s\n' "${started[@]}"
echo "started watcher: synth_${SHARD_ID}_watch_merge"
echo "attach watcher: tmux attach -t synth_${SHARD_ID}_watch_merge"
echo "merged output will be: $MERGED_OUT"
