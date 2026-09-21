#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
FORMAL_ROOT="${FORMAL_ROOT:-$ROOT/03_synthetic_generation/synthetic_formal_v2}"
SESSION="${SESSION:-s50_generation_v2}"
TARGET_HEIGHT="${TARGET_HEIGHT:-64}"
REPLACE="${REPLACE:-0}"

if [[ "${1:-}" == "--replace" ]]; then
  REPLACE=1
elif [[ -n "${1:-}" && "${1:-}" != "--worker" ]]; then
  echo "Usage: bash scripts/synth/run_s50_generation_tmux.sh [--replace]" >&2
  exit 2
fi

if [[ "${S50_WORKER:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    if [[ "$REPLACE" == "1" ]]; then
      tmux kill-session -t "$SESSION"
    else
      echo "tmux session already exists: $SESSION" >&2
      exit 1
    fi
  fi
  if [[ -d "$FORMAL_ROOT" && "$REPLACE" != "1" ]]; then
    echo "Formal output exists: $FORMAL_ROOT" >&2
    echo "Use --replace only when intentionally restarting S50." >&2
    exit 1
  fi
  LOG_DIR="$ROOT/03_synthetic_generation/logs/s50_generation_v2_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$LOG_DIR"
  tmux new-session -d -s "$SESSION" \
    "cd '$ROOT' && S50_WORKER=1 REPLACE='$REPLACE' LOG_DIR='$LOG_DIR' ROOT='$ROOT' FORMAL_ROOT='$FORMAL_ROOT' TARGET_HEIGHT='$TARGET_HEIGHT' bash '$ROOT/scripts/synth/run_s50_generation_tmux.sh' --worker"
  echo "S50 generation supervisor started."
  echo "Attach: tmux attach -t $SESSION"
  echo "Logs: $LOG_DIR"
  exit 0
fi

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1

if [[ "$REPLACE" == "1" ]]; then
  rm -rf "$FORMAL_ROOT"
fi
mkdir -p "$FORMAL_ROOT"

for shard_index in 0 1 2 3 4; do
  shard_id="$(printf 'shard_%04d' "$shard_index")"
  watcher="synth_${shard_id}_watch_merge"
  echo "===== starting $shard_id ====="
  export ROOT FORMAL_ROOT
  export SHARD_ID="$shard_id"
  export FORMAL_SHARD_COUNT=5
  export TEXT_SHARD_COUNT=2
  export PER_PART=5000
  export TEXT_RESERVE=128
  export TEXT_PLAN_SEED=20260728
  export MAX_TEXT_OCCURRENCES=2
  export STYLE_PROFILE=formal_diverse
  export DIFFICULTY_PROFILE=formal_diverse
  export RENDER_RETRIES=5
  export REVIEW_SAMPLES=200
  export TARGET_HEIGHT
  export MAX_ASPECT=40

  tmux kill-session -t "$watcher" 2>/dev/null || true
  bash "$ROOT/scripts/synth/run_formal_parallel_shards_tmux.sh"
  while tmux has-session -t "$watcher" 2>/dev/null; do
    sleep 60
  done

  merged="$FORMAL_ROOT/synthetic_${shard_id}_parallel"
  if [[ ! -s "$merged/merged_summary.json" ]]; then
    echo "Missing merged summary for $shard_id" >&2
    exit 1
  fi
  python - "$merged" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
errors = json.loads(
    (root / "merge_check_errors.json").read_text(encoding="utf-8")
)
summary = json.loads(
    (root / "merged_summary.json").read_text(encoding="utf-8")
)
expected = {"zh": 10000, "ug": 10000, "kk": 10000}
if errors:
    raise SystemExit(f"Shard merge checks failed: {errors}")
if summary.get("per_lang") != expected:
    raise SystemExit(
        f"Unexpected shard counts: {summary.get('per_lang')} != {expected}"
    )
print(json.dumps({
    "status": "passed",
    "dataset_dir": str(root),
    "per_lang": summary["per_lang"],
}, ensure_ascii=False))
PY
  echo "===== completed $shard_id ====="
done

echo "===== building nested S10/S25/S50 manifests ====="
source /home/wudayu/anaconda3/etc/profile.d/conda.sh
conda activate paddleocr_vl15
python "$ROOT/scripts/synth/build_s50_nested_subsets.py" \
  --root "$ROOT" \
  --formal-dir "$FORMAL_ROOT" \
  --workers 8 \
  --replace \
  2>&1 | tee "$LOG_DIR/subsets.log"

echo "S50_GENERATION_DONE"
echo "Formal root: $FORMAL_ROOT"
echo "Subset summary: $FORMAL_ROOT/subsets/s50_build_summary.json"
