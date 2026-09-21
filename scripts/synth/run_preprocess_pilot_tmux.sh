#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
PILOT_ROOT="${PILOT_ROOT:-$ROOT/03_synthetic_generation/preprocess_pilot_v1}"

export ROOT
export FORMAL_ROOT="$PILOT_ROOT"
export SHARD_ID="shard_0000"
export FORMAL_SHARD_COUNT=1
export TEXT_SHARD_COUNT=2
export PER_PART=2500
export TEXT_RESERVE=128
export MAX_TEXT_OCCURRENCES=1
export STYLE_PROFILE=formal_diverse
export DIFFICULTY_PROFILE=formal_diverse
export TARGET_HEIGHT=64
export MAX_ASPECT=40
export REVIEW_SAMPLES=200
export RENDER_RETRIES=5
export TEXT_PLAN_SEED=20260728

for session in \
  synth_shard_0000_zh_part0 \
  synth_shard_0000_zh_part1 \
  synth_shard_0000_ug_part0 \
  synth_shard_0000_ug_part1 \
  synth_shard_0000_kk_part0 \
  synth_shard_0000_kk_part1 \
  synth_shard_0000_watch_merge; do
  tmux kill-session -t "$session" 2>/dev/null || true
done

rm -rf "$PILOT_ROOT"
mkdir -p "$PILOT_ROOT"

bash "$ROOT/scripts/synth/run_formal_parallel_shards_tmux.sh"

echo
echo "Preprocessing pilot generation started."
echo "Expected output:"
echo "  $PILOT_ROOT/synthetic_shard_0000_parallel"
echo "Expected counts:"
echo "  zh=5000 ug=5000 kk=5000"
echo "Watch merge:"
echo "  tmux attach -t synth_shard_0000_watch_merge"
