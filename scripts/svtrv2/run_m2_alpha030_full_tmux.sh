#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition
SESSION=m2_alpha030_full_p1
SCRIPT_PATH=$(readlink -f "$0")

if [[ "${1:-}" != "--worker" ]]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  LOG_DIR="$ROOT/04_model_training/logs/m2_alpha030_full_$STAMP"
  mkdir -p "$LOG_DIR"
  printf '%s\n' "$LOG_DIR" > \
    "$ROOT/04_model_training/logs/m2_alpha030_full_latest.txt"
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  FORWARDED_ARGS=""
  if (( $# > 0 )); then
    printf -v FORWARDED_ARGS '%q ' "$@"
  fi
  tmux new-session -d -s "$SESSION" \
    "bash '$SCRIPT_PATH' --worker '$LOG_DIR' $FORWARDED_ARGS"
  tmux set-window-option -t "$SESSION":0 remain-on-exit on
  echo "M2 alpha=0.30 full two-stage run started."
  echo "Attach: tmux attach -t $SESSION"
  echo "Log: $LOG_DIR/full_run.log"
  exit 0
fi

LOG_DIR=$2
shift 2
exec > >(tee -a "$LOG_DIR/full_run.log") 2>&1
PREFLIGHT_ONLY=false
for argument in "$@"; do
  if [[ "$argument" == "--preflight-only" ]]; then
    PREFLIGHT_ONLY=true
  fi
done

source /home/wudayu/anaconda3/etc/profile.d/conda.sh
conda activate openocr_svtrv2
cd "$ROOT"
export PYTHONUNBUFFERED=1

python scripts/svtrv2/preflight_p1_formal.py \
  --root "$ROOT" \
  --minimum-gpus 2 \
  --output "$LOG_DIR/formal_preflight.json"

python scripts/svtrv2/test_dual_order_components.py \
  --root "$ROOT" \
  --output "$LOG_DIR/component_preflight.json"

python scripts/svtrv2/run_dual_order_method_suite.py \
  --root "$ROOT" \
  --log-dir "$LOG_DIR" \
  --methods m2 \
  --m2-consistency-weight 0.30 \
  --summary-name m2_alpha_030_full_clean_dev_summary \
  --ddp-gpus 0,1 \
  --eval-gpu 0 \
  --master-port-base 29960 \
  --max-epoch 50 \
  --synthetic-eval-every 5 \
  --target-eval-every 2 \
  --synthetic-patience 4 \
  --target-patience 5 \
  --synthetic-min-epoch 15 \
  --target-min-epoch 10 \
  "$@"

if [[ "$PREFLIGHT_ONLY" == true ]]; then
  echo "M2_ALPHA_030_PREFLIGHT_ONLY_OK"
else
  echo "M2_ALPHA_030_FULL_TWO_STAGE_OK"
fi
