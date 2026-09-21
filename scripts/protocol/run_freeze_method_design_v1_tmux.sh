#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition
SESSION=freeze_method_design_v1
SCRIPT_PATH=$(readlink -f "$0")

if [[ "${1:-}" != "--worker" ]]; then
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" \
    "bash '$SCRIPT_PATH' --worker"
  tmux set-window-option -t "$SESSION":0 remain-on-exit on
  echo "Method-design freeze started."
  echo "Attach: tmux attach -t $SESSION"
  echo "Output: $ROOT/00_docs/frozen_method_design_v1"
  exit 0
fi

source /home/wudayu/anaconda3/etc/profile.d/conda.sh
conda activate openocr_svtrv2
cd "$ROOT"
export PYTHONUNBUFFERED=1

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="04_model_training/logs/method_design_freeze_$STAMP"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/freeze.log") 2>&1

python scripts/protocol/freeze_method_design_v1.py \
  --root "$ROOT" \
  --mode freeze

python scripts/protocol/freeze_method_design_v1.py \
  --root "$ROOT" \
  --mode verify

echo "METHOD_DESIGN_FREEZE_AND_VERIFY_OK"
