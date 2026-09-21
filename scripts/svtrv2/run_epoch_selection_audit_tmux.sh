#!/usr/bin/env bash
set -euo pipefail

ROOT="/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition"
SESSION="epoch_selection_audit_v1"
PYTHON="/home/wudayu/anaconda3/envs/openocr_svtrv2/bin/python"
SCRIPT="$ROOT/scripts/svtrv2/audit_epoch_selection.py"
OUT="$ROOT/05_evaluation/epoch_selection_audit_v1"
LOG_DIR="$ROOT/05_evaluation/epoch_selection_audit_v1/logs/$(date +%Y%m%d_%H%M%S)"

if [[ "${1:-}" == "--replace" ]]; then
  tmux kill-session -t "$SESSION" 2>/dev/null || true
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Already running: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi

mkdir -p "$LOG_DIR"
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && '$PYTHON' '$SCRIPT' --root '$ROOT' 2>&1 | tee '$LOG_DIR/audit.log'"

echo "Started: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR/audit.log"
