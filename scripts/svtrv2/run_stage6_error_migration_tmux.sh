#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
SESSION="stage6_error_migration_v1"
CONDA_SH="${CONDA_SH:-/home/wudayu/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-openocr_svtrv2}"
LOG_DIR="$ROOT/04_model_training/logs/stage6_error_migration_v1"

if [[ "${1:-}" == "--worker" ]]; then
  shift
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  cd "$ROOT"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONUNBUFFERED=1
  python scripts/svtrv2/run_stage6_error_migration.py \
    --root "$ROOT" \
    --require-checkpoint-files \
    "$@"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  pane_dead="$(tmux display-message -p -t "$SESSION":0.0 '#{pane_dead}')"
  if [[ "$pane_dead" != "1" ]]; then
    echo "Session already running. Attach: tmux attach -t $SESSION"
    exit 0
  fi
  tmux kill-session -t "$SESSION"
fi

mkdir -p "$LOG_DIR"
printf -v worker_command '%q ' bash "$ROOT/scripts/svtrv2/run_stage6_error_migration_tmux.sh" --worker "$@"
tmux new-session -d -s "$SESSION"
tmux set-window-option -t "$SESSION":0 remain-on-exit on
printf -v command 'set -o pipefail; %s 2>&1 | tee -a %q; rc=${PIPESTATUS[0]}; printf "STAGE6_EXIT_CODE=%%s\n" "$rc"; exit "$rc"' \
  "$worker_command" "$LOG_DIR/stage6.log"
printf -v launch 'bash -c %q' "$command"
tmux respawn-pane -k -t "$SESSION":0.0 "$launch"

echo "Stage 6 analysis started."
echo "GPU policy: only physical GPU0 is visible; frozen predictions are reused, so GPU use should remain zero."
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR/stage6.log"
