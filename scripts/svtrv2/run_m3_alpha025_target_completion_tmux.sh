#!/usr/bin/env bash
set -euo pipefail

REPLACE_TARGET=0
if [[ "${1:-}" == "--replace" ]]; then
  REPLACE_TARGET=1
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--replace]" >&2
  exit 2
fi

if [[ -z "${ROOT:-}" ]]; then
  for candidate in \
    /s1home/turdy_stu/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition \
    /data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition \
    /home/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition; do
    if [[ -d "$candidate" ]]; then ROOT="$candidate"; break; fi
  done
fi
if [[ -z "${PYTHON:-}" ]]; then
  for candidate in \
    /s1home/turdy_stu/wudayu/envs/openocr_svtrv2/bin/python \
    /home/wudayu/anaconda3/envs/openocr_svtrv2/bin/python; do
    if [[ -x "$candidate" ]]; then PYTHON="$candidate"; break; fi
  done
fi
: "${ROOT:?Set ROOT to the project root}"
: "${PYTHON:?Set PYTHON to the openocr_svtrv2 Python executable}"

SESSION="m3_alpha025_target_completion"
# This completion is intentionally locked to physical GPU0. Its per-card batch
# is derived as 32 so the original 2 x 16 effective batch remains unchanged.
DDP_GPUS="0"
EVAL_GPU="0"
LOG_DIR="$ROOT/04_model_training/logs/m3_alpha025_target_completion_v1"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  DEAD="$(tmux display-message -p -t "$SESSION":0.0 '#{pane_dead}')"
  if [[ "$DEAD" == "1" ]]; then
    tmux kill-session -t "$SESSION"
  else
    echo "Session already running: tmux attach -t $SESSION"
    exit 0
  fi
fi

tmux new-session -d -s "$SESSION"
tmux set-window-option -t "$SESSION":0 remain-on-exit on
EXTRA_ARGS=""
if [[ "$REPLACE_TARGET" == "1" ]]; then
  EXTRA_ARGS="--replace-target"
fi
printf -v COMMAND \
  "set -o pipefail; cd %q && NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 %q scripts/svtrv2/complete_m3_alpha025_target.py --root %q --log-dir %q --ddp-gpus %q --eval-gpu %q %s 2>&1 | tee %q; rc=\${PIPESTATUS[0]}; echo EXIT_CODE=\$rc; exit \$rc" \
  "$ROOT" "$PYTHON" "$ROOT" "$LOG_DIR" "$DDP_GPUS" "$EVAL_GPU" \
  "$EXTRA_ARGS" "$LOG_DIR/run.log"
printf -v LAUNCH 'bash -c %q' "$COMMAND"
tmux respawn-pane -k -t "$SESSION":0.0 "$LAUNCH"

echo "Started: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR/run.log"
echo "Physical training GPUs: $DDP_GPUS; evaluation GPU: $EVAL_GPU"
echo "Stage mode: $([[ "$REPLACE_TARGET" == "1" ]] && echo replace || echo resume-existing)"
echo "This completes only M3 alpha=.25 target fine-tuning; Test is forbidden."
