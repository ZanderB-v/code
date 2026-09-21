#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
SESSION="near_miss_hem_v1"
CONDA_SH="${CONDA_SH:-/home/wudayu/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-openocr_svtrv2}"
LOG_DIR="$ROOT/04_model_training/logs/near_miss_hem_v1"

if [[ "${1:-}" == "--worker" ]]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  cd "$ROOT"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=0,1
  export PYTHONUNBUFFERED=1

  python scripts/svtrv2/check_near_miss_hem_gate.py \
    --root "$ROOT" \
    --require-eligible

  echo "HEM_ELIGIBILITY_PASSED"
  echo "Formal HEM training is intentionally not launched until target-train mining and clean-hard-sample audit are materialized."
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi
mkdir -p "$LOG_DIR"

printf -v worker '%q ' bash "$ROOT/scripts/svtrv2/run_near_miss_hem_tmux.sh" --worker
printf -v command 'set -o pipefail; %s 2>&1 | tee -a %q; rc=${PIPESTATUS[0]}; printf "HEM_GATE_EXIT_CODE=%%s\n" "$rc"; exit "$rc"' \
  "$worker" "$LOG_DIR/gate.log"
printf -v launch 'bash -c %q' "$command"

tmux new-session -d -s "$SESSION"
tmux set-window-option -t "$SESSION":0 remain-on-exit on
tmux respawn-pane -k -t "$SESSION":0.0 "$launch"

echo "Near-Miss HEM gate started."
echo "Physical GPUs 0 and 1 are visible; eligibility audit itself does not use GPU."
echo "Formal HEM keeps batch_size_per_card=16 (global batch=32) to match frozen controls."
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR/gate.log"
