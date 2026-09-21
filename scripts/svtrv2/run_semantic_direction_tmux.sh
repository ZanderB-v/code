#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
SESSION=semantic_direction_v1
CONDA_SH="${CONDA_SH:-/home/wudayu/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-openocr_svtrv2}"
LOG_DIR="$ROOT/04_model_training/logs/semantic_direction_v1"

if [[ "${1:-}" == "--worker" ]]; then
  shift
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  cd "$ROOT"
  export PYTHONUNBUFFERED=1
  python scripts/svtrv2/preflight_p1_formal.py --root "$ROOT" \
    --minimum-gpus 2 --output "$LOG_DIR/runtime_preflight.json"
  python -m torch.distributed.run --nproc_per_node=2 --master_port=29929 \
    scripts/svtrv2/test_semantic_direction.py --root "$ROOT" --ddp
  python scripts/svtrv2/run_semantic_direction_study.py \
    --root "$ROOT" --log-dir "$LOG_DIR" "$@"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  DEAD=$(tmux display-message -p -t "$SESSION":0.0 '#{pane_dead}')
  if [[ "$DEAD" != "1" ]]; then
    echo "Session already running. Attach: tmux attach -t $SESSION"
    exit 0
  fi
  tmux kill-session -t "$SESSION"
fi
mkdir -p "$LOG_DIR"
printf -v WORKER '%q ' bash "$ROOT/scripts/svtrv2/run_semantic_direction_tmux.sh" --worker "$@"
# Create a waiting pane so remain-on-exit is set before any command can fail.
tmux new-session -d -s "$SESSION"
tmux set-window-option -t "$SESSION":0 remain-on-exit on
printf -v COMMAND 'set -o pipefail; %s 2>&1 | tee -a %q; rc=${PIPESTATUS[0]}; printf "SEMANTIC_DIRECTION_EXIT_CODE=%%s\\n" "$rc"; exit "$rc"' \
  "$WORKER" "$LOG_DIR/study.log"
printf -v LAUNCH 'bash -c %q' "$COMMAND"
tmux respawn-pane -k -t "$SESSION":0.0 "$LAUNCH"
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR/study.log"
echo "Resume: rerun this command without --preflight-only."
