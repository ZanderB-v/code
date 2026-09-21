#!/usr/bin/env bash
set -euo pipefail

SESSION="b1_scale_step_control_v1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/wudayu/anaconda3/envs/openocr_svtrv2/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/04_model_training/logs/b1_scale_step_control_v1_$STAMP"

REPLACE=0
PREFLIGHT=0
for arg in "$@"; do
  case "$arg" in
    --replace) REPLACE=1 ;;
    --preflight-only) PREFLIGHT=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  PANE_DEAD="$(tmux display-message -p -t "$SESSION" '#{pane_dead}')"
  if [[ "$REPLACE" -eq 1 || "$PANE_DEAD" == "1" ]]; then
    tmux kill-session -t "$SESSION"
  else
    echo "Session already exists. Attach with: tmux attach -t $SESSION" >&2
    exit 1
  fi
fi

mkdir -p "$LOG_DIR"
RUN_SH="$LOG_DIR/run.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -uo pipefail'
  printf 'cd %q\n' "$ROOT"
  echo 'export CUDA_VISIBLE_DEVICES=0'
  echo 'export MALLOC_ARENA_MAX=2'
  echo 'export OMP_NUM_THREADS=1'
  echo 'export MKL_NUM_THREADS=1'
  printf '%q ' "$PYTHON" "$ROOT/scripts/svtrv2/run_b1_scale_step_control.py" \
    --root "$ROOT" --log-dir "$LOG_DIR" --physical-gpu 0 \
    --batch-size 32 --num-workers 8
  [[ "$REPLACE" -eq 1 ]] && printf '%q ' --replace
  [[ "$PREFLIGHT" -eq 1 ]] && printf '%q ' --preflight-only
  printf '2>&1 | tee %q\n' "$LOG_DIR/summary.log"
  echo 'rc=${PIPESTATUS[0]}'
  printf 'echo "EXIT_CODE=$rc" | tee %q\n' "$LOG_DIR/EXIT_CODE"
  echo 'exit "$rc"'
} > "$RUN_SH"
chmod +x "$RUN_SH"

tmux new-session -d -s "$SESSION" -c "$ROOT" "bash '$RUN_SH'"
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
echo "Started: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR/summary.log"
echo "Only physical GPU0 is visible; FP32 batch=32 preserves global batch 32."
