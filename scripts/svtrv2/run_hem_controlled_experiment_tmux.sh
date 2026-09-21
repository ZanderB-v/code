#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-hem_controlled_v1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON="${PYTHON:-/home/wudayu/anaconda3/envs/openocr_svtrv2/bin/python}"
MODE="formal"
REPLACE=0

for arg in "$@"; do
  case "$arg" in
    --preflight-only) MODE="preflight" ;;
    --replace) REPLACE=1 ;;
    *) echo "Usage: $0 [--preflight-only] [--replace]" >&2; exit 2 ;;
  esac
done

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ "$REPLACE" -eq 1 ]]; then
    tmux kill-session -t "$SESSION"
  else
    echo "Session already exists: $SESSION"
    echo "Attach: tmux attach -t $SESSION"
    exit 0
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/04_model_training/logs/hem_controlled_v1_${STAMP}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/console.log"
RUNNER="$LOG_DIR/run.sh"

EXTRA_ARGS=""
if [[ "$MODE" == "preflight" ]]; then
  EXTRA_ARGS="$EXTRA_ARGS --preflight-only"
fi
if [[ "$REPLACE" -eq 1 ]]; then
  EXTRA_ARGS="$EXTRA_ARGS --replace"
fi

cat >"$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "===== HEM V1 controlled experiment ====="
echo "mode=$MODE"
echo "physical GPU=1 only"
echo "world_size=1, batch_per_card=32, global_batch=32"
echo "models=B1+HEM, SOAR-SVTR+HEM"
echo "Test=forbidden"
"$PYTHON" scripts/svtrv2/run_hem_controlled_experiment.py \
  --root "$ROOT" \
  --log-dir "$LOG_DIR" \
  --smoke-steps 200 \
  --master-port 29931 \
  $EXTRA_ARGS
echo "HEM_CONTROLLED_V1_${MODE^^}_COMPLETE"
echo "Test was not evaluated."
EOF
chmod +x "$RUNNER"

tmux new-session -d -s "$SESSION" -n hem
tmux set-window-option -t "$SESSION" remain-on-exit on
tmux send-keys -t "$SESSION":0.0 \
  "bash '$RUNNER' 2>&1 | tee '$LOG'; exit \${PIPESTATUS[0]}" C-m

echo "Started: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG"
echo "Only physical GPU1 is used. GPU0 is never selected."
