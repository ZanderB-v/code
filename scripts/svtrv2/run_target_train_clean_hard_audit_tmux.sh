#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-target_train_clean_hard_audit_v1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON="${PYTHON:-/home/wudayu/anaconda3/envs/openocr_svtrv2/bin/python}"
OUTPUT="${OUTPUT:-$ROOT/04_model_training/hem_v1/target_train_clean_hard_audit_v1}"
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-512}"
MAX_BASE_BATCH_SIZE="${MAX_BASE_BATCH_SIZE:-2048}"
TARGET_GPU_MEMORY_GIB="${TARGET_GPU_MEMORY_GIB:-20}"
IO_WORKERS="${IO_WORKERS:-8}"
REVIEW_PER_STRATUM="${REVIEW_PER_STRATUM:-30}"
REVIEW_SEED="${REVIEW_SEED:-20260911}"
REPLACE=0

if [[ "${1:-}" == "--replace" ]]; then
  REPLACE=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--replace]" >&2
  exit 2
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$ROOT/scripts/svtrv2/target_train_clean_hard_audit.py" ]]; then
  echo "Audit script not found under ROOT=$ROOT" >&2
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

if [[ "$REPLACE" -eq 1 ]]; then
  rm -rf -- "$OUTPUT"
fi
mkdir -p "$OUTPUT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUTPUT/logs/mine_${STAMP}.log"
RUNNER="$OUTPUT/logs/run_${STAMP}.sh"

cat >"$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
echo "===== target train clean hard audit ====="
echo "physical GPU: 1"
echo "CUDA_VISIBLE_DEVICES=1"
echo "M3 alpha: 0.15"
echo "target memory: ${TARGET_GPU_MEMORY_GIB} GiB"
echo "Test: forbidden"
CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/svtrv2/target_train_clean_hard_audit.py mine \
  --root "$ROOT" \
  --output "$OUTPUT" \
  --base-batch-size "$BASE_BATCH_SIZE" \
  --maximum-base-batch-size "$MAX_BASE_BATCH_SIZE" \
  --target-gpu-memory-gib "$TARGET_GPU_MEMORY_GIB" \
  --io-workers "$IO_WORKERS" \
  --review-per-stratum "$REVIEW_PER_STRATUM" \
  --review-seed "$REVIEW_SEED"
"$PYTHON" scripts/svtrv2/generate_target_train_hard_review_html.py \
  --root "$ROOT" \
  --audit-dir "$OUTPUT"
echo "TARGET_TRAIN_CLEAN_HARD_AUDIT_REVIEW_READY"
echo "Test was not evaluated."
EOF
chmod +x "$RUNNER"

tmux new-session -d -s "$SESSION" -n audit
tmux set-window-option -t "$SESSION" remain-on-exit on
tmux send-keys -t "$SESSION":0.0 \
  "bash '$RUNNER' 2>&1 | tee '$LOG'; exit \${PIPESTATUS[0]}" C-m

echo "Started: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG"
echo "GPU policy: physical GPU1 only; adaptive batch targets ${TARGET_GPU_MEMORY_GIB} GiB"
echo "No Test data is read and no formal HEM manifest is created before review."
