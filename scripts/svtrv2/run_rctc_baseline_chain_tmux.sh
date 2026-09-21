#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
CONDA_SH="${CONDA_SH:-/home/wudayu/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-openocr_svtrv2}"
SESSION="${SESSION:-rctc_baseline_chain_p1}"
DDP_GPUS="${DDP_GPUS:-0,1}"
EVAL_GPU="${EVAL_GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-50}"
BATCH_SIZE_PER_CARD="${BATCH_SIZE_PER_CARD:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-0.00005}"

REPLACE=0
REBUILD_MSR_LMDB=0
PREFLIGHT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --replace) REPLACE=1 ;;
    --rebuild-msr-lmdb) REBUILD_MSR_LMDB=1 ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    -h|--help)
      echo "Usage: bash scripts/svtrv2/run_rctc_baseline_chain_tmux.sh [--replace] [--rebuild-msr-lmdb] [--preflight-only]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ "$REPLACE" == "1" ]]; then
    tmux kill-session -t "$SESSION"
  else
    echo "tmux session already exists: $SESSION" >&2
    exit 1
  fi
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

LOG_DIR="$ROOT/04_model_training/logs/rctc_baseline_chain_p1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
printf '%s\n' "$LOG_DIR" > \
  "$ROOT/04_model_training/logs/rctc_baseline_chain_p1_latest.txt"

python "$ROOT/scripts/svtrv2/preflight_p1_formal.py" \
  --root "$ROOT" \
  --minimum-gpus 2 \
  --output "$LOG_DIR/preflight.json" \
  2>&1 | tee "$LOG_DIR/preflight.log"

COMMAND=(
  python "$ROOT/scripts/svtrv2/run_p1_rctc_baseline_chain.py"
  --root "$ROOT"
  --log-dir "$LOG_DIR"
  --max-epoch "$MAX_EPOCH"
  --batch-size-per-card "$BATCH_SIZE_PER_CARD"
  --num-workers "$NUM_WORKERS"
  --lr "$LR"
  --ddp-gpus "$DDP_GPUS"
  --eval-gpu "$EVAL_GPU"
)
if [[ "$REPLACE" == "1" ]]; then
  COMMAND+=(--replace)
fi
if [[ "$REBUILD_MSR_LMDB" == "1" ]]; then
  COMMAND+=(--rebuild-msr-lmdb)
fi
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  COMMAND+=(--preflight-only)
fi

printf -v COMMAND_TEXT '%q ' "${COMMAND[@]}"
tmux new-session -d -s "$SESSION" \
  "set -o pipefail; source '$CONDA_SH' && conda activate '$CONDA_ENV' && cd '$ROOT' && ${COMMAND_TEXT} 2>&1 | tee '$LOG_DIR/chain.log'"
tmux set-window-option -t "$SESSION":0 remain-on-exit on

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "Formal P1/MSR zero-training preflight started."
else
  echo "Formal P1/MSR baseline chain started."
fi
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR/chain.log"
echo "Follow: tail -F $LOG_DIR/chain.log"
echo "The first run builds verified LMDB caches; later runs reuse them."
