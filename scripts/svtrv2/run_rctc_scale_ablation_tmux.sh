#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
CONDA_SH="${CONDA_SH:-/home/wudayu/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-openocr_svtrv2}"
SESSION="${SESSION:-rctc_scale_ablation_p1}"
DDP_GPUS="${DDP_GPUS:-0,1}"
EVAL_GPU="${EVAL_GPU:-0}"
SCALES="${SCALES:-s10 s25 s50}"

REPLACE=0
RESUME=0
FORCE_RERUN_S50=0
PREFLIGHT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --replace) REPLACE=1 ;;
    --resume) RESUME=1 ;;
    --force-rerun-s50) FORCE_RERUN_S50=1 ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    -h|--help)
      echo "Usage: bash scripts/svtrv2/run_rctc_scale_ablation_tmux.sh [--replace|--resume] [--force-rerun-s50] [--preflight-only]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ "$REPLACE" == "1" && "$RESUME" == "1" ]]; then
  echo "--replace and --resume are mutually exclusive" >&2
  exit 2
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ "$REPLACE" == "1" || "$RESUME" == "1" ]]; then
    tmux kill-session -t "$SESSION"
  else
    echo "tmux session already exists: $SESSION" >&2
    exit 1
  fi
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

LATEST_FILE="$ROOT/04_model_training/logs/rctc_scale_ablation_p1_latest.txt"
if [[ "$RESUME" == "1" ]]; then
  if [[ ! -s "$LATEST_FILE" ]]; then
    echo "Missing scale-ablation latest pointer: $LATEST_FILE" >&2
    exit 1
  fi
  LOG_DIR="$(cat "$LATEST_FILE")"
  if [[ ! -d "$LOG_DIR" ]]; then
    echo "Resume log directory does not exist: $LOG_DIR" >&2
    exit 1
  fi
else
  LOG_DIR="$ROOT/04_model_training/logs/rctc_scale_ablation_p1_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$LOG_DIR"
  printf '%s\n' "$LOG_DIR" > "$LATEST_FILE"
fi

python "$ROOT/scripts/svtrv2/preflight_p1_formal.py" \
  --root "$ROOT" \
  --minimum-gpus 2 \
  --output "$LOG_DIR/preflight.json" \
  2>&1 | tee "$LOG_DIR/preflight.log"

python "$ROOT/scripts/svtrv2/test_segmented_scheduler_control.py" \
  2>&1 | tee "$LOG_DIR/segmented_scheduler_control.log"

read -r -a SCALE_ARGS <<< "$SCALES"
COMMAND=(
  python "$ROOT/scripts/svtrv2/run_p1_scale_ablation.py"
  --root "$ROOT"
  --log-dir "$LOG_DIR"
  --scales "${SCALE_ARGS[@]}"
  --ddp-gpus "$DDP_GPUS"
  --eval-gpu "$EVAL_GPU"
)
if [[ "$REPLACE" == "1" ]]; then
  COMMAND+=(--replace)
fi
if [[ "$RESUME" == "1" ]]; then
  COMMAND+=(--resume)
fi
if [[ "$FORCE_RERUN_S50" == "1" ]]; then
  COMMAND+=(--force-rerun-s50)
fi
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  COMMAND+=(--preflight-only)
fi

printf -v COMMAND_TEXT '%q ' "${COMMAND[@]}"
tmux new-session -d -s "$SESSION" \
  "set -o pipefail; source '$CONDA_SH' && conda activate '$CONDA_ENV' && cd '$ROOT' && ${COMMAND_TEXT} 2>&1 | tee -a '$LOG_DIR/scale_chain.log'"
tmux set-window-option -t "$SESSION":0 remain-on-exit on

echo "P1/MSR scale ablation started."
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR/scale_chain.log"
echo "Follow: tail -F $LOG_DIR/scale_chain.log"
