#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition
SESSION=b1_full_svtrv2_p1
CONDA_ENV=openocr_svtrv2
REPLACE=
REBUILD=
PREFLIGHT_ONLY=

for arg in "$@"; do
  case "$arg" in
    --replace) REPLACE=--replace ;;
    --rebuild-msr-lmdb) REBUILD=--rebuild-msr-lmdb ;;
    --preflight-only) PREFLIGHT_ONLY=--preflight-only ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ "$REPLACE" == "--replace" ]]; then
    tmux kill-session -t "$SESSION"
  else
    echo "Session already exists: $SESSION" >&2
    echo "Attach: tmux attach -t $SESSION" >&2
    exit 1
  fi
fi

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$ROOT/04_model_training/logs/b1_full_svtrv2_p1_$STAMP"
mkdir -p "$LOG_DIR"
printf '%s\n' "$LOG_DIR" > \
  "$ROOT/04_model_training/logs/b1_full_svtrv2_p1_latest.txt"

source /home/wudayu/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
python "$ROOT/scripts/svtrv2/preflight_p1_formal.py" \
  --root "$ROOT" \
  --minimum-gpus 2 \
  --output "$LOG_DIR/preflight.json" \
  2>&1 | tee "$LOG_DIR/preflight.log"

tmux new-session -d -s "$SESSION" \
  "set -o pipefail; source /home/wudayu/anaconda3/etc/profile.d/conda.sh && \
   conda activate '$CONDA_ENV' && \
   cd '$ROOT' && \
   PYTHONUNBUFFERED=1 python scripts/svtrv2/run_b1_full_svtrv2.py \
     --root '$ROOT' \
     --log-dir '$LOG_DIR' \
     --max-epoch 50 \
     --synthetic-eval-every 5 \
     --target-eval-every 2 \
     --synthetic-patience 4 \
     --target-patience 5 \
     --synthetic-min-epoch 15 \
     --target-min-epoch 10 \
     --batch-size-per-card 16 \
     --num-workers 4 \
     --lr 0.000025 \
     --ddp-gpus 0,1 \
     --eval-gpu 0 \
     $REPLACE $REBUILD $PREFLIGHT_ONLY \
     2>&1 | tee '$LOG_DIR/pipeline.log'"
tmux set-window-option -t "$SESSION":0 remain-on-exit on

echo "B1 full SVTRv2-S pipeline started."
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR/pipeline.log"
echo "Follow: tail -F $LOG_DIR/pipeline.log"
