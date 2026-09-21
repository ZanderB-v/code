#!/usr/bin/env bash
set -euo pipefail

ROOT="/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition"
SESSION="corruption_protocol_v1_build"
PROTOCOL_ROOT="$ROOT/05_evaluation/corruption_protocol_v1"
REPLACE_FLAG=""

if [[ "${1:-}" == "--replace" ]]; then
  REPLACE_FLAG="--replace"
elif [[ -n "${1:-}" ]]; then
  echo "Usage: bash scripts/corruption/run_corruption_protocol_v1_tmux.sh [--replace]"
  exit 2
fi

mkdir -p "$PROTOCOL_ROOT/logs"
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n build \
  "source /home/wudayu/anaconda3/etc/profile.d/conda.sh && \
   conda activate openocr_svtrv2 && \
   cd '$ROOT' && \
   python scripts/corruption/build_corruption_protocol.py \
     --root '$ROOT' \
     --split dev \
     --seed 20260801 \
     --workers 8 \
     --review-samples-per-language 2 \
     $REPLACE_FLAG \
     2>&1 | tee '$PROTOCOL_ROOT/logs/build_dev.log'; \
   status=\${PIPESTATUS[0]}; \
   echo CORRUPTION_DEV_BUILD_EXIT_CODE=\$status; \
   exec bash"

echo "Started: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Review after completion:"
echo "  $PROTOCOL_ROOT/dev_calibration_review.html"
echo "This step does not freeze the protocol and does not touch test."
