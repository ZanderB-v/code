#!/usr/bin/env bash
set -euo pipefail

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
: "${ROOT:?Set ROOT to the svtrv2_line_recognition project root}"
: "${PYTHON:?Set PYTHON to the openocr_svtrv2 Python executable}"
SESSION="clean_dev_v2_reselection"
OUT="$ROOT/05_evaluation/clean_dev_v2_checkpoint_reselection"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/05_evaluation/clean_dev_v2_checkpoint_reselection_logs/$STAMP"
REPLACE=""

if [[ "${1:-}" == "--replace" ]]; then
  REPLACE="--replace"
  tmux kill-session -t "$SESSION" 2>/dev/null || true
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Already running: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi

mkdir -p "$LOG_DIR"
tmux new-session -d -s "$SESSION"
tmux set-window-option -t "$SESSION":0 remain-on-exit on
printf -v COMMAND \
  "set -o pipefail; cd %q && %q scripts/protocol/verify_clean_dev_v2.py --root %q && %q scripts/protocol/test_clean_dev_v2_evaluator_contract.py && %q scripts/svtrv2/rescore_clean_dev_v2.py --root %q --preflight-only 2>&1 | tee %q && %q scripts/svtrv2/rescore_clean_dev_v2.py --root %q --output-dir %q %s 2>&1 | tee %q; rc=\${PIPESTATUS[0]}; echo EXIT_CODE=\$rc | tee %q; exit \$rc" \
  "$ROOT" "$PYTHON" "$ROOT" "$PYTHON" "$PYTHON" "$ROOT" \
  "$LOG_DIR/preflight.log" "$PYTHON" "$ROOT" "$OUT" "$REPLACE" \
  "$LOG_DIR/reselection.log" "$LOG_DIR/exit_code"
printf -v LAUNCH 'bash -c %q' "$COMMAND"
tmux respawn-pane -k -t "$SESSION":0.0 "$LAUNCH"

echo "Started: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR/reselection.log"
echo "Output: $OUT/reselection_manifest.json"
echo "No training, Corrupted Dev, or Test is run."
