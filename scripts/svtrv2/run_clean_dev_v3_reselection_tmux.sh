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

SESSION="clean_dev_v3_reselection"
PROTOCOL="$ROOT/01_data_preparation/clean_dev_v3_adjudicated"
OUT="$ROOT/05_evaluation/clean_dev_v3_checkpoint_reselection"
PROFILE="$ROOT/05_evaluation/clean_dev_v3_m3_error_profile_v1"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/05_evaluation/clean_dev_v3_reselection_logs/$STAMP"
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

for required in \
  "$PROTOCOL/frozen_clean_dev_v3_manifest.json" \
  "$PROTOCOL/labels/zh.txt" \
  "$PROTOCOL/labels/ug.txt" \
  "$PROTOCOL/labels/kk.txt"; do
  [[ -f "$required" ]] || { echo "MISSING: $required"; exit 1; }
done

mkdir -p "$LOG_DIR"
tmux new-session -d -s "$SESSION"
tmux set-window-option -t "$SESSION":0 remain-on-exit on
printf -v COMMAND \
  "set -o pipefail; cd %q && %q scripts/protocol/test_clean_dev_v3_adjudicated.py 2>&1 | tee %q && %q scripts/svtrv2/rescore_clean_dev_v2.py --root %q --protocol-dir %q --frozen-manifest frozen_clean_dev_v3_manifest.json --protocol-label 'Clean Dev V3' --output-dir %q --preflight-only 2>&1 | tee %q && %q scripts/svtrv2/rescore_clean_dev_v2.py --root %q --protocol-dir %q --frozen-manifest frozen_clean_dev_v3_manifest.json --protocol-label 'Clean Dev V3' --output-dir %q %s 2>&1 | tee %q && %q scripts/svtrv2/build_clean_dev_v2_m3_error_profile.py --root %q --reselection-dir %q --protocol-dir %q --frozen-manifest frozen_clean_dev_v3_manifest.json --protocol-label 'Clean Dev V3' --model auto --output %q --replace 2>&1 | tee %q; rc=\${PIPESTATUS[0]}; echo EXIT_CODE=\$rc | tee %q; exit \$rc" \
  "$ROOT" "$PYTHON" "$LOG_DIR/protocol_check.log" \
  "$PYTHON" "$ROOT" "$PROTOCOL" "$OUT" "$LOG_DIR/preflight.log" \
  "$PYTHON" "$ROOT" "$PROTOCOL" "$OUT" "$REPLACE" "$LOG_DIR/reselection.log" \
  "$PYTHON" "$ROOT" "$OUT" "$PROTOCOL" "$PROFILE" "$LOG_DIR/error_profile.log" \
  "$LOG_DIR/exit_code"
printf -v LAUNCH 'bash -c %q' "$COMMAND"
tmux respawn-pane -k -t "$SESSION":0.0 "$LAUNCH"

echo "Started: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR"
echo "Output: $OUT/reselection_manifest.json"
echo "Review pages: $PROFILE/review/{zh,ug,kk}_review.html"
echo "No training, GPU inference, Corrupted Dev, or Test is run."
