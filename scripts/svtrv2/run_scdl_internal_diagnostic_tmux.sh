#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/s1home/turdy_stu/wudayu/envs/openocr_svtrv2/bin/python}"
SESSION="${SESSION:-scdl_internal_diagnostic_v2}"
CONFIG="${CONFIG:-$ROOT/04_model_training/configs/svtrv2_s_m3_scdl_full_v1_dual_order_s50_to_target.yml}"
CHECKPOINT="${CHECKPOINT:-$ROOT/04_model_training/runs/svtrv2_s_m3_scdl_full_v1_dual_order_s50_to_target/best_clean_dev_macro_cer.pth}"
if [[ ! -f "$CHECKPOINT" && "$CHECKPOINT" == "$ROOT/04_model_training/runs/svtrv2_s_m3_scdl_full_v1_dual_order_s50_to_target/best_clean_dev_macro_cer.pth" ]]; then
  data_checkpoint="/s1home/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition/04_model_training/runs/svtrv2_s_m3_scdl_full_v1_dual_order_s50_to_target/best_clean_dev_macro_cer.pth"
  if [[ -f "$data_checkpoint" ]]; then CHECKPOINT="$data_checkpoint"; fi
fi
OUTPUT="$ROOT/05_evaluation/m3_scdl_diagnostic_v1/internal_v2"
LOG="$ROOT/04_model_training/logs/scdl_internal_diagnostic_v2.log"

for required in "$PYTHON" "$CONFIG" "$CHECKPOINT"; do
  [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 1; }
done
[[ ! -e "$OUTPUT" ]] || {
  echo "Existing audit found: $OUTPUT" >&2
  echo "Review it before requesting another run; no outputs are overwritten." >&2
  exit 1
}
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: tmux attach -t $SESSION"
  exit 0
fi

mkdir -p "$(dirname "$LOG")"
command=(
  "$PYTHON" "$ROOT/scripts/svtrv2/diagnose_scdl_internal.py"
  --root "$ROOT"
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --output "$OUTPUT"
  --max-batches 64
  --batch-size 8
  --negative-examples-per-script 100
  --seed 20260731
)
printf -v worker '%q ' env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  PYTHONDONTWRITEBYTECODE=1 "${command[@]}"
printf -v launch 'set -o pipefail; %s 2>&1 | tee %q; rc=${PIPESTATUS[0]}; printf "EXIT_CODE=%%s\n" "$rc" | tee -a %q; exit "$rc"' \
  "$worker" "$LOG" "$LOG"
printf -v shell_command 'bash -c %q' "$launch"

tmux new-session -d -s "$SESSION" -n audit "$shell_command"
tmux set-window-option -t "$SESSION":audit remain-on-exit on
echo "Started read-only SCDL internal diagnostic on physical GPU0."
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG"
echo "Output: $OUTPUT"
echo "Target Train only; no training, Corrupted Dev, or Test."
