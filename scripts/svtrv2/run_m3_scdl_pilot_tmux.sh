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

SESSION="${SESSION:-m3_scdl_target_only_pilot_v1}"
GPU_ID="${GPU_ID:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
M3_S50_CHECKPOINT="${M3_S50_CHECKPOINT:-$ROOT/04_model_training/runs/svtrv2_s_m3_dual_order_s50/best_clean_dev_macro_cer.pth}"
PROFILE="$ROOT/05_evaluation/clean_dev_v4_m3_error_profile_v1/error_profile_summary.json"
RESELECTION="$ROOT/05_evaluation/clean_dev_v4_checkpoint_reselection"
PROTOCOL="$ROOT/01_data_preparation/clean_dev_v4_adjudicated"

mode="formal"
replace=""
for argument in "$@"; do
  case "$argument" in
    --preflight-only) mode="preflight" ;;
    --replace) replace="--replace" ;;
    *) echo "Unknown argument: $argument" >&2; exit 2 ;;
  esac
done

for required in "$M3_S50_CHECKPOINT" "$PROFILE" \
  "$RESELECTION/final_decisions.json" \
  "$PROTOCOL/frozen_clean_dev_v4_manifest.json"; do
  [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 1; }
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
  pane_dead="$(tmux display-message -p -t "$SESSION":0.0 '#{pane_dead}')"
  if [[ "$pane_dead" != "1" ]]; then
    echo "Session already running: tmux attach -t $SESSION"
    exit 0
  fi
  tmux kill-session -t "$SESSION"
fi

stamp="$(date +%Y%m%d_%H%M%S)"
state_dir="${STATE_DIR:-$ROOT/04_model_training/logs/m3_scdl_target_only_pilot_v1_state}"
console_log="$ROOT/04_model_training/logs/m3_scdl_target_only_pilot_v1_$stamp.log"
mkdir -p "$state_dir"
command=(
  "$PYTHON" "$ROOT/scripts/svtrv2/run_m3_scdl_pilot.py"
  --root "$ROOT"
  --log-dir "$state_dir"
  --m3-s50-checkpoint "$M3_S50_CHECKPOINT"
  --m3-profile "$PROFILE"
  --v4-reselection-dir "$RESELECTION"
  --v4-protocol-dir "$PROTOCOL"
  --v4-frozen-manifest frozen_clean_dev_v4_manifest.json
  --gpu 0
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
)
if [[ "$mode" == "preflight" ]]; then
  command+=(--preflight-only)
elif [[ -n "$replace" ]]; then
  command+=(--replace)
else
  command+=(--resume-existing)
fi

printf -v worker '%q ' env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
  "${command[@]}"
printf -v launch 'set -o pipefail; %s 2>&1 | tee -a %q; rc=${PIPESTATUS[0]}; printf "EXIT_CODE=%%s\\n" "$rc" | tee -a %q; exit "$rc"' \
  "$worker" "$console_log" "$console_log"
printf -v shell_command 'bash -c %q' "$launch"

tmux new-session -d -s "$SESSION"
tmux set-window-option -t "$SESSION":0 remain-on-exit on
tmux respawn-pane -k -t "$SESSION":0.0 "$shell_command"

echo "Started: $SESSION"
echo "Physical GPU: $GPU_ID only; frozen global batch size: $BATCH_SIZE"
echo "Console log: $console_log"
echo "Audited resume state: $state_dir"
echo "Attach: tmux attach -t $SESSION"
echo "Pilot: target-only M3+SCDL; Clean Dev V4 selection; Corrupted Dev/Test forbidden."
