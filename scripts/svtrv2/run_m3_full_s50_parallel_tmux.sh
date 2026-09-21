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

SESSION="${SESSION:-m3_full_s50_v1_parallel}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE=32
NUM_WORKERS="${NUM_WORKERS:-8}"
RCTC_INIT="${RCTC_INIT:-$ROOT/04_model_training/checkpoint_adapters/b1_full_s50/d2_rctc_for_full_svtrv2.pth}"
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

for required in "$RCTC_INIT" "$PROFILE" \
  "$RESELECTION/final_decisions.json" \
  "$PROTOCOL/frozen_clean_dev_v4_manifest.json"; do
  [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 1; }
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
  live_panes="$(tmux list-panes -t "$SESSION" -F '#{pane_dead}' | grep -c '^0$' || true)"
  if [[ "$live_panes" -gt 0 ]]; then
    echo "Session already running: tmux attach -t $SESSION"
    exit 0
  fi
  tmux kill-session -t "$SESSION"
fi

stamp="$(date +%Y%m%d_%H%M%S)"

if [[ "$mode" == "formal" ]]; then
  for track in sldr scdl; do
    marker="$ROOT/04_model_training/logs/m3_${track}_full_s50_v1_state/preflight_complete.json"
    [[ -f "$marker" ]] || {
      echo "MISSING PASSED PREFLIGHT: $marker" >&2
      echo "Run this first: bash scripts/svtrv2/run_m3_full_s50_parallel_tmux.sh --preflight-only" >&2
      exit 1
    }
    "$PYTHON" - "$ROOT" "$marker" "$track" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
marker = Path(sys.argv[2])
track = sys.argv[3]
payload = json.loads(marker.read_text(encoding="utf-8-sig"))
expected_status = f"M3_{track.upper()}_FULL_S50_V1_PREFLIGHT_OK"
if payload.get("status") != expected_status:
    raise SystemExit(f"Invalid preflight status in {marker}: {payload.get('status')}")

files = {
    "runner": root / "scripts/svtrv2/run_m3_full_refinement.py",
    "protocol": root / "scripts/svtrv2/dual_order_protocol.py",
    "validator": root / "scripts/svtrv2/validate_dual_order_method_config.py",
    "smoke": root / "scripts/svtrv2/smoke_dual_order_method.py",
}
for name, path in files.items():
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = payload.get("code_sha256", {}).get(name)
    if digest != expected:
        raise SystemExit(
            f"Preflight is stale for {track}: {name} changed; rerun --preflight-only"
        )
print(f"{expected_status}: hash binding verified")
PY
  done
fi

build_worker() {
  local track="$1"
  local state_dir="$ROOT/04_model_training/logs/m3_${track}_full_s50_v1_state"
  local console_log="$ROOT/04_model_training/logs/m3_${track}_full_s50_v1_${stamp}.log"
  local -a command=(
    "$PYTHON" "$ROOT/scripts/svtrv2/run_m3_full_refinement.py"
    --track "$track"
    --root "$ROOT"
    --log-dir "$state_dir"
    --rctc-init "$RCTC_INIT"
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
  mkdir -p "$state_dir"
  printf -v worker '%q ' env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${command[@]}"
  printf -v launch 'set -o pipefail; %s 2>&1 | tee -a %q; rc=${PIPESTATUS[0]}; printf "EXIT_CODE=%%s\\n" "$rc" | tee -a %q; exit "$rc"' \
    "$worker" "$console_log" "$console_log"
  printf 'bash -c %q' "$launch"
}

sldr_worker="$(build_worker sldr)"
scdl_worker="$(build_worker scdl)"

tmux new-session -d -s "$SESSION" -n sldr "$sldr_worker"
tmux set-window-option -t "$SESSION":sldr remain-on-exit on
tmux new-window -d -t "$SESSION" -n scdl "$scdl_worker"
tmux set-window-option -t "$SESSION":scdl remain-on-exit on
tmux select-window -t "$SESSION":sldr

echo "Started parallel full-S50 tracks in one tmux session: $SESSION"
echo "  window sldr: M3 + SLDR_FULL_S50_V1"
echo "  window scdl: M3 + SCDL_FULL_S50_V1"
echo "Both windows use physical GPU$GPU_ID concurrently, batch size 32 each."
echo "Attach: tmux attach -t $SESSION"
echo "Switch windows: Ctrl-b n (next), Ctrl-b p (previous)"
echo "List windows: tmux list-windows -t $SESSION"
echo "Clean Dev V4 selects checkpoints; Corrupted Dev and Test are forbidden."
