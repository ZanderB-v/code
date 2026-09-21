#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SESSION="${SESSION:-s50_m3_targeted_diagnostics_v1}"
PYTHON="${PYTHON:-$(command -v python || true)}"
GPU_ID="${GPU_ID:-0}"
OUTPUT="$ROOT/04_model_training/eval_reports/s50_m3_targeted_diagnostics_v1"
ERROR_REVIEW_OUTPUT="$ROOT/04_model_training/eval_reports/S50_M3_error_review_v1"
RAW_PATHS="$OUTPUT/ctc_raw_paths"
M3_CONFIG="${M3_CONFIG:-$ROOT/04_model_training/configs/svtrv2_s_m3_dual_order_s50_to_target.yml}"
M3_CHECKPOINT="${M3_CHECKPOINT:-$ROOT/04_model_training/runs/svtrv2_s_m3_dual_order_s50_to_target/best_clean_dev_macro_cer.pth}"
M3_RCTC_INIT="${M3_RCTC_INIT:-$ROOT/04_model_training/checkpoint_adapters/b1_full_s50/d2_rctc_for_full_svtrv2.pth}"

choose_report() {
  local canonical="$ROOT/04_model_training/eval_reports/svtrv2_s_m3_dual_order_s50_to_target_best_clean_dev"
  local migrated="$ROOT/04_model_training/eval_reports/migration_acceptance_m3_clean_dev_v1/result"
  if [[ -f "$canonical/metrics_macro_summary.json" ]]; then
    printf '%s\n' "$canonical"
  elif [[ -f "$migrated/metrics_macro_summary.json" ]]; then
    printf '%s\n' "$migrated"
  else
    return 1
  fi
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  mode="diagnostics"
  replace=0
  for argument in "$@"; do
    case "$argument" in
      --preflight-only) mode="preflight" ;;
      --diagnostics-only) mode="diagnostics" ;;
      --summarize-diagnostics) mode="summarize" ;;
      --run-nococ) mode="formal" ;;
      --replace) replace=1 ;;
      *) echo "Unknown argument: $argument" >&2; exit 2 ;;
    esac
  done
  [[ -n "$PYTHON" && -x "$PYTHON" ]] || { echo "PYTHON is not executable: $PYTHON" >&2; exit 1; }
  M3_REPORT="${M3_REPORT:-$(choose_report)}"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  export PYTHONUNBUFFERED=1
  cd "$ROOT"
  mkdir -p "$OUTPUT"

  "$PYTHON" scripts/svtrv2/test_s50_m3_targeted_diagnostics.py

  if [[ "$mode" != "diagnostics" && "$mode" != "summarize" && ! -f "$M3_RCTC_INIT" ]]; then
    cat >&2 <<EOF
MISSING_MATCHED_M3_RCTC_INITIALIZATION: $M3_RCTC_INIT
M3-noCOC must start from the exact pre-SGM RCTC checkpoint used by the frozen M3.
Set M3_RCTC_INIT=/absolute/path/to/d2_rctc_for_full_svtrv2.pth after transferring it.
Do not initialize this ablation from a trained M3 checkpoint.
EOF
    exit 1
  fi

  if [[ "$mode" != "summarize" ]]; then
    if [[ ! -f "$RAW_PATHS/raw_path_export_summary.json" || "$replace" == "1" ]]; then
      raw_args=()
      [[ "$replace" == "1" ]] && raw_args+=(--replace)
      "$PYTHON" scripts/svtrv2/export_ctc_raw_paths.py \
        --root "$ROOT" \
        --config "$M3_CONFIG" \
        --checkpoint "$M3_CHECKPOINT" \
        --m3-report "$M3_REPORT" \
        --output "$RAW_PATHS" \
        --device-id 0 \
        "${raw_args[@]}"
    else
      echo "Reusing hash-bound raw CTC paths: $RAW_PATHS"
    fi
  fi

  "$PYTHON" scripts/svtrv2/analyze_m3_ug_kk_errors.py \
    --root "$ROOT" \
    --m3-report "$M3_REPORT" \
    --raw-path-dir "$RAW_PATHS" \
    --output "$OUTPUT"

  "$PYTHON" scripts/svtrv2/build_s50_m3_error_review.py \
    --root "$ROOT" \
    --m3-report "$M3_REPORT" \
    --output "$ERROR_REVIEW_OUTPUT"

  if [[ "$mode" == "diagnostics" || "$mode" == "summarize" ]]; then
    echo "S50_M3_ERROR_REVIEW_V1_COMPLETE"
    echo "Review pages: $ERROR_REVIEW_OUTPUT/{zh,ug,kk}/review.html"
    exit 0
  fi

  readiness="$ROOT/00_docs/frozen_punctuation_protocol_v1/model_development_readiness.json"
  if [[ ! -f "$readiness" ]]; then
    cat <<EOF
PUNCTUATION_PROTOCOL_MUST_BE_CLOSED_BEFORE_M3_NOCOC
No formal M3-noCOC training was started.
If punctuation is excluded, all saved M3 epochs must first be re-evaluated and
the checkpoint reselected under the frozen normalized Clean Dev Macro CER.
If full transcription is required, GT punctuation must first be repaired and audited.
EOF
    exit 3
  fi
  "$PYTHON" - "$readiness" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
if payload.get("status") != "PUNCTUATION_PROTOCOL_CLOSED_FOR_MODEL_DEVELOPMENT":
    raise SystemExit("Punctuation readiness artifact is not approved")
if payload.get("test_evaluated") is not False:
    raise SystemExit("Punctuation readiness does not preserve the Test prohibition")
print("PUNCTUATION_PROTOCOL_READY_FOR_M3_NOCOC")
PY

  stamp="$(date +%Y%m%d_%H%M%S)"
  log_dir="$ROOT/04_model_training/logs/s50_m3_targeted_diagnostics_v1_$stamp"
  mkdir -p "$log_dir"
  nococ_args=(
    --root "$ROOT"
    --log-dir "$log_dir"
    --rctc-init "$M3_RCTC_INIT"
    --m3-report "$M3_REPORT"
    --punctuation-readiness "$readiness"
    --gpu 0
    --batch-size 32
    --num-workers 8
  )
  if [[ "$mode" == "preflight" ]]; then
    nococ_args+=(--preflight-only)
  elif [[ "$replace" == "1" ]]; then
    nococ_args+=(--replace)
  else
    nococ_args+=(--resume-existing)
  fi
  "$PYTHON" scripts/svtrv2/run_m3_nococ_ablation.py "${nococ_args[@]}"
  if [[ "$mode" == "preflight" ]]; then
    echo "S50_M3_TARGETED_DIAGNOSTICS_PREFLIGHT_OK"
  else
    echo "S50_M3_TARGETED_DIAGNOSTICS_AND_NOCOC_COMPLETE"
  fi
  exit 0
fi

[[ -n "$PYTHON" && -x "$PYTHON" ]] || {
  echo "Activate openocr_svtrv2 or set PYTHON=/absolute/path/to/python" >&2
  exit 1
}
if tmux has-session -t "$SESSION" 2>/dev/null; then
  pane_dead="$(tmux display-message -p -t "$SESSION":0.0 '#{pane_dead}')"
  if [[ "$pane_dead" != "1" ]]; then
    echo "Session already running. Attach: tmux attach -t $SESSION"
    exit 0
  fi
  tmux kill-session -t "$SESSION"
fi

mkdir -p "$OUTPUT/logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
log="$OUTPUT/logs/run_$timestamp.log"
printf -v worker '%q ' env ROOT="$ROOT" PYTHON="$PYTHON" GPU_ID="$GPU_ID" \
  M3_CONFIG="$M3_CONFIG" M3_CHECKPOINT="$M3_CHECKPOINT" M3_RCTC_INIT="$M3_RCTC_INIT" \
  bash "$ROOT/scripts/svtrv2/run_s50_m3_targeted_diagnostics_tmux.sh" --worker "$@"
printf -v command 'set -o pipefail; %s 2>&1 | tee -a %q; rc=${PIPESTATUS[0]}; printf "EXIT_CODE=%%s\n" "$rc" | tee -a %q; exit "$rc"' \
  "$worker" "$log" "$log"
printf -v launch 'bash -c %q' "$command"
tmux new-session -d -s "$SESSION"
tmux set-option -t "$SESSION" remain-on-exit on
tmux respawn-pane -k -t "$SESSION":0.0 "$launch"

echo "Started: $SESSION"
echo "GPU: physical GPU$GPU_ID only"
echo "Log: $log"
echo "Attach: tmux attach -t $SESSION"
