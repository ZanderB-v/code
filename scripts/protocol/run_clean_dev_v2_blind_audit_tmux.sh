#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python || true)}"
SESSION="${SESSION:-clean_dev_v2_blind_audit_v1}"
OUTPUT="$ROOT/01_data_preparation/clean_dev_v2_protocol/blind_audit_v1"

if [[ "${1:-}" == "--worker" ]]; then
  shift
  [[ -n "$PYTHON" && -x "$PYTHON" ]] || {
    echo "PYTHON is not executable: $PYTHON" >&2
    exit 1
  }
  cd "$ROOT"
  "$PYTHON" scripts/protocol/build_clean_dev_v2_blind_audit.py \
    --root "$ROOT" \
    --output "$OUTPUT"
  "$PYTHON" scripts/protocol/test_clean_dev_v2_blind_audit.py \
    --root "$ROOT" \
    --audit-dir "$OUTPUT"
  echo "CLEAN_DEV_V2_BLIND_AUDIT_PACKAGE_UPDATED"
  echo "Review pages: $OUTPUT/{zh,ug,kk}_blind_review.html"
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
stamp="$(date +%Y%m%d_%H%M%S)"
log="$OUTPUT/logs/build_$stamp.log"
printf -v worker '%q ' env ROOT="$ROOT" PYTHON="$PYTHON" \
  bash "$ROOT/scripts/protocol/run_clean_dev_v2_blind_audit_tmux.sh" --worker
printf -v command 'set -o pipefail; %s 2>&1 | tee -a %q; rc=${PIPESTATUS[0]}; printf "EXIT_CODE=%%s\n" "$rc" | tee -a %q; exit "$rc"' \
  "$worker" "$log" "$log"
printf -v launch 'bash -c %q' "$command"
tmux new-session -d -s "$SESSION"
tmux set-option -t "$SESSION" remain-on-exit on
tmux respawn-pane -k -t "$SESSION":0.0 "$launch"

echo "Started: $SESSION"
echo "No GPU and no model prediction are used."
echo "Log: $log"
echo "Attach: tmux attach -t $SESSION"
