#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition
SESSION=m2_alpha_narrow_sweep_p1
SCRIPT_DIR="$ROOT/scripts/svtrv2"

if [[ "${1:-}" != "--worker" ]]; then
  if [[ "${1:-}" == "--replace" ]]; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
  elif tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Already running: $SESSION"
    echo "Attach: tmux attach -t $SESSION"
    exit 0
  elif [[ -f "$ROOT/04_model_training/logs/m2_alpha_narrow_sweep_latest.txt" ]]; then
    echo "A previous sweep exists. Use --replace to start a new sweep."
    exit 1
  fi
  STAMP=$(date +%Y%m%d_%H%M%S)
  LOG_DIR="$ROOT/04_model_training/logs/m2_alpha_narrow_sweep_$STAMP"
  mkdir -p "$LOG_DIR"
  printf '%s\n' "$LOG_DIR" > \
    "$ROOT/04_model_training/logs/m2_alpha_narrow_sweep_latest.txt"
  tmux new-session -d -s "$SESSION" \
    "bash '$0' --worker '$LOG_DIR' 2>&1 | tee '$LOG_DIR/sweep.log'"
  tmux set-window-option -t "$SESSION":0 remain-on-exit on
  echo "Started: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  echo "Log: $LOG_DIR/sweep.log"
  exit 0
fi

LOG_DIR=$2
source /home/wudayu/anaconda3/etc/profile.d/conda.sh
conda activate openocr_svtrv2
cd "$ROOT"
export PYTHONUNBUFFERED=1

python "$SCRIPT_DIR/preflight_p1_formal.py" \
  --root "$ROOT" \
  --minimum-gpus 2 \
  --output "$LOG_DIR/formal_preflight.json" \
  2>&1 | tee "$LOG_DIR/formal_preflight.log"

python "$SCRIPT_DIR/test_dual_order_components.py" \
  --root "$ROOT" \
  --output "$LOG_DIR/component_preflight.json" \
  2>&1 | tee "$LOG_DIR/component_preflight.log"

first_alpha=true
for spec in 015:0.15 020:0.20 025:0.25 030:0.30; do
  tag="${spec%%:*}"
  alpha="${spec##*:}"
  alpha_log_dir="$LOG_DIR/alpha_$tag"
  mkdir -p "$alpha_log_dir"
  echo "========== START M2 alpha=$alpha =========="

  rebuild_arg=()
  if [[ "$first_alpha" == true ]]; then
    rebuild_arg=(--rebuild-dual-order-lmdb)
    first_alpha=false
  fi

  python "$SCRIPT_DIR/run_dual_order_method_suite.py" \
    --root "$ROOT" \
    --log-dir "$alpha_log_dir" \
    --methods m2 \
    --m2-consistency-weight "$alpha" \
    --summary-name "m2_alpha_${tag}_narrow_clean_dev_summary" \
    --ddp-gpus 0,1 \
    --eval-gpu 0 \
    --master-port-base "$((29960 + 10#$tag))" \
    --max-epoch 50 \
    --synthetic-eval-every 5 \
    --target-eval-every 2 \
    --synthetic-patience 4 \
    --target-patience 5 \
    --synthetic-min-epoch 15 \
    --target-min-epoch 10 \
    --min-delta 0.0001 \
    --batch-size-per-card 16 \
    --num-workers 4 \
    --lr 0.000025 \
    --replace \
    "${rebuild_arg[@]}" \
    2>&1 | tee "$alpha_log_dir/run.log"

  echo "========== DONE M2 alpha=$alpha =========="
done

python - "$ROOT" "$LOG_DIR" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

root, log_dir = map(Path, sys.argv[1:])
eval_root = root / "04_model_training" / "eval_reports"
rows = []
for tag, alpha in (("015", 0.15), ("020", 0.20), ("025", 0.25), ("030", 0.30)):
    summary_path = eval_root / f"m2_alpha_{tag}_narrow_clean_dev_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    result = summary["results"]["m2"]
    dev = result["dev"]
    rows.append({
        "alpha": alpha,
        "tag": tag,
        "experiment": result.get("experiment", ""),
        "best_epoch": result["best_epoch"],
        "macro_cer": dev["macro_cer"],
        "macro_wer": dev["macro_wer"],
        "macro_one_minus_ned": sum(
            dev["languages"][language]["one_minus_ned_macro"]
            for language in ("zh", "ug", "kk")
        ) / 3.0,
        "macro_line_accuracy": dev["macro_line_accuracy"],
        "best_checkpoint": result["best_checkpoint"],
        "best_checkpoint_sha256": result["best_checkpoint_sha256"],
        "config": result["config"],
        "config_sha256": result["config_sha256"],
        "summary": str(summary_path),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    })

selected = min(rows, key=lambda row: (row["macro_cer"], row["alpha"]))
output = {
    "status": "M2_ALPHA_NARROW_SWEEP_OK",
    "protocol": {
        "candidate_alphas": [0.15, 0.20, 0.25, 0.30],
        "selection": "argmin_alpha_clean_dev_macro_CER",
        "tie_breaking": "smaller_alpha",
        "primary_metric": "macro_cer",
        "secondary_metrics": [
            "macro_line_accuracy",
            "macro_wer",
            "macro_one_minus_ned",
        ],
        "same_s50": True,
        "same_target_train_dev": True,
        "same_preprocess_protocol": "P1_MSR_V3",
        "test_evaluated": False,
        "corrupted_dev_used": False,
    },
    "selected_alpha_candidate": selected["alpha"],
    "rows": rows,
    "lock_status": "PENDING_PERMANENT_METHOD_LOCK",
    "lock_note": "Do not change M3 or Full until this selected alpha is written to the frozen method protocol.",
}
(log_dir / "m2_alpha_narrow_sweep_summary.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
with (log_dir / "m2_alpha_narrow_sweep_summary.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(output, ensure_ascii=False, indent=2))
PY

printf '%s\n' "M2_ALPHA_NARROW_SWEEP_OK" | tee "$LOG_DIR/COMPLETE"
