#!/usr/bin/env bash
set -euo pipefail

ROOT="/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition"
PROTOCOL_ROOT="${CORRUPTION_PROTOCOL_ROOT:-$ROOT/05_evaluation/corruption_protocol_v1}"
SESSION="${CORRUPTION_EVAL_SESSION:-corruption_dev_v1_eval}"
REPLACE_FLAG=""

for arg in "$@"; do
  if [[ "$arg" == "--replace" ]]; then
    REPLACE_FLAG="--replace"
  else
    echo "Usage: bash scripts/corruption/run_corruption_dev_eval_tmux.sh [--replace]"
    exit 2
  fi
done

python - "$PROTOCOL_ROOT" "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

protocol_root = Path(sys.argv[1])
root = Path(sys.argv[2])
required = [
    protocol_root / "corruption_protocol_v1.json",
    protocol_root / "freeze_report.json",
    root / "04_model_training/runs/svtrv2_s_e0_random_target_only/best_clean_dev_macro_cer.pth",
    root / "04_model_training/runs/svtrv2_s_e1_target_only/best_clean_dev_macro_cer.pth",
    root / "04_model_training/runs/svtrv2_s_e5_d2_to_target/best_clean_dev_macro_cer.pth",
    root / "04_model_training/runs/svtrv2_s_b1_full_s50_to_target/best_clean_dev_macro_cer.pth",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("Missing required files:\n" + "\n".join(missing))
report = json.loads((protocol_root / "freeze_report.json").read_text(encoding="utf-8-sig"))
if report.get("status") != "passed" or report.get("test_inference_run") is not False:
    raise SystemExit(f"Invalid freeze report: {report}")
print("CORRUPTION_EVAL_PREFLIGHT_OK")
PY

mkdir -p "$PROTOCOL_ROOT/logs" "$PROTOCOL_ROOT/eval_reports"
tmux kill-session -t "$SESSION" 2>/dev/null || true
rm -f "$PROTOCOL_ROOT/logs/gpu0.exit" "$PROTOCOL_ROOT/logs/gpu1.exit"

if [[ -n "$REPLACE_FLAG" ]]; then
  rm -rf \
    "$PROTOCOL_ROOT/eval_reports/e0" \
    "$PROTOCOL_ROOT/eval_reports/e1" \
    "$PROTOCOL_ROOT/eval_reports/b0" \
    "$PROTOCOL_ROOT/eval_reports/b1"
  rm -f \
    "$PROTOCOL_ROOT/eval_reports/invocation_e0_b0.json" \
    "$PROTOCOL_ROOT/eval_reports/invocation_e1_b1.json" \
    "$PROTOCOL_ROOT/eval_reports/clean_corrupted_dev_comparison.json" \
    "$PROTOCOL_ROOT/eval_reports/clean_corrupted_dev_comparison.csv"
fi

COMMON="source /home/wudayu/anaconda3/etc/profile.d/conda.sh && conda activate openocr_svtrv2 && cd '$ROOT'"

tmux new-session -d -s "$SESSION" -n gpu0 \
  "$COMMON && \
   CUDA_VISIBLE_DEVICES=0 python scripts/corruption/evaluate_corruption_dev.py \
     --root '$ROOT' \
     --protocol-root '$PROTOCOL_ROOT' \
     --models e0 b0 \
     --device-id 0 \
     --batch-size 1 \
     $REPLACE_FLAG \
     2>&1 | tee '$PROTOCOL_ROOT/logs/eval_gpu0_e0_b0.log'; \
   status=\${PIPESTATUS[0]}; \
   printf '%s\n' \"\$status\" > '$PROTOCOL_ROOT/logs/gpu0.exit'; \
   echo GPU0_EVAL_EXIT_CODE=\$status; \
   exec bash"

tmux new-window -t "$SESSION" -n gpu1 \
  "$COMMON && \
   CUDA_VISIBLE_DEVICES=1 python scripts/corruption/evaluate_corruption_dev.py \
     --root '$ROOT' \
     --protocol-root '$PROTOCOL_ROOT' \
     --models e1 b1 \
     --device-id 0 \
     --batch-size 1 \
     $REPLACE_FLAG \
     2>&1 | tee '$PROTOCOL_ROOT/logs/eval_gpu1_e1_b1.log'; \
   status=\${PIPESTATUS[0]}; \
   printf '%s\n' \"\$status\" > '$PROTOCOL_ROOT/logs/gpu1.exit'; \
   echo GPU1_EVAL_EXIT_CODE=\$status; \
   exec bash"

tmux new-window -t "$SESSION" -n summary \
  "$COMMON && \
   while true; do \
     for status_file in gpu0.exit gpu1.exit; do \
       if [[ -f '$PROTOCOL_ROOT/logs/'\$status_file ]]; then \
         status=\$(cat '$PROTOCOL_ROOT/logs/'\$status_file); \
         if [[ \$status -ne 0 ]]; then \
           echo \"corruption evaluator failed: \$status_file=\$status\" >&2; \
           exit \$status; \
         fi; \
       fi; \
     done; \
     complete=1; \
     for model in e0 e1 b0 b1; do \
       test -f '$PROTOCOL_ROOT/eval_reports/'\$model'/model_summary.json' || complete=0; \
     done; \
     if [[ \$complete -eq 1 ]]; then break; fi; \
     echo \"waiting for e0/e1/b0/b1 summaries: \$(date)\"; \
     sleep 60; \
   done; \
   python scripts/corruption/summarize_corruption_dev.py \
     --root '$ROOT' \
     --protocol-root '$PROTOCOL_ROOT' \
     --models e0 e1 b0 b1 \
     2>&1 | tee '$PROTOCOL_ROOT/logs/summarize.log'; \
   status=\${PIPESTATUS[0]}; \
   echo CORRUPTION_SUMMARY_EXIT_CODE=\$status; \
   exec bash"

tmux select-window -t "$SESSION:gpu0"
echo "Started: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Windows: gpu0 (E0+B0), gpu1 (E1+B1), summary"
echo "Test is not evaluated by this workflow."
