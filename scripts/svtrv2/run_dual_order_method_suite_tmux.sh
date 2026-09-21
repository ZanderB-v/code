#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition
SESSION=dual_order_method_suite
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$ROOT/04_model_training/logs/dual_order_method_suite_$STAMP"
EXTRA_ARGS=("$@")

tmux kill-session -t "$SESSION" 2>/dev/null || true
mkdir -p "$LOG_DIR"
printf '%s\n' "$LOG_DIR" > \
  "$ROOT/04_model_training/logs/dual_order_method_suite_latest.txt"

source /home/wudayu/anaconda3/etc/profile.d/conda.sh
conda activate openocr_svtrv2
python "$ROOT/scripts/svtrv2/preflight_p1_formal.py" \
  --root "$ROOT" \
  --minimum-gpus 2 \
  --output "$LOG_DIR/formal_preflight.json" \
  2>&1 | tee "$LOG_DIR/formal_preflight.log"
python "$ROOT/scripts/svtrv2/test_dual_order_components.py" \
  --root "$ROOT" \
  --output "$LOG_DIR/component_preflight.json" \
  2>&1 | tee "$LOG_DIR/component_preflight.log"

cat > "$LOG_DIR/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source /home/wudayu/anaconda3/etc/profile.d/conda.sh
conda activate openocr_svtrv2
cd "$ROOT"
export PYTHONUNBUFFERED=1
python scripts/svtrv2/run_dual_order_method_suite.py \\
  --root "$ROOT" \\
  --log-dir "$LOG_DIR" \\
  --ddp-gpus 0,1 \\
  --eval-gpu 0 \\
  --max-epoch 50 \\
  --synthetic-eval-every 5 \\
  --target-eval-every 2 \\
  --synthetic-patience 4 \\
  --target-patience 5 \\
  --synthetic-min-epoch 15 \\
  --target-min-epoch 10 \\
  ${EXTRA_ARGS[*]} \\
  2>&1 | tee "$LOG_DIR/suite.log"
EOF
chmod +x "$LOG_DIR/run.sh"

tmux new-session -d -s "$SESSION" "bash '$LOG_DIR/run.sh'"
tmux set-window-option -t "$SESSION":0 remain-on-exit on
echo "Dual-order method suite started."
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR/suite.log"
echo "Follow: tail -F $LOG_DIR/suite.log"
echo "Latest-log pointer: $ROOT/04_model_training/logs/dual_order_method_suite_latest.txt"
