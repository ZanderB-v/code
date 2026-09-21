#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
CONDA_ENV="${CONDA_ENV:-openocr_svtrv2}"
SESSION="${SESSION:-pretrained_baselines_v1_preflight}"
DEVICE="${DEVICE:-cuda:0}"
REPLACE=0
if [[ "${1:-}" == "--replace" ]]; then
  REPLACE=1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ "$REPLACE" -eq 1 ]]; then
    tmux kill-session -t "$SESSION"
  else
    echo "Session exists: $SESSION" >&2
    exit 1
  fi
fi

source /home/wudayu/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
python - <<'PY'
import cv2
import importlib_metadata
import mmcv
import mmengine
import timm
import torch
import urllib3
import yapf

assert torch.cuda.is_available(), "CUDA is not available"
print({
    "status": "PRETRAINED_BASELINE_RUNTIME_IMPORTS_OK",
    "torch": torch.__version__,
    "cuda_devices": torch.cuda.device_count(),
    "timm": timm.__version__,
    "opencv": cv2.__version__,
    "mmcv": mmcv.__version__,
    "mmengine": mmengine.__version__,
    "importlib_metadata": importlib_metadata.version("importlib-metadata"),
    "urllib3": urllib3.__version__,
    "yapf": importlib_metadata.version("yapf"),
})
PY

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/Comparison/pretrained_baselines_v1/logs/preflight_$STAMP"
OUTPUT_DIR="$ROOT/Comparison/pretrained_baselines_v1/outputs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR/pretrained_baselines_v1_preflight_summary.json"

cat > "$LOG_DIR/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
trap 'rc=\$?; printf "PRETRAINED_BASELINES_V1_PREFLIGHT_FAILED exit_code=%s\\n" "\$rc" | tee "$LOG_DIR/FAILED"; exit "\$rc"' ERR
source /home/wudayu/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$ROOT"

export PYTHONPATH="$ROOT/Comparison/scripts:$ROOT/Comparison/third_party/parseq_v1_0_0:$ROOT/Comparison/third_party/mmocr_v1_0_1:\${PYTHONPATH:-}"

echo "===== FULL DATA / LABEL / CAPACITY AUDIT ====="
python Comparison/scripts/audit_pretrained_formal_protocol.py \
  --output "$OUTPUT_DIR/formal_protocol_audit.json" \
  2>&1 | tee "$LOG_DIR/formal_protocol_audit.log"

for model in crnn svtr parseq abinet; do
  echo "===== AUDIT \$model ====="
  python Comparison/scripts/audit_pretrained_loading.py \
    --model "\$model" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_DIR/\${model}_audit.log"

  echo "===== SINGLE BATCH \$model ====="
  python Comparison/scripts/preflight_pretrained_baseline.py \
    --model "\$model" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_DIR/\${model}_preflight.log"

  echo "===== TWO-GPU LONG-LINE DDP PREFLIGHT \$model ====="
  CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --standalone \
    --nproc_per_node=2 \
    Comparison/scripts/preflight_formal_baseline_ddp.py \
    --model "\$model" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_DIR/\${model}_ddp_preflight.log"
done

python Comparison/scripts/summarize_pretrained_preflight.py \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_DIR/summary.log"

touch "$LOG_DIR/PASSED"
EOF
chmod +x "$LOG_DIR/run.sh"

tmux new-session -d -s "$SESSION" "bash '$LOG_DIR/run.sh'"
echo "Preflight started: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR"
echo "No formal training is started by this script."
