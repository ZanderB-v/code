#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
CONDA_SH="${CONDA_SH:-/home/wudayu/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-openocr_svtrv2}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/04_model_training/logs/formal_prelaunch_audit_$STAMP"

mkdir -p "$LOG_DIR"
printf '%s\n' "$LOG_DIR" > \
  "$ROOT/04_model_training/logs/formal_prelaunch_audit_latest.txt"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$ROOT"

if pgrep -af 'tools/train_rec.py|run_p1_rctc_stage.py|run_b1_full_svtrv2.py|run_dual_order_method_suite.py' \
  > "$LOG_DIR/active_training_processes.txt"; then
  echo "Active formal training process detected; stop it before prelaunch audit:" >&2
  cat "$LOG_DIR/active_training_processes.txt" >&2
  exit 3
fi

bash -n \
  scripts/svtrv2/run_rctc_baseline_chain_tmux.sh \
  scripts/svtrv2/run_rctc_scale_ablation_tmux.sh \
  scripts/svtrv2/run_b1_full_svtrv2_tmux.sh \
  scripts/svtrv2/run_dual_order_method_suite_tmux.sh \
  scripts/corruption/run_corruption_dev_eval_tmux.sh

mapfile -t PYTHON_FILES < <(
  find scripts/svtrv2 scripts/corruption -maxdepth 1 -type f -name '*.py' -print
)
PYTHON_FILES+=(
  third_party/OpenOCR/tools/data/__init__.py
  third_party/OpenOCR/tools/data/ratio_sampler.py
  third_party/OpenOCR/openrec/losses/__init__.py
  third_party/OpenOCR/openrec/losses/dual_order_gtc_loss.py
  third_party/OpenOCR/openrec/modeling/decoders/__init__.py
  third_party/OpenOCR/openrec/modeling/decoders/dual_order_gtc_decoder.py
  third_party/OpenOCR/openrec/preprocess/__init__.py
  third_party/OpenOCR/openrec/preprocess/dual_order_gtc_label_encode.py
)
python -m py_compile "${PYTHON_FILES[@]}"

python scripts/protocol/freeze_protocol_v2.py \
  --root "$ROOT" \
  --mode verify \
  --workers 8 \
  2>&1 | tee "$LOG_DIR/protocol_v2_verify.log"

python scripts/svtrv2/preflight_p1_formal.py \
  --root "$ROOT" \
  --minimum-gpus 2 \
  --output "$LOG_DIR/formal_preflight.json" \
  2>&1 | tee "$LOG_DIR/formal_preflight.log"

python scripts/svtrv2/test_dual_order_components.py \
  --root "$ROOT" \
  --output "$LOG_DIR/dual_order_components.json" \
  2>&1 | tee "$LOG_DIR/dual_order_components.log"

echo "FORMAL_PRELAUNCH_AUDIT_OK"
echo "log_dir=$LOG_DIR"
