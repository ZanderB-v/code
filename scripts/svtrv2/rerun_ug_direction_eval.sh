#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
CONDA_SH="${CONDA_SH:-/home/wudayu/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-openocr_svtrv2}"
OPENOCR_ROOT="$ROOT/third_party/OpenOCR"
CONFIG_DIR="$ROOT/04_model_training/configs"
RUN_DIR="$ROOT/04_model_training/runs"
EVAL_DIR="$ROOT/04_model_training/eval_reports"
LOG_DIR="$ROOT/04_model_training/logs/ug_direction_eval_rerun_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$ROOT"

U1_CKPT="$RUN_DIR/svtrv2_s_ug_u1_reverse_synth5k_safe60/best.pth"
U2_CKPT="$RUN_DIR/svtrv2_s_ug_u2_visual_synth5k_safe60/best.pth"

if [[ ! -s "$U1_CKPT" ]]; then
  echo "Missing U1 checkpoint: $U1_CKPT" >&2
  exit 1
fi
if [[ ! -s "$U2_CKPT" ]]; then
  echo "Missing U2 checkpoint: $U2_CKPT" >&2
  exit 1
fi

echo "Running U1 eval with final best checkpoint..."
CUDA_VISIBLE_DEVICES=0 python scripts/svtrv2/infer_label_file_metrics.py \
  --openocr-root "$OPENOCR_ROOT" \
  --config "$CONFIG_DIR/svtrv2_s_ug_u1_reverse_real_pilot_eval_safe60.yml" \
  --checkpoint "$U1_CKPT" \
  --data-dir "$ROOT/01_data_preparation/real_line_dataset_eval_reviewed" \
  --label-file "$ROOT/01_data_preparation/real_line_dataset_eval_reviewed/labels_ug_direction/real_pilot_ug_logical.txt" \
  --metadata-csv "$ROOT/01_data_preparation/real_line_dataset_eval_reviewed/metadata.csv" \
  --output-dir "$EVAL_DIR/ug_u1_reverse_real_pilot_safe60_rerun" \
  2>&1 | tee "$LOG_DIR/u1_eval.log"

echo "Running U2 eval with final best checkpoint..."
CUDA_VISIBLE_DEVICES=0 python scripts/svtrv2/infer_label_file_metrics.py \
  --openocr-root "$OPENOCR_ROOT" \
  --config "$CONFIG_DIR/svtrv2_s_ug_u2_visual_real_pilot_eval_safe60.yml" \
  --checkpoint "$U2_CKPT" \
  --data-dir "$ROOT/01_data_preparation/real_line_dataset_eval_reviewed" \
  --label-file "$ROOT/01_data_preparation/real_line_dataset_eval_reviewed/labels_ug_direction/real_pilot_ug_logical.txt" \
  --metadata-csv "$ROOT/01_data_preparation/real_line_dataset_eval_reviewed/metadata.csv" \
  --gt-transform ug_logical_to_visual \
  --output-dir "$EVAL_DIR/ug_u2_visual_real_pilot_safe60_rerun" \
  2>&1 | tee "$LOG_DIR/u2_eval.log"

echo "DONE"
echo "$EVAL_DIR/ug_u1_reverse_real_pilot_safe60_rerun/metrics_summary.json"
echo "$EVAL_DIR/ug_u2_visual_real_pilot_safe60_rerun/metrics_summary.json"
