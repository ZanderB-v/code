#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_LEGACY_EXPERIMENT:-0}" != "1" ]]; then
  echo "LEGACY ENTRYPOINT DISABLED: U2 is frozen; use the formal P1/MSR V3 runners." >&2
  echo "Set ALLOW_LEGACY_EXPERIMENT=1 only to reproduce the historical U1/U2 pilot." >&2
  exit 2
fi

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
CONDA_SH="${CONDA_SH:-/home/wudayu/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-openocr_svtrv2}"
OPENOCR_ROOT="$ROOT/third_party/OpenOCR"
CONFIG_DIR="$ROOT/04_model_training/configs"
RUN_DIR="$ROOT/04_model_training/runs"
EVAL_DIR="$ROOT/04_model_training/eval_reports"
LOG_DIR="$ROOT/04_model_training/logs/ug_direction_$(date +%Y%m%d_%H%M%S)"

U1_SESSION="${U1_SESSION:-ug_dir_u1}"
U2_SESSION="${U2_SESSION:-ug_dir_u2}"
EVAL_SESSION="${EVAL_SESSION:-ug_dir_eval}"

usage() {
  cat <<EOF
Usage:
  bash scripts/svtrv2/run_ug_direction_tmux.sh [--replace]

Environment overrides:
  ROOT=/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition
  CONDA_SH=/home/wudayu/anaconda3/etc/profile.d/conda.sh
  CONDA_ENV=openocr_svtrv2
EOF
}

REPLACE=0
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
elif [[ "${1:-}" == "--replace" ]]; then
  REPLACE=1
elif [[ -n "${1:-}" ]]; then
  echo "Unknown argument: $1" >&2
  usage
  exit 2
fi

for session in "$U1_SESSION" "$U2_SESSION" "$EVAL_SESSION"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    if [[ "$REPLACE" == "1" ]]; then
      tmux kill-session -t "$session"
    else
      echo "tmux session already exists: $session"
      echo "Use --replace to kill and restart these sessions."
      exit 1
    fi
  fi
done

mkdir -p "$LOG_DIR" "$EVAL_DIR"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$ROOT"

echo "[1/4] Preparing Uyghur U1/U2 labels..."
python scripts/svtrv2/prepare_ug_direction_labels.py \
  --synthetic-dir "$ROOT/03_synthetic_generation/synthetic_closure_synth5k_v1" \
  --real-dir "$ROOT/01_data_preparation/real_line_dataset_eval_reviewed" \
  --synthetic-val-count 500 \
  --real-pilot-count 150 \
  --max-label-len 60 \
  --seed 20260724 | tee "$LOG_DIR/prepare_labels.log"

echo "[2/4] Starting U1 training in tmux session: $U1_SESSION"
tmux new-session -d -s "$U1_SESSION" "bash -lc '
set -euo pipefail
source \"$CONDA_SH\"
conda activate \"$CONDA_ENV\"
cd \"$OPENOCR_ROOT\"
CUDA_VISIBLE_DEVICES=0 python tools/train_rec.py \
  -c \"$CONFIG_DIR/svtrv2_s_ug_u1_reverse_synth5k_safe60.yml\" \
  2>&1 | tee \"$LOG_DIR/u1_train.log\"
'"

echo "[3/4] Starting U2 training in tmux session: $U2_SESSION"
tmux new-session -d -s "$U2_SESSION" "bash -lc '
set -euo pipefail
source \"$CONDA_SH\"
conda activate \"$CONDA_ENV\"
cd \"$OPENOCR_ROOT\"
CUDA_VISIBLE_DEVICES=1 python tools/train_rec.py \
  -c \"$CONFIG_DIR/svtrv2_s_ug_u2_visual_synth5k_safe60.yml\" \
  2>&1 | tee \"$LOG_DIR/u2_train.log\"
'"

echo "[4/4] Starting auto-eval watcher in tmux session: $EVAL_SESSION"
tmux new-session -d -s "$EVAL_SESSION" "bash -lc '
set -euo pipefail
source \"$CONDA_SH\"
conda activate \"$CONDA_ENV\"
cd \"$ROOT\"

U1_CKPT=\"$RUN_DIR/svtrv2_s_ug_u1_reverse_synth5k_safe60/best.pth\"
U2_CKPT=\"$RUN_DIR/svtrv2_s_ug_u2_visual_synth5k_safe60/best.pth\"

echo \"Waiting for U1 training session to finish: $U1_SESSION\"
while tmux has-session -t \"$U1_SESSION\" 2>/dev/null; do sleep 60; done
echo \"Waiting for U1 checkpoint: \$U1_CKPT\"
until [[ -s \"\$U1_CKPT\" ]]; do sleep 10; done
echo \"U1 checkpoint found. Running U1 real-pilot eval...\"
CUDA_VISIBLE_DEVICES=0 python scripts/svtrv2/infer_label_file_metrics.py \
  --openocr-root \"$OPENOCR_ROOT\" \
  --config \"$CONFIG_DIR/svtrv2_s_ug_u1_reverse_real_pilot_eval_safe60.yml\" \
  --checkpoint \"\$U1_CKPT\" \
  --data-dir \"$ROOT/01_data_preparation/real_line_dataset_eval_reviewed\" \
  --label-file \"$ROOT/01_data_preparation/real_line_dataset_eval_reviewed/labels_ug_direction/real_pilot_ug_logical.txt\" \
  --metadata-csv \"$ROOT/01_data_preparation/real_line_dataset_eval_reviewed/metadata.csv\" \
  --output-dir \"$EVAL_DIR/ug_u1_reverse_real_pilot_safe60\" \
  2>&1 | tee \"$LOG_DIR/u1_eval.log\"

echo \"Waiting for U2 training session to finish: $U2_SESSION\"
while tmux has-session -t \"$U2_SESSION\" 2>/dev/null; do sleep 60; done
echo \"Waiting for U2 checkpoint: \$U2_CKPT\"
until [[ -s \"\$U2_CKPT\" ]]; do sleep 10; done
echo \"U2 checkpoint found. Running U2 real-pilot eval...\"
CUDA_VISIBLE_DEVICES=1 python scripts/svtrv2/infer_label_file_metrics.py \
  --openocr-root \"$OPENOCR_ROOT\" \
  --config \"$CONFIG_DIR/svtrv2_s_ug_u2_visual_real_pilot_eval_safe60.yml\" \
  --checkpoint \"\$U2_CKPT\" \
  --data-dir \"$ROOT/01_data_preparation/real_line_dataset_eval_reviewed\" \
  --label-file \"$ROOT/01_data_preparation/real_line_dataset_eval_reviewed/labels_ug_direction/real_pilot_ug_logical.txt\" \
  --metadata-csv \"$ROOT/01_data_preparation/real_line_dataset_eval_reviewed/metadata.csv\" \
  --gt-transform ug_logical_to_visual \
  --output-dir \"$EVAL_DIR/ug_u2_visual_real_pilot_safe60\" \
  2>&1 | tee \"$LOG_DIR/u2_eval.log\"

echo \"DONE. Eval reports:\"
echo \"$EVAL_DIR/ug_u1_reverse_real_pilot_safe60/metrics_summary.json\"
echo \"$EVAL_DIR/ug_u2_visual_real_pilot_safe60/metrics_summary.json\"
'"

echo
echo "Started:"
echo "  tmux attach -t $U1_SESSION"
echo "  tmux attach -t $U2_SESSION"
echo "  tmux attach -t $EVAL_SESSION"
echo
echo "Logs:"
echo "  $LOG_DIR"
echo
echo "Final reports will be:"
echo "  $EVAL_DIR/ug_u1_reverse_real_pilot_safe60/metrics_summary.json"
echo "  $EVAL_DIR/ug_u2_visual_real_pilot_safe60/metrics_summary.json"
