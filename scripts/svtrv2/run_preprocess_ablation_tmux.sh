#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${ALLOW_LEGACY_EXPERIMENT:-0}" != "1" ]]; then
  echo "LEGACY ENTRYPOINT DISABLED: P1 was selected; use the formal P1/MSR V3 runners." >&2
  echo "Set ALLOW_LEGACY_EXPERIMENT=1 only to reproduce the historical ablation." >&2
  exit 2
fi

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
CONDA_SH="${CONDA_SH:-/home/wudayu/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-openocr_svtrv2}"
OPENOCR_ROOT="$ROOT/third_party/OpenOCR"
PILOT_DIR="${PILOT_DIR:-$ROOT/03_synthetic_generation/preprocess_pilot_v1/synthetic_shard_0000_parallel}"
DATASET_DIR="$ROOT/04_model_training/datasets/preprocess_ablation_v1"
LABELS_DIR="$DATASET_DIR/labels"
TARGET_DIR="$ROOT/01_data_preparation/real_line_dataset_eval_reviewed"
CONFIG_DIR="$ROOT/04_model_training/configs"
RUNS_DIR="$ROOT/04_model_training/runs"
EVAL_ROOT="$ROOT/04_model_training/eval_reports"
LOG_ROOT="$ROOT/04_model_training/logs"
SESSION="${SESSION:-preprocess_ablation_v1}"
DDP_GPUS="${DDP_GPUS:-0,1}"
EVAL_GPU="${EVAL_GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-8}"
EVAL_EVERY="${EVAL_EVERY:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29551}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-0.00005}"
MAX_RATIO="${MAX_RATIO:-40}"

REPLACE=0
if [[ "${1:-}" == "--replace" ]]; then
  REPLACE=1
elif [[ -n "${1:-}" && "${1:-}" != "--worker" ]]; then
  echo "Usage: bash scripts/svtrv2/run_preprocess_ablation_tmux.sh [--replace]" >&2
  exit 2
fi

if [[ "${PREPROCESS_ABLATION_WORKER:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    if [[ "$REPLACE" == "1" ]]; then
      tmux kill-session -t "$SESSION"
    else
      echo "tmux session already exists: $SESSION" >&2
      exit 1
    fi
  fi
  LOG_DIR="$LOG_ROOT/preprocess_ablation_v1_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$LOG_DIR"
  tmux new-session -d -s "$SESSION" \
    "cd '$ROOT' && PREPROCESS_ABLATION_WORKER=1 LOG_DIR='$LOG_DIR' ROOT='$ROOT' CONDA_SH='$CONDA_SH' CONDA_ENV='$CONDA_ENV' PILOT_DIR='$PILOT_DIR' DDP_GPUS='$DDP_GPUS' EVAL_GPU='$EVAL_GPU' MAX_EPOCH='$MAX_EPOCH' EVAL_EVERY='$EVAL_EVERY' MASTER_PORT_BASE='$MASTER_PORT_BASE' NUM_WORKERS='$NUM_WORKERS' LR='$LR' MAX_RATIO='$MAX_RATIO' bash '$ROOT/scripts/svtrv2/run_preprocess_ablation_tmux.sh' --worker"
  echo "Preprocessing ablation started."
  echo "Attach: tmux attach -t $SESSION"
  echo "Log directory: $LOG_DIR"
  exit 0
fi

mkdir -p "$LOG_DIR" "$EVAL_ROOT" "$RUNS_DIR"
exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$ROOT"

IFS=',' read -r -a GPU_IDS <<< "$DDP_GPUS"
NPROC_PER_NODE="${#GPU_IDS[@]}"
if (( NPROC_PER_NODE < 1 )); then
  echo "DDP_GPUS must contain at least one GPU." >&2
  exit 2
fi

echo "[1/4] Preparing identical P0/P1 datasets, labels, LMDB, and configs..."
python scripts/svtrv2/prepare_preprocess_ablation.py \
  --root "$ROOT" \
  --pilot-dir "$PILOT_DIR" \
  --max-epoch "$MAX_EPOCH" \
  --eval-every "$EVAL_EVERY" \
  --p0-batch-size-per-card 16 \
  --p1-first-batch-size-per-card 32 \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --max-ratio "$MAX_RATIO" \
  --replace-lmdb \
  2>&1 | tee "$LOG_DIR/prepare.log"

monitor_gpu() {
  local output="$1"
  (
    echo "timestamp,gpu_index,memory_used_mib,utilization_gpu_percent"
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits || true
      sleep 10
    done
  ) > "$output" 2>&1 &
  MONITOR_PID=$!
}

evaluate_and_select() {
  local mode="$1"
  local config="$2"
  local run_dir="$3"
  local report_prefix="$4"
  local best_ckpt="$run_dir/best_target_dev_macro_cer.pth"
  local best_json="$run_dir/best_target_dev_macro_cer.json"
  local best_cer="999999"
  local best_epoch="0"

  for epoch in $(seq "$EVAL_EVERY" "$EVAL_EVERY" "$MAX_EPOCH"); do
    local checkpoint="$run_dir/epoch_${epoch}.pth"
    if [[ ! -s "$checkpoint" ]]; then
      echo "Missing expected checkpoint: $checkpoint" >&2
      exit 1
    fi
    local epoch_pad
    epoch_pad="$(printf '%04d' "$epoch")"
    local output="$EVAL_ROOT/${report_prefix}_dev_epoch_${epoch_pad}"
    rm -rf "$output"
    python scripts/svtrv2/evaluate_d2_checkpoint.py \
      --root "$ROOT" \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --split dev \
      --data-dir "$TARGET_DIR" \
      --labels-dir "$LABELS_DIR" \
      --label-prefix target \
      --output-dir "$output" \
      --device-id "$EVAL_GPU" \
      --batch-size 32 \
      --require-python-bidi \
      2>&1 | tee "$LOG_DIR/${mode}_eval_epoch_${epoch_pad}.log"
    local current_cer
    current_cer="$(python -c "import json; print(json.load(open('$output/metrics_macro_summary.json', encoding='utf-8'))['macro_cer'])")"
    local improved
    improved="$(python -c "print(1 if float('$current_cer') < float('$best_cer') else 0)")"
    if [[ "$improved" == "1" ]]; then
      best_cer="$current_cer"
      best_epoch="$epoch"
      cp "$checkpoint" "$best_ckpt"
      python -c "import json,pathlib; pathlib.Path('$best_json').write_text(json.dumps({'mode':'$mode','epoch':$best_epoch,'target_dev_macro_cer':float('$best_cer'),'source_checkpoint':'$checkpoint','checkpoint':'$best_ckpt','selection_metric':'target_dev_macro_CER'},ensure_ascii=False,indent=2),encoding='utf-8')"
    fi
  done

  local best_report="$EVAL_ROOT/${report_prefix}_best_dev"
  rm -rf "$best_report"
  python scripts/svtrv2/evaluate_d2_checkpoint.py \
    --root "$ROOT" \
    --config "$config" \
    --checkpoint "$best_ckpt" \
    --split dev \
    --data-dir "$TARGET_DIR" \
    --labels-dir "$LABELS_DIR" \
    --label-prefix target \
    --output-dir "$best_report" \
    --device-id "$EVAL_GPU" \
    --batch-size 32 \
    --require-python-bidi \
    2>&1 | tee "$LOG_DIR/${mode}_best_dev.log"
  echo "$mode best_epoch=$best_epoch target_dev_macro_CER=$best_cer"
}

train_one() {
  local mode="$1"
  local config="$2"
  local run_dir="$3"
  local report_prefix="$4"
  local master_port="$5"
  local gpu_csv="$LOG_DIR/${mode}_gpu.csv"
  local train_log="$LOG_DIR/${mode}_train.log"

  rm -rf "$run_dir"
  mkdir -p "$run_dir"
  local MONITOR_PID
  monitor_gpu "$gpu_csv"
  set +e
  (
    cd "$OPENOCR_ROOT"
    CUDA_VISIBLE_DEVICES="$DDP_GPUS" python -m torch.distributed.launch \
      --nproc_per_node="$NPROC_PER_NODE" \
      --master_port="$master_port" \
      tools/train_rec.py \
      -c "$config"
  ) 2>&1 | tee "$train_log"
  local status=${PIPESTATUS[0]}
  set -e
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  if (( status != 0 )); then
    echo "$mode training failed with code $status" >&2
    exit "$status"
  fi
  evaluate_and_select "$mode" "$config" "$run_dir" "$report_prefix"
}

P0_CONFIG="$CONFIG_DIR/svtrv2_s_preprocess_p0_fixed.yml"
P1_CONFIG="$CONFIG_DIR/svtrv2_s_preprocess_p1_msr.yml"
P0_RUN="$RUNS_DIR/svtrv2_s_preprocess_p0_fixed"
P1_RUN="$RUNS_DIR/svtrv2_s_preprocess_p1_msr"

echo "[2/4] Training and evaluating P0 fixed 48x640..."
train_one \
  "p0_fixed" \
  "$P0_CONFIG" \
  "$P0_RUN" \
  "preprocess_ablation_p0" \
  "$MASTER_PORT_BASE"

echo "[3/4] Training and evaluating P1 official-style MSR..."
train_one \
  "p1_msr" \
  "$P1_CONFIG" \
  "$P1_RUN" \
  "preprocess_ablation_p1" \
  "$((MASTER_PORT_BASE + 1))"

echo "[4/4] Summarizing clean dev, length/aspect buckets, GPU memory, and throughput..."
python scripts/svtrv2/summarize_preprocess_ablation.py \
  --root "$ROOT" \
  --p0-report "$EVAL_ROOT/preprocess_ablation_p0_best_dev" \
  --p1-report "$EVAL_ROOT/preprocess_ablation_p1_best_dev" \
  --p0-log "$LOG_DIR/p0_fixed_train.log" \
  --p1-log "$LOG_DIR/p1_msr_train.log" \
  --p0-gpu-csv "$LOG_DIR/p0_fixed_gpu.csv" \
  --p1-gpu-csv "$LOG_DIR/p1_msr_gpu.csv" \
  --output "$EVAL_ROOT/preprocess_ablation_v1_summary.json" \
  2>&1 | tee "$LOG_DIR/summary.log"

echo "DONE"
echo "Summary: $EVAL_ROOT/preprocess_ablation_v1_summary.json"
echo "No target test evaluation was run."
