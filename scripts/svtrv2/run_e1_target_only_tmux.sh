#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_LEGACY_EXPERIMENT:-0}" != "1" ]]; then
  echo "LEGACY ENTRYPOINT DISABLED: use run_rctc_baseline_chain_tmux.sh." >&2
  echo "Set ALLOW_LEGACY_EXPERIMENT=1 only to reproduce historical invalidated runs." >&2
  exit 2
fi

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
CONDA_SH="${CONDA_SH:-/home/wudayu/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-openocr_svtrv2}"
OPENOCR_ROOT="$ROOT/third_party/OpenOCR"
CONFIG_PATH="$ROOT/04_model_training/configs/svtrv2_s_e1_target_only.yml"
DATASET_DIR="$ROOT/04_model_training/datasets/e1_target_only"
LABELS_DIR="$DATASET_DIR/labels"
TARGET_DIR="$ROOT/01_data_preparation/real_line_dataset_eval_reviewed"
RUN_MODEL_DIR="$ROOT/04_model_training/runs/svtrv2_s_e1_target_only"
EVAL_ROOT="$ROOT/04_model_training/eval_reports"
LOG_DIR="$ROOT/04_model_training/logs/e1_target_only_$(date +%Y%m%d_%H%M%S)"

TRAIN_SESSION="${TRAIN_SESSION:-e1_target_only_train}"
WATCH_SESSION="${WATCH_SESSION:-e1_target_only_watch_eval}"
DDP_GPUS="${DDP_GPUS:-0,1}"
EVAL_GPU="${EVAL_GPU:-1}"
MASTER_PORT="${MASTER_PORT:-29531}"
MAX_EPOCH="${MAX_EPOCH:-50}"
EVAL_EVERY="${EVAL_EVERY:-2}"
PATIENCE_EVALS="${PATIENCE_EVALS:-5}"
MIN_EPOCH="${MIN_EPOCH:-10}"
MIN_DELTA="${MIN_DELTA:-0.0001}"
BATCH_SIZE_PER_CARD="${BATCH_SIZE_PER_CARD:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-0.00005}"

usage() {
  cat <<EOF
Usage:
  bash scripts/svtrv2/run_e1_target_only_tmux.sh [--replace]

E1 trains SVTRv2-S only on the reviewed target-domain train split:
  zh: natural Chinese meme lines
  ug: target-domain re-rendered Uyghur meme lines
  kk: target-domain re-rendered Kazakh meme lines

Defaults:
  DDP_GPUS=0,1
  MAX_EPOCH=50
  EVAL_EVERY=2
  PATIENCE_EVALS=5
  MIN_EPOCH=10
  BATCH_SIZE_PER_CARD=16
  LR=0.00005
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

IFS=',' read -r -a GPU_IDS <<< "$DDP_GPUS"
NPROC_PER_NODE="${#GPU_IDS[@]}"
if [[ "$NPROC_PER_NODE" -lt 1 ]]; then
  echo "DDP_GPUS must contain at least one GPU id." >&2
  exit 2
fi

for session in "$TRAIN_SESSION" "$WATCH_SESSION"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    if [[ "$REPLACE" == "1" ]]; then
      tmux kill-session -t "$session"
    else
      echo "tmux session already exists: $session" >&2
      echo "Use --replace to kill and restart E1 sessions." >&2
      exit 1
    fi
  fi
done

mkdir -p "$LOG_DIR" "$EVAL_ROOT" "$RUN_MODEL_DIR"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$ROOT"

echo "[1/3] Preparing E1 target-domain labels and config..."
python scripts/svtrv2/prepare_e1_target_only.py \
  --root "$ROOT" \
  --max-epoch "$MAX_EPOCH" \
  --batch-size-per-card "$BATCH_SIZE_PER_CARD" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --internal-eval-every "$EVAL_EVERY" \
  2>&1 | tee "$LOG_DIR/prepare.log"

cat > "$LOG_DIR/train_ddp.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$OPENOCR_ROOT"

CUDA_VISIBLE_DEVICES="$DDP_GPUS" python -m torch.distributed.launch \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  tools/train_rec.py \
  -c "$CONFIG_PATH" \
  2>&1 | tee "$LOG_DIR/train.log"
EOF
chmod +x "$LOG_DIR/train_ddp.sh"

cat > "$LOG_DIR/watch_eval.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$ROOT"

BEST_CKPT="$RUN_MODEL_DIR/best_target_dev_macro_cer.pth"
BEST_JSON="$RUN_MODEL_DIR/best_target_dev_macro_cer.json"
WATCH_JSONL="$LOG_DIR/target_dev_epoch_metrics.jsonl"
BEST_CER=999999.0
BEST_EPOCH=0
BAD_EVALS=0

checkpoint_ready() {
  local ckpt="\$1"
  local previous_size=-1
  local current_size=0
  while true; do
    if [[ -s "\$ckpt" ]]; then
      current_size=\$(stat -c %s "\$ckpt")
      if [[ "\$current_size" -eq "\$previous_size" ]]; then
        return 0
      fi
      previous_size="\$current_size"
    fi
    if ! tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
      [[ -s "\$ckpt" ]] && return 0
      return 1
    fi
    sleep 5
  done
}

echo "Watching checkpoints in: $RUN_MODEL_DIR"
echo "Selection metric: target_dev_macro_CER"
echo "External evaluation every $EVAL_EVERY epochs"

for EPOCH in \$(seq "$EVAL_EVERY" "$EVAL_EVERY" "$MAX_EPOCH"); do
  CKPT="$RUN_MODEL_DIR/epoch_\${EPOCH}.pth"
  echo "Waiting for \$CKPT"
  if ! checkpoint_ready "\$CKPT"; then
    echo "Training ended before epoch \$EPOCH checkpoint was produced."
    break
  fi

  EPOCH_PAD=\$(printf "%04d" "\$EPOCH")
  OUT_DIR="$EVAL_ROOT/e1_target_only_dev_epoch_\${EPOCH_PAD}"
  python scripts/svtrv2/evaluate_d2_checkpoint.py \
    --root "$ROOT" \
    --config "$CONFIG_PATH" \
    --checkpoint "\$CKPT" \
    --split dev \
    --data-dir "$TARGET_DIR" \
    --labels-dir "$LABELS_DIR" \
    --label-prefix target \
    --output-dir "\$OUT_DIR" \
    --device-id "$EVAL_GPU" \
    --batch-size 32 \
    --require-python-bidi \
    2>&1 | tee "$LOG_DIR/eval_epoch_\${EPOCH_PAD}.log"

  CUR_CER=\$(python -c "import json; from pathlib import Path; p=Path('\$OUT_DIR')/'metrics_macro_summary.json'; print(json.load(p.open(encoding='utf-8'))['macro_cer'])")
  echo "{\"epoch\": \$EPOCH, \"target_dev_macro_cer\": \$CUR_CER, \"checkpoint\": \"\$CKPT\", \"report\": \"\$OUT_DIR\"}" >> "\$WATCH_JSONL"

  IMPROVED=\$(python -c "import sys; cur=float(sys.argv[1]); best=float(sys.argv[2]); delta=float(sys.argv[3]); print(1 if cur < best - delta else 0)" "\$CUR_CER" "\$BEST_CER" "$MIN_DELTA")
  if [[ "\$IMPROVED" == "1" ]]; then
    BEST_CER="\$CUR_CER"
    BEST_EPOCH="\$EPOCH"
    BAD_EVALS=0
    cp "\$CKPT" "\$BEST_CKPT"
    python -c "import json, pathlib, sys; data={'epoch': int(sys.argv[1]), 'target_dev_macro_cer': float(sys.argv[2]), 'checkpoint': sys.argv[3], 'copied_checkpoint': sys.argv[4], 'report_dir': sys.argv[5], 'selection_metric': 'target_dev_macro_CER'}; pathlib.Path(sys.argv[6]).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')" "\$BEST_EPOCH" "\$BEST_CER" "\$CKPT" "\$BEST_CKPT" "\$OUT_DIR" "\$BEST_JSON"
    echo "New best: epoch=\$BEST_EPOCH target_dev_macro_CER=\$BEST_CER"
  else
    BAD_EVALS=\$((BAD_EVALS + 1))
    echo "No improvement: bad_evals=\$BAD_EVALS/$PATIENCE_EVALS, best_epoch=\$BEST_EPOCH, best_CER=\$BEST_CER"
  fi

  if [[ "\$EPOCH" -ge "$MIN_EPOCH" && "\$BAD_EVALS" -ge "$PATIENCE_EVALS" ]]; then
    echo "Early stopping at epoch \$EPOCH."
    tmux kill-session -t "$TRAIN_SESSION" 2>/dev/null || true
    break
  fi
done

while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  sleep 10
done

if [[ ! -s "\$BEST_CKPT" ]]; then
  echo "No best checkpoint was selected." >&2
  exit 1
fi

echo "Evaluating selected checkpoint on target-domain dev..."
rm -rf "$EVAL_ROOT/e1_target_only_best_target_dev"
python scripts/svtrv2/evaluate_d2_checkpoint.py \
  --root "$ROOT" \
  --config "$CONFIG_PATH" \
  --checkpoint "\$BEST_CKPT" \
  --split dev \
  --data-dir "$TARGET_DIR" \
  --labels-dir "$LABELS_DIR" \
  --label-prefix target \
  --output-dir "$EVAL_ROOT/e1_target_only_best_target_dev" \
  --device-id "$EVAL_GPU" \
  --batch-size 32 \
  --require-python-bidi \
  2>&1 | tee "$LOG_DIR/final_dev_eval.log"

echo "Evaluating selected checkpoint once on target-domain test..."
rm -rf "$EVAL_ROOT/e1_target_only_best_target_test"
python scripts/svtrv2/evaluate_d2_checkpoint.py \
  --root "$ROOT" \
  --config "$CONFIG_PATH" \
  --checkpoint "\$BEST_CKPT" \
  --split test \
  --data-dir "$TARGET_DIR" \
  --labels-dir "$LABELS_DIR" \
  --label-prefix target \
  --output-dir "$EVAL_ROOT/e1_target_only_best_target_test" \
  --device-id "$EVAL_GPU" \
  --batch-size 32 \
  --require-python-bidi \
  2>&1 | tee "$LOG_DIR/final_test_eval.log"

python -c "import json, pathlib; out=pathlib.Path('$EVAL_ROOT/e1_target_only_final_summary.json'); best=json.load(open('$RUN_MODEL_DIR/best_target_dev_macro_cer.json', encoding='utf-8')); dev=json.load(open('$EVAL_ROOT/e1_target_only_best_target_dev/metrics_macro_summary.json', encoding='utf-8')); test=json.load(open('$EVAL_ROOT/e1_target_only_best_target_test/metrics_macro_summary.json', encoding='utf-8')); data={'experiment':'E1_target_domain_only','best':best,'dev':dev,'test':test,'logs':'$LOG_DIR','run_dir':'$RUN_MODEL_DIR'}; out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8'); print(json.dumps(data, ensure_ascii=False, indent=2))" \
  2>&1 | tee "$LOG_DIR/final_summary.log"

echo "DONE"
echo "Best checkpoint: \$BEST_CKPT"
echo "Final summary: $EVAL_ROOT/e1_target_only_final_summary.json"
EOF
chmod +x "$LOG_DIR/watch_eval.sh"

echo "[2/3] Starting dual-GPU DDP training: $TRAIN_SESSION"
tmux new-session -d -s "$TRAIN_SESSION" \
  "bash '$LOG_DIR/train_ddp.sh'"

echo "[3/3] Starting target-dev evaluator: $WATCH_SESSION"
tmux new-session -d -s "$WATCH_SESSION" \
  "bash '$LOG_DIR/watch_eval.sh' 2>&1 | tee '$LOG_DIR/watch_eval.log'"

echo
echo "E1 started."
echo "Train: tmux attach -t $TRAIN_SESSION"
echo "Watch: tmux attach -t $WATCH_SESSION"
echo "Logs: $LOG_DIR"
echo "Best checkpoint: $RUN_MODEL_DIR/best_target_dev_macro_cer.pth"
echo "Final summary: $EVAL_ROOT/e1_target_only_final_summary.json"
