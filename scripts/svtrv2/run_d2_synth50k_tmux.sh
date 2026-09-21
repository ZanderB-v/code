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
CONFIG_PATH="$ROOT/04_model_training/configs/svtrv2_s_d2_synth50k.yml"
RUN_MODEL_DIR="$ROOT/04_model_training/runs/svtrv2_s_d2_synth50k"
EVAL_DIR="$ROOT/04_model_training/eval_reports"
LOG_DIR="$ROOT/04_model_training/logs/d2_synth50k_$(date +%Y%m%d_%H%M%S)"

TRAIN_SESSION="${TRAIN_SESSION:-d2_synth50k_train}"
WATCH_SESSION="${WATCH_SESSION:-d2_synth50k_watch_eval}"
TRAIN_GPU="${TRAIN_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-1}"
MAX_EPOCH="${MAX_EPOCH:-50}"
PATIENCE="${PATIENCE:-8}"
MIN_EPOCH="${MIN_EPOCH:-12}"
MIN_DELTA="${MIN_DELTA:-0.0005}"
BATCH_SIZE_PER_CARD="${BATCH_SIZE_PER_CARD:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-0.00005}"
SHARDS="${SHARDS:-shard_0000 shard_0001 shard_0002 shard_0003 shard_0004}"

usage() {
  cat <<EOF
Usage:
  bash scripts/svtrv2/run_d2_synth50k_tmux.sh [--replace]

Default D2 uses exact formal shards:
  $SHARDS

Environment overrides:
  ROOT=$ROOT
  TRAIN_GPU=0
  EVAL_GPU=1
  MAX_EPOCH=50
  PATIENCE=8
  MIN_EPOCH=12
  MIN_DELTA=0.0005
  BATCH_SIZE_PER_CARD=16
  NUM_WORKERS=4
  LR=0.00005
  SHARDS="shard_0000 shard_0001 shard_0002 shard_0003 shard_0004"
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

for session in "$TRAIN_SESSION" "$WATCH_SESSION"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    if [[ "$REPLACE" == "1" ]]; then
      tmux kill-session -t "$session"
    else
      echo "tmux session already exists: $session" >&2
      echo "Use --replace to kill and restart D2 sessions." >&2
      exit 1
    fi
  fi
done

mkdir -p "$LOG_DIR" "$EVAL_DIR" "$RUN_MODEL_DIR"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$ROOT"

echo "[1/3] Preparing D2 synthetic 50k/lang labels and config..."
# shellcheck disable=SC2086
python scripts/svtrv2/prepare_d2_synth50k.py \
  --root "$ROOT" \
  --shards $SHARDS \
  --max-epoch "$MAX_EPOCH" \
  --batch-size-per-card "$BATCH_SIZE_PER_CARD" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  2>&1 | tee "$LOG_DIR/prepare.log"

cat > "$LOG_DIR/train_cmd.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$OPENOCR_ROOT"
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python tools/train_rec.py -c "$CONFIG_PATH" 2>&1 | tee "$LOG_DIR/train.log"
EOF
chmod +x "$LOG_DIR/train_cmd.sh"

cat > "$LOG_DIR/watch_eval_loop.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$ROOT"

BEST_CKPT="$RUN_MODEL_DIR/best_real_dev_macro_cer.pth"
BEST_JSON="$RUN_MODEL_DIR/best_real_dev_macro_cer.json"
WATCH_JSONL="$LOG_DIR/real_dev_epoch_metrics.jsonl"
BEST_CER=999999.0
BEST_EPOCH=0
BAD_EPOCHS=0

echo "Watching epoch checkpoints in: $RUN_MODEL_DIR"
echo "Selection metric: real_dev_macro_CER, lower is better"

for EPOCH in \$(seq 1 "$MAX_EPOCH"); do
  CKPT="$RUN_MODEL_DIR/epoch_\${EPOCH}.pth"
  echo "Waiting for \$CKPT"
  while [[ ! -s "\$CKPT" ]]; do
    if ! tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
      echo "Training session ended before epoch \$EPOCH checkpoint appeared."
      break 2
    fi
    sleep 30
  done

  EPOCH_PAD=\$(printf "%04d" "\$EPOCH")
  OUT_DIR="$EVAL_DIR/d2_synth50k_dev_epoch_\${EPOCH_PAD}"
  echo "Evaluating epoch \$EPOCH on real dev -> \$OUT_DIR"
  python scripts/svtrv2/evaluate_d2_checkpoint.py \
    --root "$ROOT" \
    --config "$CONFIG_PATH" \
    --checkpoint "\$CKPT" \
    --split dev \
    --output-dir "\$OUT_DIR" \
    --device-id "$EVAL_GPU" \
    --batch-size 32 \
    --require-python-bidi \
    2>&1 | tee "$LOG_DIR/eval_epoch_\${EPOCH_PAD}.log"

  CUR_CER=\$(python -c "import json; from pathlib import Path; p=Path('\$OUT_DIR')/'metrics_macro_summary.json'; print(json.load(p.open(encoding='utf-8'))['macro_cer'])")
  echo "{\"epoch\": \$EPOCH, \"macro_cer\": \$CUR_CER, \"checkpoint\": \"\$CKPT\", \"report\": \"\$OUT_DIR\"}" >> "\$WATCH_JSONL"

  IMPROVED=\$(python -c "import sys; cur=float(sys.argv[1]); best=float(sys.argv[2]); delta=float(sys.argv[3]); print(1 if cur < best - delta else 0)" "\$CUR_CER" "\$BEST_CER" "$MIN_DELTA")
  if [[ "\$IMPROVED" == "1" ]]; then
    BEST_CER="\$CUR_CER"
    BEST_EPOCH="\$EPOCH"
    BAD_EPOCHS=0
    cp "\$CKPT" "\$BEST_CKPT"
    python -c "import json, pathlib, sys; data={'epoch': int(sys.argv[1]), 'real_dev_macro_cer': float(sys.argv[2]), 'checkpoint': sys.argv[3], 'copied_checkpoint': sys.argv[4], 'report_dir': sys.argv[5], 'selection_metric': 'real_dev_macro_CER'}; pathlib.Path(sys.argv[6]).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')" "\$BEST_EPOCH" "\$BEST_CER" "\$CKPT" "\$BEST_CKPT" "\$OUT_DIR" "\$BEST_JSON"
    echo "New best: epoch=\$BEST_EPOCH real_dev_macro_CER=\$BEST_CER"
  else
    BAD_EPOCHS=\$((BAD_EPOCHS + 1))
    echo "No improvement. bad_epochs=\$BAD_EPOCHS / $PATIENCE; best_epoch=\$BEST_EPOCH best_CER=\$BEST_CER"
  fi

  if [[ "\$EPOCH" -ge "$MIN_EPOCH" && "\$BAD_EPOCHS" -ge "$PATIENCE" ]]; then
    echo "Early stopping triggered at epoch \$EPOCH. Killing train session: $TRAIN_SESSION"
    tmux kill-session -t "$TRAIN_SESSION" 2>/dev/null || true
    break
  fi
done

while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  echo "Waiting for train session to exit..."
  sleep 30
done

if [[ ! -s "\$BEST_CKPT" ]]; then
  if [[ -s "$RUN_MODEL_DIR/latest.pth" ]]; then
    echo "No best checkpoint found; falling back to latest.pth" >&2
    cp "$RUN_MODEL_DIR/latest.pth" "\$BEST_CKPT"
  else
    echo "No best checkpoint or latest checkpoint was produced. Training failed before saving a checkpoint." >&2
    exit 1
  fi
fi

echo "Running final best checkpoint on real dev..."
python scripts/svtrv2/evaluate_d2_checkpoint.py \
  --root "$ROOT" \
  --config "$CONFIG_PATH" \
  --checkpoint "\$BEST_CKPT" \
  --split dev \
  --output-dir "$EVAL_DIR/d2_synth50k_best_real_dev" \
  --device-id "$EVAL_GPU" \
  --batch-size 32 \
  --require-python-bidi \
  2>&1 | tee "$LOG_DIR/final_dev_eval.log"

echo "Running final best checkpoint on real test..."
python scripts/svtrv2/evaluate_d2_checkpoint.py \
  --root "$ROOT" \
  --config "$CONFIG_PATH" \
  --checkpoint "\$BEST_CKPT" \
  --split test \
  --output-dir "$EVAL_DIR/d2_synth50k_best_real_test" \
  --device-id "$EVAL_GPU" \
  --batch-size 32 \
  --require-python-bidi \
  2>&1 | tee "$LOG_DIR/final_test_eval.log"

python -c "import json, pathlib; root=pathlib.Path('$ROOT'); out=root/'04_model_training'/'eval_reports'/'d2_synth50k_final_summary.json'; best=json.load(open('$RUN_MODEL_DIR/best_real_dev_macro_cer.json', encoding='utf-8')) if pathlib.Path('$RUN_MODEL_DIR/best_real_dev_macro_cer.json').exists() else {}; dev=json.load(open('$EVAL_DIR/d2_synth50k_best_real_dev/metrics_macro_summary.json', encoding='utf-8')); test=json.load(open('$EVAL_DIR/d2_synth50k_best_real_test/metrics_macro_summary.json', encoding='utf-8')); data={'experiment':'D2_synth50k_per_lang','best':best,'dev':dev,'test':test,'logs':'$LOG_DIR','run_dir':'$RUN_MODEL_DIR'}; out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8'); print(json.dumps(data, ensure_ascii=False, indent=2))" \
  2>&1 | tee "$LOG_DIR/final_summary.log"

echo "DONE"
echo "Best checkpoint: \$BEST_CKPT"
echo "Final summary: $EVAL_DIR/d2_synth50k_final_summary.json"
EOF
chmod +x "$LOG_DIR/watch_eval_loop.sh"

echo "[2/3] Starting train tmux session: $TRAIN_SESSION"
tmux new-session -d -s "$TRAIN_SESSION" "bash '$LOG_DIR/train_cmd.sh'"

echo "[3/3] Starting real-dev watcher/evaluator tmux session: $WATCH_SESSION"
tmux new-session -d -s "$WATCH_SESSION" "bash '$LOG_DIR/watch_eval_loop.sh' 2>&1 | tee '$LOG_DIR/watch_eval.log'"

echo
echo "Started D2 synthetic 50k/lang:"
echo "  Train: tmux attach -t $TRAIN_SESSION"
echo "  Watch: tmux attach -t $WATCH_SESSION"
echo
echo "Logs:"
echo "  $LOG_DIR"
echo
echo "Key outputs:"
echo "  $CONFIG_PATH"
echo "  $RUN_MODEL_DIR/best_real_dev_macro_cer.pth"
echo "  $RUN_MODEL_DIR/best_real_dev_macro_cer.json"
echo "  $EVAL_DIR/d2_synth50k_best_real_dev/metrics_macro_summary.json"
echo "  $EVAL_DIR/d2_synth50k_best_real_test/metrics_macro_summary.json"
echo "  $EVAL_DIR/d2_synth50k_final_summary.json"
