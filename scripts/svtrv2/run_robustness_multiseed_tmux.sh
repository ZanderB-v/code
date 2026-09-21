#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
CONDA_ENV="${CONDA_ENV:-openocr_svtrv2}"
SESSION="${SESSION:-method_robustness_multiseed_v1}"
REPLACE=0
PREFLIGHT_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --replace) REPLACE=1 ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ "$REPLACE" == "1" ]]; then
    tmux kill-session -t "$SESSION"
  else
    echo "tmux session already exists: $SESSION" >&2
    echo "Attach with: tmux attach -t $SESSION" >&2
    exit 1
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/04_model_training/logs/method_robustness_multiseed_${STAMP}"
PLAN="$ROOT/04_model_training/multiseed/method_target_seed_plan_v1.json"
mkdir -p "$LOG_DIR"

cat > "$LOG_DIR/pipeline.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source /home/wudayu/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$ROOT"
export PYTHONUNBUFFERED=1
trap 'code=\$?; echo "PIPELINE_FAILED exit_code=\$code line=\$LINENO"; exit \$code' ERR

echo "===== 0/6 verify frozen method and corruption protocols ====="
python scripts/protocol/freeze_protocol_v2.py \
  --root "$ROOT" \
  --mode verify \
  --workers 8
python scripts/protocol/freeze_method_design_v1.py \
  --root "$ROOT" \
  --mode verify
python scripts/corruption/evaluate_frozen_methods_corruption_dev.py \
  --root "$ROOT" \
  --freeze-evaluation-binding \
  --verify-only

echo "===== 1/6 prepare isolated target-finetune seeds ====="
python scripts/svtrv2/prepare_method_multiseed.py \
  --root "$ROOT" \
  --seeds 20260811 20260812 \
  --output "$PLAN"

echo "===== 2/6 preflight every new seed config before training ====="
python scripts/svtrv2/run_method_multiseed.py \
  --root "$ROOT" \
  --plan "$PLAN" \
  --log-dir "$LOG_DIR" \
  --preflight-only

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "ROBUSTNESS_MULTISEED_PREFLIGHT_ONLY_OK"
  exit 0
fi

echo "===== 3/6 frozen Corrupted Dev on two GPUs ====="
replace_arg=()
if [[ "$REPLACE" == "1" ]]; then replace_arg=(--replace); fi
CUDA_VISIBLE_DEVICES=0 python scripts/corruption/evaluate_frozen_methods_corruption_dev.py \
  --root "$ROOT" --device-id 0 --models b0 b2 m2 m3 "\${replace_arg[@]}" \
  2>&1 | tee "$LOG_DIR/corruption_gpu0.log" &
pid0=\$!
CUDA_VISIBLE_DEVICES=1 python scripts/corruption/evaluate_frozen_methods_corruption_dev.py \
  --root "$ROOT" --device-id 0 --models b1 m1 full "\${replace_arg[@]}" \
  2>&1 | tee "$LOG_DIR/corruption_gpu1.log" &
pid1=\$!
set +e
wait \$pid0; status0=\$?
wait \$pid1; status1=\$?
set -e
if [[ \$status0 -ne 0 || \$status1 -ne 0 ]]; then
  echo "Corruption evaluation failed: gpu0=\$status0 gpu1=\$status1" >&2
  exit 1
fi
python scripts/corruption/summarize_frozen_methods_corruption.py --root "$ROOT"

echo "===== 4/6 target-only multiseed finetuning (dual-GPU, sequential) ====="
train_args=()
if [[ "$REPLACE" == "1" ]]; then train_args=(--replace); fi
python scripts/svtrv2/run_method_multiseed.py \
  --root "$ROOT" \
  --plan "$PLAN" \
  --log-dir "$LOG_DIR" \
  --ddp-gpus 0,1 \
  --eval-gpu 0 \
  "\${train_args[@]}"

echo "===== 5/6 mean/std and paired bootstrap ====="
python scripts/svtrv2/summarize_method_multiseed.py \
  --root "$ROOT" \
  --plan "$PLAN" \
  --bootstrap-iterations 10000 \
  --bootstrap-seed 20260813

echo "===== 6/6 complete ====="
echo "METHOD_ROBUSTNESS_AND_MULTISEED_V1_COMPLETE"
echo "Test was not evaluated."
EOF

chmod +x "$LOG_DIR/pipeline.sh"
tmux new-session -d -s "$SESSION" \
  "bash '$LOG_DIR/pipeline.sh' 2>&1 | tee '$LOG_DIR/pipeline.log'"
tmux set-option -t "$SESSION" remain-on-exit on
ln -sfn "$LOG_DIR" "$ROOT/04_model_training/logs/method_robustness_multiseed_latest"

echo "Started: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR/pipeline.log"
echo "Plan: $PLAN"
