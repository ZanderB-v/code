#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition
SESSION=m2_consistency_sweep_p1
SCRIPT_DIR="$ROOT/scripts/svtrv2"
LOG_DIR="$ROOT/04_model_training/logs/m2_consistency_sweep_$(date +%Y%m%d_%H%M%S)"

if [[ "${1:-}" != "--worker" ]]; then
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  mkdir -p "$LOG_DIR"
  printf '%s\n' "$LOG_DIR" > "$ROOT/04_model_training/logs/m2_consistency_sweep_latest.txt"
  tmux new-session -d -s "$SESSION" \
    "bash '$0' --worker '$LOG_DIR' 2>&1 | tee '$LOG_DIR/sweep.log'"
  tmux set-window-option -t "$SESSION":0 remain-on-exit on
  echo "Started: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  echo "Log: $LOG_DIR/sweep.log"
  exit 0
fi

LOG_DIR="$2"
source /home/wudayu/anaconda3/etc/profile.d/conda.sh
conda activate openocr_svtrv2
cd "$ROOT"
export PYTHONUNBUFFERED=1

BASE_CFG="$ROOT/04_model_training/configs/svtrv2_s_m2_dual_order_s50_to_target.yml"
SOURCE_CKPT="$ROOT/04_model_training/runs/svtrv2_s_m1_dual_order_s50/best_clean_dev_macro_cer.pth"
LABELS_DIR="$ROOT/04_model_training/datasets/e1_target_only/labels"

[[ -f "$BASE_CFG" ]] || { echo "Missing config: $BASE_CFG"; exit 1; }
[[ -f "$SOURCE_CKPT" ]] || { echo "Missing source checkpoint: $SOURCE_CKPT"; exit 1; }
[[ -d "$LABELS_DIR" ]] || { echo "Missing labels directory: $LABELS_DIR"; exit 1; }

for spec in 000:0.00 005:0.05 010:0.10 015:0.15 030:0.30; do
  tag="${spec%%:*}"
  alpha="${spec##*:}"
  stage="m2_alpha_${tag}"
  model="svtrv2_s_${stage}_to_target"
  cfg="$ROOT/04_model_training/configs/${model}.yml"
  run_dir="$ROOT/04_model_training/runs/$model"

  echo "========== START $stage alpha=$alpha =========="

  python - "$BASE_CFG" "$cfg" "$run_dir" "$SOURCE_CKPT" "$alpha" "$model" <<'PY'
import re
import sys
from pathlib import Path

base, out, run_dir, checkpoint, alpha, model = sys.argv[1:]
text = Path(base).read_text(encoding="utf-8")
old_run = "/home/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition/04_model_training/runs/svtrv2_s_m2_dual_order_s50_to_target"
text = text.replace(old_run, run_dir)
text = re.sub(r"^  pretrained_model:.*$", f"  pretrained_model: {checkpoint}", text, count=1, flags=re.M)
text = re.sub(r"^  project_name:.*$", f"  project_name: {model}", text, count=1, flags=re.M)
text = re.sub(r"^  model_protocol:.*$", f"  model_protocol: MULTISCRIPT_DUAL_ORDER_SGM_V1_M2_ALPHA_{alpha.replace('.', '_')}", text, count=1, flags=re.M)
text = re.sub(r"^  consistency_weight:.*$", f"  consistency_weight: {alpha}", text, count=1, flags=re.M)
Path(out).write_text(text, encoding="utf-8")
PY

  python scripts/svtrv2/run_p1_rctc_stage.py \
    --root "$ROOT" \
    --stage "$stage" \
    --config "$cfg" \
    --run-dir "$run_dir" \
    --labels-dir "$LABELS_DIR" \
    --label-prefix target \
    --eval-prefix "$stage" \
    --expected-initialization checkpoint \
    --config-kind dual_order \
    --prediction-branch ctc \
    --max-epoch 50 \
    --eval-every 2 \
    --patience-evals 5 \
    --min-epoch 10 \
    --min-delta 0.0001 \
    --ddp-gpus 0,1 \
    --eval-gpu 0 \
    --master-port "$((29910 + 10#${tag}))" \
    --log-dir "$LOG_DIR" \
    --replace

  echo "========== DONE $stage =========="
done

python - "$ROOT" "$LOG_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

root, log_dir = map(Path, sys.argv[1:])
rows = []
for path in sorted((root / "04_model_training/eval_reports").glob("m2_alpha_*_final_summary.json")):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    dev = data.get("dev", {})
    rows.append({
        "experiment": data.get("experiment", path.stem),
        "best_epoch": data.get("best_epoch"),
        "macro_cer": dev.get("macro_cer", data.get("best_clean_dev_macro_cer")),
        "macro_line_accuracy": dev.get("macro_line_accuracy"),
        "config": data.get("config"),
        "best_checkpoint": data.get("best_checkpoint"),
    })
out = log_dir / "m2_consistency_sweep_summary.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if rows:
    with (log_dir / "m2_consistency_sweep_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps({"status": "M2_CONSISTENCY_SWEEP_OK", "rows": rows}, ensure_ascii=False, indent=2))
PY
