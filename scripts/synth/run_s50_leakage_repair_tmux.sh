#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
CONDA_SH="${CONDA_SH:-/home/wudayu/anaconda3/etc/profile.d/conda.sh}"
RENDER_ENV="${RENDER_ENV:-paddleocr_vl15}"
MODEL_ENV="${MODEL_ENV:-openocr_svtrv2}"
SESSION="${SESSION:-s50_leakage_repair_v1}"
CHROME="${CHROME:-/data_home/wudayu/local_chrome/149.0.7827.22/chrome-linux64/chrome}"
CHROMEDRIVER="${CHROMEDRIVER:-/data_home/wudayu/local_chrome/149.0.7827.22/chromedriver-linux64/chromedriver}"
REPAIR_ROOT="$ROOT/03_synthetic_generation/synthetic_formal_v2/leakage_repair_v1"

if [[ "${1:-}" != "--replace" ]]; then
  echo "Usage: bash scripts/synth/run_s50_leakage_repair_tmux.sh --replace" >&2
  exit 2
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

source "$CONDA_SH"
conda activate "$RENDER_ENV"
python -c "from bidi.algorithm import get_display; print('PYTHON_BIDI_OK')"
test -x "$CHROME"
test -x "$CHROMEDRIVER"

LOG_DIR="$ROOT/03_synthetic_generation/synthetic_formal_v2/logs/leakage_repair_v1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
RUNNER="$LOG_DIR/run_repair.sh"

cat > "$RUNNER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

source "$CONDA_SH"
conda activate "$RENDER_ENV"
cd "$ROOT"

python scripts/synth/prepare_s50_leakage_repair.py \
  --root "$ROOT" \
  --output-dir "$REPAIR_ROOT" \
  --seed 20260729 \
  --reserve-per-language 64 \
  --replace

for lang in zh ug kk; do
  case "$lang" in
    zh) seed_offset=1 ;;
    ug) seed_offset=2 ;;
    kk) seed_offset=3 ;;
  esac
  needed_rows="$(python -c "import json; p=json.load(open('$REPAIR_ROOT/repair_plan.json', encoding='utf-8')); print(p['languages']['$lang']['leaked_rows_to_replace'])")"
  render_rows="$((needed_rows + 16))"
  part="$REPAIR_ROOT/repair_parts/synthetic_repair_${lang}"
  echo "===== render repair language=$lang needed=$needed_rows target=$render_rows ====="
  python scripts/synth/06_generate_samples.py \
    --text-pool-dir "$REPAIR_ROOT/text_pool" \
    --output-dir "$part" \
    --per-lang "$render_rows" \
    --languages "$lang" \
    --seed "$((202607290 + seed_offset))" \
    --shard-id "leakage_repair_v1_${lang}" \
    --start-index 900001 \
    --text-plan-seed 20260728 \
    --max-text-occurrences 1 \
    --style-profile formal_diverse \
    --max-text-len-zh 30 \
    --max-text-len-ugkk 45 \
    --raw-height-min 64 \
    --raw-height-max 128 \
    --max-raw-width 1200 \
    --render-retries 5 \
    --review-samples 80 \
    --chrome-binary "$CHROME" \
    --driver-path "$CHROMEDRIVER" \
    --no-sandbox \
    --require-python-bidi \
    --overwrite

  python scripts/synth/07_postprocess_images.py \
    --dataset-dir "$part" \
    --target-height 48 \
    --max-aspect 40 \
    --seed "$((202607390 + seed_offset))" \
    --difficulty-profile formal_diverse \
    --overwrite-final

  python scripts/synth/08_validate_dataset.py \
    --dataset-dir "$part" \
    --character-dict "$ROOT/04_model_training/character_dict_hz_ug_kk_v1/character_dict.txt" \
    --max-aspect 40
done

python scripts/synth/finalize_s50_repair_metadata.py \
  --repair-root "$REPAIR_ROOT"

python scripts/synth/build_s50_nested_subsets.py \
  --root "$ROOT" \
  --formal-dir "$ROOT/03_synthetic_generation/synthetic_formal_v2" \
  --workers 8 \
  --replace \
  --exclude-target-eval-texts \
  --repair-metadata \
    "$REPAIR_ROOT/repair_parts/synthetic_repair_zh/metadata_repair_valid.jsonl" \
    "$REPAIR_ROOT/repair_parts/synthetic_repair_ug/metadata_repair_valid.jsonl" \
    "$REPAIR_ROOT/repair_parts/synthetic_repair_kk/metadata_repair_valid.jsonl"

conda activate "$MODEL_ENV"
python scripts/protocol/freeze_protocol_v2.py \
  --root "$ROOT" \
  --mode freeze \
  --workers 8 \
  --replace

python scripts/svtrv2/audit_formal_research_protocol.py \
  --root "$ROOT" \
  --output "$REPAIR_ROOT/formal_research_audit.json"

echo "===== S50 leakage repair, subset rebuild, Protocol V2 freeze, and audit complete ====="
EOF

chmod +x "$RUNNER"
export ROOT CONDA_SH RENDER_ENV MODEL_ENV CHROME CHROMEDRIVER REPAIR_ROOT
tmux new-session -d -s "$SESSION" \
  "export ROOT='$ROOT' CONDA_SH='$CONDA_SH' RENDER_ENV='$RENDER_ENV' MODEL_ENV='$MODEL_ENV' CHROME='$CHROME' CHROMEDRIVER='$CHROMEDRIVER' REPAIR_ROOT='$REPAIR_ROOT'; bash '$RUNNER' 2>&1 | tee '$LOG_DIR/repair.log'"

echo "S50 leakage repair started."
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR/repair.log"
echo "This renders only replacement rows, then rebuilds S10/S25/S50 and Protocol V2."
