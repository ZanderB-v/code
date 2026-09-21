#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
CONDA_ENV="${CONDA_ENV:-openocr_svtrv2}"
SESSION="${SESSION:-pretrained_baselines_v1_formal}"
REPLACE=0
if [[ "${1:-}" == "--replace" ]]; then
  REPLACE=1
  shift
fi
if [[ "$#" -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(crnn svtr parseq abinet)
fi
REPLACE_ARG=""
if [[ "$REPLACE" -eq 1 ]]; then
  REPLACE_ARG="--replace"
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
cd "$ROOT"

python - <<'PY'
import json
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path("Comparison/scripts").resolve()))
from formal_baseline_data import (
    formal_implementation_hashes,
    load_protocol,
    preflight_implementation_hashes,
    protocol_artifact_hashes,
)

p = Path("Comparison/pretrained_baselines_v1/outputs/pretrained_baselines_v1_preflight_summary.json")
if not p.is_file():
    raise SystemExit(f"Missing enhanced preflight summary: {p}")
x = json.loads(p.read_text(encoding="utf-8"))
protocol = Path("Comparison/pretrained_baselines_v1/protocol.json")
protocol_sha256 = hashlib.sha256(protocol.read_bytes()).hexdigest()
protocol_payload = load_protocol(protocol)
implementation_sha256 = formal_implementation_hashes()
preflight_sha256 = preflight_implementation_hashes()
artifact_sha256 = protocol_artifact_hashes(protocol_payload)
assert x["status"] == "passed", x
assert x["formal_training_allowed"] is True, x
assert x["formal_protocol_audit"] == "passed", x
assert x["protocol_sha256"] == protocol_sha256, (x["protocol_sha256"], protocol_sha256)
assert x["implementation_sha256"] == implementation_sha256, "preflight code is stale"
assert x["preflight_implementation_sha256"] == preflight_sha256, "preflight gate code is stale"
assert x["artifact_sha256"] == artifact_sha256, "data or public weights changed after preflight"
for model in ("crnn", "svtr", "parseq", "abinet"):
    assert x["models"][model]["formal_ddp_preflight"] == "passed", (model, x)
print("FORMAL_TMUX_GATE_OK")
PY

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/Comparison/pretrained_baselines_v1/logs/formal_$STAMP"
mkdir -p "$LOG_DIR"

cat > "$LOG_DIR/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
trap 'rc=\$?; printf "PRETRAINED_BASELINES_V1_FORMAL_FAILED exit_code=%s\\n" "\$rc" | tee "$LOG_DIR/FAILED"; exit "\$rc"' ERR
source /home/wudayu/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$ROOT"
export PYTHONPATH="$ROOT/Comparison/scripts:$ROOT/Comparison/third_party/parseq_v1_0_0:$ROOT/Comparison/third_party/mmocr_v1_0_1:\${PYTHONPATH:-}"

python Comparison/scripts/run_pretrained_baselines_formal.py \
  --models ${MODELS[*]} \
  $REPLACE_ARG \
  2>&1 | tee "$LOG_DIR/formal.log"

touch "$LOG_DIR/PASSED"
echo "PRETRAINED_BASELINES_V1_FORMAL_TMUX_COMPLETE"
EOF
chmod +x "$LOG_DIR/run.sh"

tmux new-session -d -s "$SESSION" "bash '$LOG_DIR/run.sh'"
echo "Formal comparison started: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR/formal.log"
echo "Models: ${MODELS[*]}"
echo "Test is blocked."
