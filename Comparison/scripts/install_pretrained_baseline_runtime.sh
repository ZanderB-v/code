#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition}"
CONDA_ENV="${CONDA_ENV:-openocr_svtrv2}"
WHEELS="$ROOT/Comparison/third_party/wheels"

source /home/wudayu/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

(cd "$WHEELS" && sha256sum -c SHA256SUMS.txt)

python -m pip install --no-index --no-deps --find-links "$WHEELS" \
  addict==2.4.0 \
  importlib-metadata==8.7.0 \
  markdown-it-py==3.0.0 \
  mdurl==0.1.2 \
  opencv-python-headless==4.10.0.84 \
  platformdirs==4.2.2 \
  pygments==2.18.0 \
  rich==13.7.1 \
  termcolor==2.4.0 \
  tomli==2.0.1 \
  urllib3==1.26.20 \
  zipp==3.23.0 \
  yapf==0.40.2 \
  mmengine==0.10.7 \
  mmcv-lite==2.0.1 \
  timm==0.4.12

python - <<'PY'
import torch
import timm
import cv2
import importlib_metadata
import requests
import urllib3
import yapf
import mmcv
import mmengine
print({
    "torch": torch.__version__,
    "timm": timm.__version__,
    "opencv": cv2.__version__,
    "importlib_metadata": importlib_metadata.version("importlib-metadata"),
    "requests": requests.__version__,
    "urllib3": urllib3.__version__,
    "yapf": importlib_metadata.version("yapf"),
    "mmcv": mmcv.__version__,
    "mmengine": mmengine.__version__,
})
PY

echo PRETRAINED_BASELINE_RUNTIME_OK
