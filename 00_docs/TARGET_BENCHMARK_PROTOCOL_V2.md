# Target Benchmark Protocol V2

Status: target-domain benchmark only. Synthetic data and model preprocessing
remain intentionally unfrozen.

## Frozen benchmark

Canonical root:

`01_data_preparation/real_line_dataset_eval_reviewed`

The split unit is `source_id`, representing the complete source meme image.
Every line from one source image must remain in one split.

| Split | Chinese | Uyghur | Kazakh | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 23,954 | 21,899 | 21,001 | 66,854 |
| dev | 346 | 295 | 310 | 951 |
| test | 1,142 | 931 | 993 | 3,066 |

Chinese samples are natural Chinese meme lines. Uyghur and Kazakh samples are
in-domain re-rendered meme lines and must not be described as naturally
occurring Uyghur or Kazakh meme text.

## Frozen labels

- Source labels are stored in standard Unicode logical order.
- Chinese and Kazakh CTC labels equal their logical labels.
- The frozen Uyghur U2 baseline is:
  `bidi.algorithm.get_display(normalize_spaces(text), base_dir="R")`.
- `python-bidi` is mandatory. `text[::-1]` is forbidden.
- Proposed direction-alignment models may use logical labels instead of U2,
  but must retain U2 as the frozen baseline protocol.

## Frozen dictionary and metrics

Character dictionary:

`04_model_training/character_dict_hz_ug_kk_v1/character_dict.txt`

Metric implementation:

`scripts/svtrv2/metrics_v1.py`

Metrics are CER, whitespace-token WER, macro 1-NED, and exact Line Accuracy.
They are reported in standard logical order. The primary checkpoint-selection
metric is the arithmetic mean of Chinese, Uyghur, and Kazakh dev CER.

## Frozen evidence

The freeze records:

- exact target membership and labels;
- `source_id` split membership;
- all target image SHA-256 hashes;
- character dictionary hash and coverage;
- strict U2 implementation and test vectors;
- metric implementation hash and test vector.

Run the complete freeze on the server:

```bash
python scripts/protocol/freeze_target_benchmark_v2.py \
  --root /data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition \
  --mode freeze \
  --workers 8 \
  --replace
```

Verify later:

```bash
python scripts/protocol/freeze_target_benchmark_v2.py \
  --root /data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition \
  --mode verify \
  --workers 8
```

The freeze is valid only when
`00_docs/frozen_target_benchmark_v2/verification_report.json` has
`"status": "passed"`.

## Explicitly not frozen

- synthetic data;
- final input image size;
- fixed resize or MSR configuration;
- model architecture and training hyperparameters.
