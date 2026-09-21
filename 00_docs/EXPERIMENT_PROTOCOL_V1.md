# Multilingual Meme Line Recognition Protocol V1

> Superseded for current experiments. The target-only freeze is now defined by
> `TARGET_BENCHMARK_PROTOCOL_V2.md`. This historical file must not be used to
> claim that synthetic data or fixed `48x640` preprocessing is frozen.

Status: reset for synthetic regeneration. The target conflict revision is
retained; the synthetic corpus, generated labels, and final freeze are pending.

## 1. Frozen target-domain benchmark

Canonical root:

`01_data_preparation/real_line_dataset_eval_reviewed`

Data provenance:

- Chinese: natural Chinese meme text lines.
- Uyghur: target-domain re-rendered Uyghur meme text lines.
- Kazakh: target-domain re-rendered Cyrillic Kazakh meme text lines.

The split unit is `source_id`, representing the complete source meme image.
All lines and variants from one source image must remain in one split.

| Split | Chinese | Uyghur | Kazakh | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 23,954 | 21,899 | 21,001 | 66,854 |
| dev | 346 | 295 | 310 | 951 |
| test | 1,142 | 931 | 993 | 3,066 |

Ten objectively invalid Uyghur rows were excluded before final freeze: five exact-image groups carried two different labels. The crop files remain for provenance in the dataset directory. No model metric was used to choose these exclusions.

After the corrected protocol is accepted, do not re-render, re-split, remove, relabel, or otherwise tune the test set.

## 2. Planned auxiliary synthetic corpus

Canonical root:

`03_synthetic_generation/synthetic_formal_v1`

Shards to regenerate:

- `synthetic_shard_0000_parallel`
- `synthetic_shard_0001_parallel`
- `synthetic_shard_0002_parallel`
- `synthetic_shard_0003_parallel`
- `synthetic_shard_0004_parallel`

After regeneration and validation, the corpus will contain exactly 50,000
samples per language. It is a training and pretraining resource only and is
never the primary test set. The corpus is not frozen until the final freeze
report passes.

## 3. Frozen character dictionary

Canonical file:

`04_model_training/character_dict_hz_ug_kk_v1/character_dict.txt`

All target train/dev/test labels and all frozen synthetic labels must be fully
covered. OpenOCR adds its CTC blank and the configured space character outside
the file.

## 4. Frozen label order

- Chinese and Kazakh use standard Unicode logical LTR order.
- Uyghur text is stored in standard Unicode logical order.
- Uyghur CTC labels use strict U2 visual order:
  `bidi.algorithm.get_display(normalize_spaces(text), base_dir="R")`.
- `python-bidi` is required. A simple `text[::-1]` fallback is forbidden.
- U2 predictions are converted back to logical order before final metrics.

The exact target labels will be regenerated from the canonical metadata under:

`04_model_training/datasets/e1_target_only/labels`

## 5. Frozen input preprocessing

OpenOCR preprocessing is:

1. Decode as RGB.
2. `RecTVResize(image_shape=[48, 640], padding=True)`.
3. Preserve aspect ratio with bicubic interpolation while the resized width is
   at most 640.
4. Pad unused width on the right.
5. Convert to tensor and normalize with mean 0.5 and standard deviation 0.5.

For an input whose aspect ratio requires more than 640 pixels at height 48,
the width is capped at 640. The maximum CTC label length is 60.

## 6. Frozen metrics

Metrics are computed in standard logical order.

- CER: total character Levenshtein distance divided by total ground-truth
  characters.
- WER: total whitespace-token Levenshtein distance divided by total
  ground-truth tokens.
- 1-NED: macro average across lines of
  `1 - ED(gt,pred) / max(len(gt),len(pred),1)`.
- Line Accuracy: exact logical-string matches divided by sample count.
- Primary model-selection metric: arithmetic mean of Chinese, Uyghur, and
  Kazakh CER on the target-domain dev set.

No case folding or punctuation removal is applied. Uyghur whitespace is
normalized as part of the frozen bidi conversion.

The canonical implementation is:

`scripts/svtrv2/metrics_v1.py`

## 7. Test policy

Checkpoints and hyperparameters are selected using dev only. The frozen clean
test and its future frozen corruption protocol are run only after checkpoint
selection. Test results must not be used to modify rendering, data splits,
label rules, model architecture, or training hyperparameters.

## 8. Freeze and verify

Freeze once on the complete server copy:

```bash
python scripts/protocol/freeze_experiment_v1.py \
  --root /data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition \
  --mode freeze \
  --workers 8
```

Verify later:

```bash
python scripts/protocol/freeze_experiment_v1.py \
  --root /data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition \
  --mode verify \
  --workers 8
```

Generated artifacts:

- `00_docs/frozen_protocol_v1/frozen_manifest.json`
- `00_docs/frozen_protocol_v1/verification_report.json`
- `00_docs/frozen_protocol_v1/target_images.sha256.tsv`
- `00_docs/frozen_protocol_v1/synthetic_images.sha256.tsv`

The freeze is valid only when `verification_report.json` has
`"status": "passed"`.
