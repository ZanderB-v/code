# Corruption Protocol V1

## Purpose

This protocol measures whether a checkpoint selected on clean target-domain
Dev remains reliable under deterministic, label-preserving image degradation.
It is an evaluation protocol, not a method for enlarging the training set.

## Operator set

The protocol uses the 12 operators selected for the second augmentation round
in Xu et al.:

1. Curve
2. Distort
3. Stretch
4. Rotate
5. Perspective
6. Shrink
7. TranslateX
8. TranslateY
9. Contrast
10. Brightness
11. JpegCompression
12. Pixelate

The operator set originates from STRAug:

- https://github.com/roatienza/straug
- Rowel Atienza, "Data Augmentation for Scene Text Recognition", ICCVW 2021.
- Miaomiao Xu et al., "Correlation-guided decoding strategy for low-resource
  Uyghur scene text recognition", Complex & Intelligent Systems, 2025.

The Xu et al. paper uses random magnitude 0-3 in its first augmentation round
and random magnitude 0-2 in its second 12-operator round. Those random training
policies are not copied into this benchmark.

## Evaluation rules

- Severity 0 is the unchanged Clean Dev image.
- Corrupted Dev contains deterministic severities 1, 2, and 3.
- Each variant applies exactly one corruption.
- Corruptions are never chained.
- Every model receives the same image bytes and labels.
- Clean and corrupted inference use P1/MSR per-sample `batch_size = 1`.
- The global seed is 20260801.
- Clean Dev Macro CER remains the checkpoint-selection metric.
- Corrupted Dev is diagnostic and cannot select checkpoints.
- Test receives a frozen manifest only during development.
- Test images and predictions are generated only after all models are frozen.

The implementation is a portable Pillow/NumPy adaptation. It preserves the
operator semantics but is not claimed to be byte-identical to STRAug's
OpenCV-TPS implementation. Rotate, TranslateX, and TranslateY use a
label-safe expanding canvas so the corruption cannot remove labelled
characters. The other operators preserve the source canvas size. Pixelate
uses scales 0.80, 0.68, and 0.55 for Levels 1, 2, and 3. The implementation
file hash and the 100-sample human calibration CSV are frozen with the
protocol.

## Workflow

### 1. Build Dev draft

```bash
bash scripts/corruption/run_corruption_protocol_v1_tmux.sh --replace
tmux attach -t corruption_protocol_v1_build
```

Sync and inspect:

```text
05_evaluation/corruption_protocol_v1/dev_calibration_review.html
05_evaluation/corruption_protocol_v1/calibration_review_assets/
05_evaluation/corruption_protocol_v1/dev_build_summary.json
05_evaluation/corruption_protocol_v1/corruption_protocol_v1_draft.json
```

Approve only when Level 1 is mild, Level 2 is clearly harder, and Level 3 is
difficult but still label-preserving. Geometric transforms must not remove
whole characters systematically.

### 2. Freeze after human calibration

```bash
python scripts/corruption/freeze_corruption_protocol.py \
  --root /data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition \
  --reviewer wudayu \
  --pilot-review-csv /data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition/05_evaluation/corruption_protocol_v1_pilot100_v2/human_review/corruption_pilot100_v2_review.csv \
  --notes "Pilot100 v2 accepted; all operators and all three levels retained." \
  --approve
```

This validates Dev files and hashes, then creates a manifest-only Test plan. It
does not generate corrupted Test images and does not run a model on Test.

### 3. Evaluate Clean/Corrupted Dev

After synchronizing the final P1/MSR V3 evaluator, authorize its code
provenance without regenerating any corruption image or manifest:

```bash
python scripts/corruption/revise_evaluator_for_p1_v3.py \
  --root /data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition \
  --reason "P1/MSR V3 balanced DDP sampler and per-sample Clean Dev inference; corruption data and manifests unchanged"
```

```bash
bash scripts/corruption/run_corruption_dev_eval_tmux.sh --replace
tmux attach -t corruption_dev_v1_eval
```

The three tmux windows are:

- `gpu0`: E0 and B0/E5
- `gpu1`: E1 and B1
- `summary`: waits for all four and creates the controlled comparison

Primary outputs:

```text
05_evaluation/corruption_protocol_v1/eval_reports/clean_corrupted_dev_comparison.json
05_evaluation/corruption_protocol_v1/eval_reports/clean_corrupted_dev_comparison.csv
```

## Reported metrics

- Clean Macro CER
- Mean Corrupted Macro CER over 36 conditions
- Absolute Macro CER increase
- Relative Macro CER increase
- Per-language CER, WER, 1-NED, and Line Accuracy
- Per-operator mean CER
- Per-severity mean CER

Uyghur reports both the direct U2 visual-order metric and the existing
heuristic logical-order metric.
