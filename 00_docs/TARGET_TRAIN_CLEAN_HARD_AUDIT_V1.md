# Target Train Clean Hard Audit V1

## Scope

This protocol mines near-miss samples from the frozen target-domain **Train** split.
It does not read Dev for mining or Test for any purpose, and it never deletes or
rewrites a frozen Train record.

The sole metadata input is `train_reviewed/metadata.jsonl` (66,854 rows); the
combined target metadata and the Dev/Test metadata files are not opened.

## Frozen model and inference

- Model: M3 / SOAR-SVTR.
- Cross-order consistency weight: `0.15`.
- Checkpoint: Clean Dev Macro CER-selected M3 checkpoint, verified by SHA-256.
- Preprocessing: `P1_MSR_V3`.
- Prediction: RCTC branch in U2 visual order; Uyghur reporting uses the frozen
  second `python-bidi` display pass.
- Device: physical GPU1 only (`CUDA_VISIBLE_DEVICES=1`, process-local CUDA 0).
- Numeric mode: FP32. AMP is disabled to preserve inference semantics.

Inference uses aspect-ratio buckets and adaptive batches. The initial equivalent
batch size is 512, may grow to 2048 while peak allocated memory remains below the
20 GiB target, and recursively halves after CUDA OOM. Every retry and the observed
peak memory are recorded. The memory target is operational, not a promise that
PyTorch will allocate exactly 20 GiB.

## Per-sample record

Each of the 66,854 Train samples produces GT, U2 visual prediction, recovered
logical prediction, confidence, character edit distance, language, image path,
sample ID, image quality measurements, risk flags, and eligibility. Failed image
decodes remain in the output with ordinary weight.

## Conservative automatic exclusions

The following flags prevent upweighting but never remove a sample:

- dictionary OOV, Unicode controls/surrogates, non-NFC labels, or suspicious
  mixed primary scripts;
- metadata review status other than `pass`;
- NFKC/spacing-only GT/prediction differences, including half/fullwidth pairs;
- decode failure, abnormal dimensions, low contrast, severe blur, strong edge
  contact suggesting a bad crop, or extremely low-confidence/empty prediction.

Image/label agreement cannot be proved automatically. Every sampled candidate
therefore receives an explicit manual decision.

## Frozen review and gate

Eligible ED=1 and ED=2 candidates are sampled deterministically from six strata:
`zh/ug/kk x ED1/ED2`, at most 30 candidates per stratum, seed `20260911`.
The review membership fingerprint is frozen before decisions are entered.

Allowed decisions are:

```
genuine_ocr_error
gt_annotation_error
normalization_issue
ambiguous_image
```

Only a complete review with `genuine_ocr_error >= 70%` authorizes creation of
`formal_hem_manifest.jsonl`. Otherwise the result is
`TRAIN_HEM_NOT_AUTHORIZED`, and no formal HEM training should run.

When authorized, all Train rows are preserved. Clean ED=1 receives weight 2.0,
clean ED=2 receives 1.5, and every other or manually excluded row remains 1.0.
This Train audit is the pre-registered HEM label-quality gate; the earlier Stage 6
Dev analysis remains diagnostic evidence and is not reused as training data.
