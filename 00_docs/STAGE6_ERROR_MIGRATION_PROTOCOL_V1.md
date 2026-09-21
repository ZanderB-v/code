# Stage 6 Error Migration Protocol V1

## Scope

Stage 6 is diagnostic analysis only. It compares the frozen Clean Dev predictions
of B1 (SVTRv2-S, CER-selected epoch 20) and M3/SOAR-SVTR (alpha=0.15,
CER-selected epoch 34). It does not train, tune, evaluate Corrupted Dev, or access
Test.

## Frozen inputs

- Data: Protocol V2 Clean Dev (zh=346, ug=295, kk=310).
- Preprocessing: P1_MSR_V3.
- B1: Clean Dev Macro-CER checkpoint, epoch 20.
- M3: Clean Dev Macro-CER checkpoint, epoch 34, consistency alpha=0.15.
- Uyghur analysis uses the saved logical-order GT and logical-order prediction.
- Chinese and Kazakh use the saved evaluation-normalized GT and prediction.

The script verifies summary, configuration, checkpoint, and prediction hashes. It
also reproduces per-language CER and Line Accuracy before producing analysis.

## Deterministic edit alignment

Character operations use a minimum Levenshtein alignment. Ambiguous optimal paths
use the fixed priority: match, substitution, deletion, insertion. Operation and
confusion counts must therefore be compared only under this protocol.

## Statistical analysis

- Line Accuracy: exact two-sided McNemar/binomial test using Wrong-to-Correct and
  Correct-to-Wrong pairs.
- Macro CER: paired bootstrap, stratified by language, 10,000 repetitions, seed
  20260731. The estimand is M3 Macro CER minus B1 Macro CER.

## Script-character analysis

Kazakh-specific Cyrillic uses the frozen set:
`ӘәҒғҚқҢңӨөҰұҮүҺһІі`.

For Uyghur, the reported category is explicitly `ug_arabic_script`, not a claim
that each character is unique to Uyghur. Category error rate counts substitutions
and deletions aligned to category GT characters. Insertions are reported
separately because they cannot be assigned to a reference-character denominator.

## HEM gate

Near-Miss Hard Example Mining is only a candidate when either:

1. ED=1 accounts for at least 50% of M3 remaining error rows; or
2. ED=1 is larger than each of ED=2 and ED>=3.

The quantitative condition alone cannot authorize HEM. Up to 50 deterministic
ED=1 samples per language must be reviewed as one of:

- `genuine_ocr_error`
- `gt_annotation_error`
- `normalization_issue`
- `ambiguous_image`

HEM receives `go` only when the quantitative condition holds, review is complete,
and at least 50% of reviewed rows are genuine OCR errors. Until then the decision
is `pending_manual_ed1_review`.

## GPU and batch policy

The tmux wrapper exposes only physical GPU0. This stage reuses frozen predictions,
so it should not allocate GPU memory. Training batch size is intentionally outside
the Stage 6 protocol. Future training may use a GPU0-only memory preflight, but a
batch change must be frozen before a comparison begins and must not be introduced
mid-run or between compared methods.
