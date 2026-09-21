# M3 + SCDL Target-Only Pilot V1

## Question

Does script-aware discrimination of CTC-aligned character features reduce
same-script hard confusions without changing the frozen M3 inference path?

This is an exploratory pilot. It cannot enter the formal main table without a
second target seed, blind mechanism review, and a complete S50-to-target run.

## Frozen design

- Base: M3 / SOAR-SVTR, S50, alpha=0.15.
- Initialization: exact S50-pretrained M3 checkpoint used by frozen M3 target
  fine-tuning.
- Training: target Train only, canonical seed 20260731.
- Selection: minimum Clean Dev V4 Macro CER; earliest epoch breaks ties.
- Corrupted Dev and Test: forbidden.
- Global batch: 32, emulating the original two-rank 16+16 schedule on one GPU.
- SCDL weight: 0.05; no sweep.
- Warm-up: first 20% of optimizer updates have zero SCDL loss weight.
- Temperature: 0.1; same-script hard negatives: Top-5; no sweep.

## Method

RCTC frame features are pooled into character features using exact batched CTC
forward-backward token-state posteriors. Samples containing adjacent repeated
target characters are conservatively excluded from SCDL because positional
classes would otherwise alter CTC's mandatory-blank transition; their normal
M3 losses remain unchanged. The posterior is detached, so SCDL cannot alter
the alignment estimator through a shortcut. A checkpointed
cumulative-mean prototype bank is maintained for Han, Arabic, and Cyrillic
characters; common characters such as punctuation and digits are excluded.

For each character, only populated prototypes from the same Unicode script are
eligible negatives. The five most similar incorrect prototypes form the hard
negative set. Han, Arabic, and Cyrillic losses are averaged with equal weight.
The projection is identity-initialized. During evaluation, the projection,
alignment, mining, and prototype operations are not executed, so the prediction
path is identical to M3.

## Decision rule

The pilot is promising only if all of the following hold:

- Clean Dev V4 Macro CER improves by at least 0.02 percentage points;
- at least two of ZH, UG, and KK improve in CER;
- no language CER worsens by more than 0.10 percentage points;
- no language Line Accuracy worsens by more than 0.5 percentage points.

A smaller positive Macro CER change is weak evidence and permits only one
additional target seed. A failed primary or safety gate stops SCDL without a
lambda, temperature, Top-K, or warm-up search.
