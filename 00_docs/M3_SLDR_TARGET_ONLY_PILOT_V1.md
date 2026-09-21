# M3 + SLDR Target-Only Pilot V1

## Status

This is an exploratory pilot, not a formal main-table experiment. Clean Dev V4
found 23 local-glyph/joining errors among 60 genuine ZH+UG errors (38.33%),
below the predeclared 40% authorization threshold. A positive pilot therefore
requires blind mechanism-error review and a complete S50-to-target rerun before
SLDR can be considered for the final recipe.

## Frozen design

- Base: M3 / SOAR-SVTR, S50, alpha=0.15.
- Initialization: the exact S50-pretrained M3 checkpoint used by the frozen M3
  target fine-tuning run.
- Training: target-domain train only, canonical seed 20260731.
- Selection: minimum Clean Dev V4 Macro CER.
- Test and Corrupted Dev: forbidden.
- Global batch and optimizer schedule: frozen at 32, emulating the original
  two-rank 16+16 RatioSampler schedule on one physical GPU.

SLDR is inserted after the SVTRv2 2-D visual encoder and before both the M3
script adapter and RCTC feature rearrangement. It contains:

- 1x1 reduction with ratio 4;
- depthwise 3x3, 1x5, and 5x1 local branches, fused by summation;
- channel, height, and width recalibration;
- a script-conditioned residual using the existing M3 script posterior;
- four LayerScale values initialized to 0.001.

No direction reversal, AOM, new decoder, new loss, dictionary change,
augmentation change, or alpha search is allowed.

## Decision rule

The pilot must improve Clean Dev V4 Macro CER and pass both safety guardrails:

- no language CER worsens by more than 0.10 percentage points;
- no language Line Accuracy worsens by more than 0.5 percentage points.

ZH or UG CER must improve to support the proposed mechanism. A Macro CER gain
below 0.02 percentage points is only weak-positive evidence. A promising result
still requires a blind error review showing fewer local-glyph errors, followed
by a complete S50 synthetic pretraining to target fine-tuning experiment.
