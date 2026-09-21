# B1 S25/S50 Equal-Step Scale Selection V1

## Purpose

Select the final synthetic-data scale with official full SVTRv2-S (B1) while
changing only the nested synthetic subset. This protocol supersedes the old
same-epoch S25/S50 comparison for scale selection.

## Frozen Controls

- Preprocessing: `P1_MSR_V3`, dynamic-width MSR and U2 labels.
- Hardware schedule: physical GPU0 only, one process, FP32, batch 32.
- Stage 1: SVTRv2-S + RCTC initialized from the same Union14M checkpoint.
- Stage 2: full B1, with each arm initialized from its own terminal Stage-1
  checkpoint after the same number of optimizer updates.
- Each synthetic stage: 115,350 optimizer updates and 4,614 warmup steps.
- Stage transitions use the exact terminal checkpoint. Clean Dev does not pick
  an intermediate synthetic checkpoint.
- Target fine-tuning uses the same data, optimizer, step-based scheduler,
  early stopping rule, seed, and Clean Dev Macro CER checkpoint selection.
- Corrupted Dev and Test cannot participate in scale selection. Test is not run.

The code audits actual sampler steps before training. A partial final epoch is
allowed so S25 and S50 stop at exactly the same optimizer update without relying
on an assumed 1:2 epoch-length ratio.

## Selection Rule

Choose the lower Clean Dev Macro CER after target fine-tuning. If the absolute
difference is below 0.02 percentage points, select S25 as the lower-cost tie.
The resulting `final_scale_lock.json` fixes the scale for B1, M1, M2, M3, CRNN,
SVTR, PARSeq, and ABINet.

No synthetic-validation split is invented after protocol freeze. Stage-A target
Dev measurements are diagnostic only and never choose the checkpoint passed to
the next stage.
