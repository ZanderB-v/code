# Formal RCTC Experiment Runbook V2

## Scope

This runbook governs the P1/MSR `SVTRv2-S + RCTC` baseline and scale
experiments. It does not define or claim the later SGM or proposed dual-order
model.

## Dataset Terminology

- Chinese: natural Chinese meme lines.
- Uyghur: in-domain re-rendered Uyghur meme lines.
- Kazakh: in-domain re-rendered Kazakh meme lines.
- S10/S25/S50: independently rendered synthetic multilingual meme lines.

Uyghur and Kazakh target-domain lines must not be described as naturally
occurring real meme text.

## Hard Gates Before Training

1. Repair every exact or punctuation/spacing-normalized S50 overlap with target
   dev/test text.
2. Freeze Protocol V2 after the repair and retain all image, metadata, label,
   dictionary, generator, font, background, and metric hashes.
3. `freeze_protocol_v2.py --mode verify` must pass without rewriting the frozen
   manifest.
4. The formal research audit must report:
   - zero target `source_id` split leaks;
   - zero identical target image hashes across splits;
   - zero exact or normalized-near S50 overlaps with target dev/test;
   - zero synthetic U2 metadata mismatches.
5. Each generated E0/E1/D2 config must pass validation and a real LMDB
   forward/loss/backward/finite-gradient smoke test before long training begins.

## Frozen Development Policy

- Input preprocessing: `P1_MSR_V3`, using `RatioDataSetTVResize` and
  `RatioSampler`.
- External Clean Dev inference: per-sample dynamic width (`batch_size = 1`).
- Architecture: `SVTRv2-S + RCTC + CTC`.
- Uyghur CTC supervision: U2 visual order produced by `python-bidi`.
- Checkpoint selection: clean target-domain dev macro CER only.
- Test is not run during model development or checkpoint selection.
- The unified CTC dictionary is a fixed closed vocabulary. The audit reports
  dev/test characters not observed in target train or S50; disclose them and
  do not describe the results as open-vocabulary OCR.
- E0 and E1 use identical target data, preprocessing, optimizer policy, and
  target seed; initialization is the controlled factor.
- E1 and E5 use identical target fine-tuning policy; initialization is the
  controlled factor.

## Uyghur Reporting

The second `python-bidi` display pass is a heuristic, not a general inverse.
Every report must retain both:

- direct U2 visual-order CER;
- heuristic logical-order CER.

The heuristic is exact for all frozen dev ground-truth labels, but is not exact
for every possible mixed RTL/LTR prediction and has one known frozen test-label
round-trip mismatch. This limitation must be disclosed and motivates later
dual-order modeling.

## Baseline Chain

1. E0: random initialization, target-domain train.
2. E1: Union14M initialization, target-domain train.
3. D2: Union14M initialization, S50 synthetic train.
4. E5: D2 best clean-dev checkpoint, target-domain train.

Each final summary must include the data-protocol fingerprint, config hash,
checkpoint hash, evaluated epochs, best epoch, global step, and early-stopping
control.

## Scale Ablation

S10, S25, and S50 are nested subsets. The current design uses a fixed maximum
epoch policy with clean-dev early stopping. Larger datasets therefore receive
more optimizer steps per epoch. Report actual global steps and describe this as
a fixed-data-pass comparison. For a strong final paper, add either a fixed-step
control or a compute-matched supplementary experiment.

## Repetition And Reporting

- One seed is sufficient for pipeline validation and method development.
- Final headline comparisons should use at least three seeds when compute
  permits and report mean and standard deviation.
- Training fixes Python, NumPy, CPU/GPU Torch seeds and requests deterministic
  cuDNN kernels. Exact bitwise identity can still depend on CUDA, driver, and
  library versions, so retain the preflight environment record.
- Report CER, WER, macro 1-NED, and exact line accuracy by language.
- Keep short/medium/long, Uyghur mixed-direction, Kazakh-specific-character,
  and corruption analyses separate from checkpoint selection.

## Innovation Boundary

The hardened RCTC chain is the baseline, not the contribution. Full SVTRv2 with
standard SGM must be reproduced before claiming gains from dual-order semantic
guidance, cross-order consistency, or script-conditioned adaptation.
