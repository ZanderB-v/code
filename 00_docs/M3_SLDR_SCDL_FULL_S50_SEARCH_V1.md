# M3 SLDR/SCDL Full-S50 Study V1

## Frozen Base Recipe

- Baseline: M3 / SOAR-SVTR, Clean Dev V4 Macro CER `1.5059%`.
- Data: S50 synthetic pretraining, then the frozen target-domain Train split.
- Selection: minimum Clean Dev V4 Macro CER; earliest epoch breaks an exact tie.
- Normalization: `normalization_v2` through the frozen Clean Dev V4 protocol.
- COC alpha: `0.15`.
- Seed: `20260731` for the first decision run.
- Global batch: `32`; optimizer, learning-rate schedule, augmentation, maximum
  epochs, evaluation interval, and early-stopping rule match frozen M3.
- Corrupted Dev and Test are forbidden during search and model selection.

## Standard Full Runs

### SLDR_FULL_S50_V1

- SLDR is active during both S50 pretraining and target fine-tuning.
- Reduction: `4`.
- Initial script residual scale: `1e-3`.
- Local branches: `3x3`, `1x5`, and `5x1` depthwise convolutions.
- Fusion: sum.
- Script-conditioned gate: enabled.

### SCDL_FULL_S50_V1

- SCDL is active after a fixed `20%` update warmup in both stages.
- Weight: `0.05`; Top-K: `5`; temperature: `0.1`.
- Hard-negative mining is same-script and the loss is script-balanced.
- The target stage resets prototype vectors and counts, while retaining the
  learned projection and all M3 model weights from S50.
- SCDL is training-only and adds no inference path.

## Predeclared Decisions

- Clear positive: Macro CER below `1.4859%` (at least `0.02 pp` better), CER
  improves for at least two languages, no language CER worsens by more than
  `0.10 pp`, and no language Line Accuracy drops by more than `0.5 pp`.
- Weak positive: Macro CER is in `[1.4859%, 1.5059%)` with safety checks passed;
  run a second seed before deciding.
- No improvement: Macro CER is at least `1.5059%`, or a safety check fails;
  stop that track.

## Locked Sequential Search Space

Search starts only if the corresponding standard full run is not rejected.
Each phase fixes the winner from the preceding phase. No extra values may be
added after observing Clean Dev V4.

SLDR:

1. `gamma_init`: `1e-4`, `1e-3`, `1e-2`, with reduction `4` and V1 kernels.
2. `reduction`: `2`, `4`, `8`, with the selected gamma and V1 kernels.
3. Kernels: V1 three-branch versus `3x3 + 5x5`, with selected gamma/reduction.

SCDL:

1. `lambda`: `0.02`, `0.05`, `0.10`, with warmup `20%`, Top-K `5`.
2. Warmup: `10%`, `20%`, `30%`, with selected lambda and Top-K `5`.
3. Top-K: `3`, `5`, `10`, with selected lambda/warmup.
4. Temperature remains fixed at `0.1`.

The standard V1 configuration is reused within each search axis and is never
rerun merely to fill a grid. Best SLDR and best SCDL proceed to three seeds.
Only if both are positive is one frozen `M3 + SLDR + SCDL` combination run.

## Launch

The two standard tracks run concurrently in separate windows of one tmux
session and both see only physical GPU 0:

```bash
bash scripts/svtrv2/run_m3_full_s50_parallel_tmux.sh --preflight-only
tmux attach -t m3_full_s50_v1_parallel

# After both preflights pass:
bash scripts/svtrv2/run_m3_full_s50_parallel_tmux.sh --replace
tmux attach -t m3_full_s50_v1_parallel
```

Use `Ctrl-b n` and `Ctrl-b p` to switch between the `sldr` and `scdl`
windows. After interruption, restart without `--replace` to use audited resume.
