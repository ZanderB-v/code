# Direction Study V2: phases 4 and 5

This revision implements the supplied follow-up recommendations. It supersedes
the V1 experiment entry point, but preserves V1 and historical model records.
No server training has been performed by the local code implementation.

## Fixed design

| Item | M3 anchor | D1 | D2 |
| --- | --- | --- | --- |
| Consistency alpha | 0.30 | 0.30 | 0.30 |
| CTC label | U2 visual | U2 visual | U2 visual |
| SGM label | Unicode logical | Unicode logical | Unicode logical |
| Shared Script Adaptation | Enabled | Enabled | Enabled |
| Direction affects CTC/shared features directly | No | No | No |
| Direction affects SGM input | No | Yes | Yes |
| New direction/gate losses | None | None | None |
| Script direction gate | None | None | One scalar per script |
| Gate initial logit | N/A | N/A | -4 for every script |
| Target initialization | Same M3 S50 checkpoint and SHA256 for all three |

The loss dictionary is exactly the M3 dictionary with alpha locked to 0.30:
the existing RCTC coefficient, SGM coefficient and script loss are preserved.
In the current config these are 0.1, 1.0 and 0.1, respectively. Direction loss
stays zero. No gate penalty, sparsity penalty or new-parameter LR multiplier
is introduced. The COC posterior transport and bidi permutation implementation
are unchanged.

M3's optimizer, weight decay, batch size, scheduler, warmup, AMP, clipping,
augmentation, charset, P1_MSR_V3, Protocol V2 and canonical seed are inherited
from its YAML template. A parsed before/after comparison enforces that only
the output paths, initialization pointer, protocol identity, locked alpha,
and new semantic direction fields can change.

## Anchor provenance

The historical M3 result CER=2.2987% belongs to alpha=0.15 in the available
config. It cannot be relabeled as alpha=0.30 or used as the matched numerical
threshold. The actual alpha=0.30 anchor CER is read from evaluated checkpoints.

When a completed M3 alpha=0.30 target summary is available, the runner verifies
it and reuses its S50 initialization for D1 and D2. By default it checks:

`04_model_training/eval_reports/svtrv2_s_m3_alpha030_dual_order_s50_to_target_final_summary.json`

An alternative M3 alpha=0.30 final summary under eval_reports can be provided
with `--anchor-summary PATH`; use the same argument on preflight and resume.

If the anchor is absent, the prerequisite is automatically prepared:

1. Clone the historical M3 synthetic config, lock alpha=0.30, and run its
   S50 stage from the same audited converted RCTC initialization.
2. Clone its target config and train the alpha=0.30 M3 anchor.
3. Train D1 and D2 independently from that exact shared S50 checkpoint.

Thus there are two new direction architectures. Building a missing matched
anchor is an additional prerequisite, not a third direction architecture.
No direction model receives an extra pass of target training from a trained
M3 target checkpoint. D2 never starts from D1's target checkpoint.

The initialization smoke requires all shared M3, SGM and RCTC state entries
to load with matching shapes. Only the newly introduced semantic direction
and gate parameters may be missing. A checkpoint containing existing direction
weights is rejected as the shared M3 source. Reports record its hash and the
fresh parameter keys.

## Direction and scalar gates

Let F = ScriptAdapter(Encoder(x)). CTC always receives F. D1 feeds F + A(F)
to SGM. D2 feeds F + sigmoid(theta_s) A(F) to SGM. There is no reversal.
Direction scale and bias embeddings initialize to zero, so the residual is
initially zero. All four theta values initialize to -4, giving gate=0.017986.
The gate is near zero, not mathematically zero. Tests check that gate gradients
become nonzero after the first residual update.

D2 obtains one sample-level script ID by averaging the existing script-head
posteriors across visual columns and taking the most probable class. It indexes
one of four learned scalar gates; that scalar broadcasts across the entire
sample feature map. No width/token gate is used and no GT language ID is fed
to the model. Mixed-script lines use their predicted dominant script.

Classes retain the existing IDs: Han, Arabic, Cyrillic and Common/Latin.
The last class must not be described as a newly annotated pure-Latin class.
No script receives a hardcoded special gate. Gates may open or close.

With no auxiliary direction supervision, the direction head's two latent
channels are not certified LTR/RTL predictions. These experiments test whether
this semantic conditioning helps; they do not establish direction recognition
accuracy or causality. Shared-encoder gradients also allow indirect effects
on CTC during joint training despite direct feature-path isolation.

## Training and selection

M3's original evaluation cadence and early-stop controls are copied exactly.
After each target stage, every saved epoch from 1 through its stopping epoch
is evaluated on Clean Dev. This adds observation, not training epochs. It
also preserves the original stopping decisions instead of retuning patience.

Each epoch exports Macro CER, Macro WER, Macro 1-NED and Macro Line Accuracy;
D2 additionally exports four learned gates. Per-language CER and Line Accuracy
are included. The exporter incrementally writes CSV/JSON and binds cache entries
to checkpoint, config, evaluator and result hashes. Missing epoch checkpoints or
mixed reports fail explicitly. It does not substitute old metrics for missing data.

Actual minimum Macro CER across all saved epochs selects each of M3/D1/D2.
All reported metrics on that row come from the same epoch. Ties prefer the
earlier epoch. Historical B1/M1/M2 results retain their audited selection scope;
do not claim their full saved-epoch coverage unless that prior audit was completed.

**D2 runs after D1 even if D1 does not improve.** No D1 performance gate remains.
The language comparison uses the matched alpha=0.30 M3, with all changes in
percentage points. In particular, inspect UG CER and UG Line Accuracy against
the actual new anchor, not a copied historical target number.

Phase 5 ranks eligible models by Clean Dev Macro CER; eligibility requires CER
below B1 and Macro Line Accuracy above B1. Line Accuracy is not a checkpoint
selector or a tie breaker. Original Full appears only as
"M3 + LDC (Diagnostic Ablation)". Historical M3 alpha=0.15 is clearly labeled.

If neither direction variant improves the matched anchor and passes the B1
guardrail, record stop_direction_route. If one does, record a candidate requiring
three-seed validation. No one-seed result automatically replaces the frozen
SOAR-SVTR method. Gate values alone are not evidence of causal script preferences.
Do not use Corrupted Dev or Test for any of these decisions. Test is never run.

## Server commands

Upload `semantic_direction_v2_sync_20260905.zip` to the project root:

```bash
cd /data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition
unzip -o semantic_direction_v2_sync_20260905.zip
sha256sum -c 00_docs/semantic_direction_v2_SHA256SUMS
bash scripts/svtrv2/run_semantic_direction_v2_tmux.sh --preflight-only
tmux attach -t semantic_direction_v2
```

When the anchor exists, require DIRECTION_V2_ALL_TARGET_PREFLIGHTS_OK and exit 0.
If the anchor must first be built, preflight ends with
DIRECTION_V2_ANCHOR_PREFLIGHT_OK and explicitly states that target preflights
are pending. In the formal run, those preflights execute once the source exists,
before any target-stage training.

```bash
bash scripts/svtrv2/run_semantic_direction_v2_tmux.sh
tmux attach -t semantic_direction_v2
```

Use the same command to resume; do not add --replace. The tmux wrapper retains
dead panes and reports DIRECTION_V2_EXIT_CODE. A live session is not killed.
The fixed log is `04_model_training/logs/semantic_direction_v2/study.log`.
Reference hashes and summaries must agree before formal work begins; stale
M2 alpha=0.30 records are a provenance error, not something to edit numerically.

Completion marker: DIRECTION_V2_PHASE4_PHASE5_COMPLETE.

## Outputs and local checks

Outputs are under `04_model_training/eval_reports/semantic_direction_v2/`:

- shared_initialization.json
- M3/D1/D2_epoch_metrics.csv and .json
- M3/D1/D2_cer_selected.json
- language_comparison.csv and .json
- structure_selection.csv and .json
- epoch_macro_cer.png/.pdf and epoch_macro_line_accuracy.png/.pdf
- D2_gate_curves.png/.pdf
- all_epoch_dev/ with per-epoch source reports and hash bindings

Synchronize that directory, the study log directory, generated configs, and
the corresponding run best-selection JSONs after completion. Preserve all
server epoch checkpoints for the complete-epoch export and reproducibility.

Local CPU tests cover legacy component regression, initial identity, direct CTC
gradient isolation, no-new-loss branch gradients, scalar gate routing, learnable
zero-residual initialization, exact config inheritance, all-epoch CER selection,
cache tamper detection, language delta units and plot export. The launcher adds
a three-step two-GPU component reducer test plus real-data forward/backward and
sampler checks. Local tests do not certify the server's CUDA/data environment.
