# S50 M3 Targeted Refinement Protocol V1

## Scope

This protocol is the final evidence-driven architecture-development branch for
SOAR-SVTR. It uses only frozen Protocol V2 Train and Clean Dev. Corrupted Dev
does not select models or hyperparameters, and Test is prohibited.

The fixed anchor is S50 M3 / SOAR-SVTR with consistency alpha 0.15. Its
checkpoint is selected by minimum Clean Dev Macro CER.

## Stage A: Required diagnostics

Complete both evidence tasks before implementing another model component:

1. Build the final M3 ZH/UG/KK error profile and punctuation audit.
2. Train M3-noCOC through the complete S50 synthetic pretraining and target
   fine-tuning pipeline.

The multilingual profile and punctuation review run first. M3-noCOC must not
start until one punctuation task definition is frozen. If punctuation is
excluded, all saved Clean-Dev epochs must be rescored and checkpoints reselected
under the new primary metric before M3-noCOC. If full transcription is selected,
the GT repair and data audit must finish first.

M3-noCOC then changes only `consistency_weight: 0.15 -> 0.0`. It retains S50,
the exact pre-SGM RCTC initialization, P1_MSR_V3, dual-order labels, Script
Adaptation, optimizer, global batch size, seed, schedule, early stopping, and
Clean Dev Macro CER checkpoint selection.

The error profile reports substitutions, deletions, insertions, ED, line
confidence, and confusion pairs for all three languages. It additionally
reports Chinese punctuation mismatches, Uyghur Arabic-run positions, and
Kazakh duplicate insertions. Kazakh duplicate claims require the pre-collapse
CTC argmax path to show same-character peaks separated by blank.

The punctuation audit never silently changes the frozen metric. It reports raw
metrics and a candidate metric that removes the same explicit Unicode set from
both GT and prediction. A human must select exactly one task definition:

1. punctuation is outside the recognition target, in which case the exact set
   is frozen and every model/split is recomputed from raw predictions; or
2. full transcription includes punctuation, in which case missing GT
   punctuation is repaired before further training.

No Chinese-only prediction suppression or punctuation model is allowed.

## Stage B: SLDR authorization

Script-Aware Local Detail Refinement (SLDR) is authorized only when all Chinese
and Uyghur error lines are manually reviewed, at least 20 per language are
reviewed, at least 50% of Chinese errors are classified as local/degraded
visual detail, and at least 50% of Uyghur errors are classified as either
`local_visual_detail` or `joining_or_contextual_form`. Punctuation and GT
annotation mismatches do not count as evidence for SLDR.

The first SLDR implementation is deliberately small:

```text
F3 = DWConv3x3(Fs)
F5 = DWConv5x5(Fs)
Flocal = Proj(F3 + F5)
Fout = Fs + gscript(Fs) * Flocal
```

The output projection is zero initialized, so the initial network is
functionally M3. SLDR is placed after Script Adaptation and before both CTC and
SGM. The first experiment changes no other factor.

Continue SLDR only if the canonical-seed result improves Macro CER and does not
degrade UG CER. An improvement to roughly 2.27% requires seed confirmation;
an improvement to 2.20% or better with clear ZH/UG visual-error reductions is a
strong candidate. Punctuation-only gains do not validate the mechanism.

## Stage C: segment-count authorization

Segment-count regularization is authorized only when Clean-Dev KK has at least
10 insertion events, at least 50% are adjacent duplicate insertions, and at
least 90% of those duplicates are confirmed as blank-separated peaks in the
raw CTC path.

If authorized, first test the count loss on M3 without SLDR:

```text
q_t = sum(c != blank) p_t(c) * (1 - p_(t-1)(c))
L_hat = sum(t) q_t
L_count = SmoothL1(L_hat, L_gt)
```

The weight is selected from one predeclared small grid on Clean Dev and is
never selected using KK alone. Legal repeated GT characters remain valid
because the target is the full GT length. No inference-time repeated-character
deletion is permitted.

Only after independent SLDR and count experiments both pass may their
combination be trained.

## Frozen selection and stopping rules

- Primary: minimum Clean Dev Macro CER.
- Guardrail: Macro Line Accuracy must remain above the frozen B1 value.
- Report all three languages and all four metrics from the selected epoch.
- A difference within 0.02 percentage points Macro CER is treated as
  inconclusive at one seed.
- No more attention blocks, decoders, augmentations, alpha searches, HEM, or
  direction modules may be added during this branch.
- A passing new structure receives two additional seeds before any Test run.

## Outputs

Stage A writes under:

```text
04_model_training/eval_reports/s50_m3_targeted_diagnostics_v1/
```

Required artifacts include `error_profile_summary.json`,
`punctuation_protocol_candidate_v1.json`, `decision_gate.json`,
`duplicate_insertion_events.csv`, `zh_error_review.{csv,html}`,
`ug_local_detail_review.{csv,html}`, and `m3_nococ_comparison.{json,csv}`.

The task-definition review is additionally materialized as
`04_model_training/eval_reports/S50_M3_error_review_v1/`. It contains one
master `all_errors.csv`, language-specific `all_errors.csv` files and review
pages, deterministic automatic candidate categories, and a generated
`label_patch_v1.csv`. Automatic categories never alter GT. A patch row is
created only after a reviewer selects `punctuation_gt_missing` or
`gt_annotation_error` and supplies a non-empty corrected GT. The frozen
Clean-Dev source remains immutable. Each review page exports
`<language>_reviewed_errors.csv`; copy that file back to the review root and
rerun the builder to regenerate the summary and patch without losing reviews.
