# Semantic Direction Study V1

This prospective study implements phases 4 and 5. Alpha is locked to 0.30.
The old M3 / SOAR-SVTR result belongs to alpha=0.15. Its scores must not be
relabeled as alpha=0.30. The existing Full checkpoint is displayed as
"M3 + LDC (Diagnostic Ablation)"; its files and frozen records remain intact.

## Controlled Models

| Variant | Consistency alpha | Direction path | Script gate |
| --- | --- | --- | --- |
| m3_alpha030 | 0.30 | None | None |
| dir_d1 | 0.30 | SGM only | None |
| dir_d2 | 0.30 | SGM only | Four learned sigmoid gates |

The M3 alpha=0.30 run is a matched control, not a third direction variant.
Completed stages are reused only after their configs, checkpoint hashes,
selected epochs, metrics and data protocol agree.

Every synthetic stage starts from the same converted D2 RCTC checkpoint.
Each target stage starts from its own synthetic best checkpoint. D2 does
not inherit the trained D1 checkpoint. All three use seed 20260731, S50,
frozen target train/dev, P1_MSR_V3, the same dictionary and U2 label rules,
AdamW LR 2.5e-5, per-device first batch size 16 and 50-epoch scheduler horizon.

Let F be the output of the existing shared script adaptation:

- CTC input: F for all three variants.
- D1 SGM input: F + A_dir(F, predicted_direction).
- D2 SGM input: F + g(F) A_dir(F, predicted_direction).
- g(F)[w] = sum_s softmax(script_logits[w])[s] sigmoid(theta_s).

The four script classes retain the existing IDs: Han, Arabic, Cyrillic,
Common/Latin. They are predicted per visual column; no oracle language ID
or per-language hardcoded routing is used. All four theta values start at
logit(0.01), not minus infinity. Direction scale/bias embeddings start at
zero, giving exact residual identity for matched shared weights. The small
nonzero gates allow residual parameters to learn; gate gradients emerge
once the residual becomes nonzero. Gates may open or close during training.
Their final values are descriptive and do not alone prove causal benefit.

Both D1 and D2 retain the same 0.05 local-direction auxiliary loss as the
original LDC experiment. Script loss is 0.10. The directional targets use
the existing visual-position alignment. No feature reversal is applied.

Direction never transforms the CTC input directly. Semantic and direction
losses can still update the shared encoder through joint training. CTC-only
inference skips the semantic direction module entirely. B2 is a diagnostic
for order mismatch, not proof that this new intervention solves it.

## Decisions

Synthetic stage: evaluate every 5 epochs, patience 4 evaluations, minimum
15 epochs. Target stage: evaluate every epoch, patience 10 evaluations,
minimum 10 epochs. The target patience window remains ten epochs, matching
the previous every-2-epochs/five-evaluations budget. The finer target
selection grid must be disclosed when comparing with historical runs.
No retrospective claim of having evaluated unsaved historical epochs is made.

Checkpoint = actual argmin Clean Dev Macro CER over evaluated epochs;
min_delta=0.0001 controls patience only. Ties prefer the earlier epoch.
All four metrics come from that one checkpoint. Macro means an unweighted
average over zh, ug and kk; UG reporting retains the existing logical-order
heuristic and its documented limitations.

D2 starts only when D1 Clean Dev Macro CER is strictly below the matched
M3 alpha=0.30 CER. This is a single-seed screening rule, not significance.
D2 preflight may run earlier; it does not perform formal training.

Phase 5 includes B1, M1, M2 alpha=0.30, matched M3, D1 and (if run) D2.
Historical M3 alpha=0.15 and M3 + LDC are diagnostic rows. Candidates must
have CER below B1 and Macro Line Accuracy above B1. Among eligible candidates,
select minimum CER. Ties prefer the earlier simpler model in the table.
WER, 1-NED and Line Accuracy are never independent checkpoint selectors.
If no candidate passes, report that outcome without naming a winner.

The new selection is written to its own report, not silently applied to
the previously frozen method design. Corrupted Dev and Test are not used.
A new winner still needs the agreed multiseed validation before a final claim.

## Run on the Server

Synchronize the files in `semantic_direction_v1_sync_files.txt` before launch.

```bash
cd /data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition
bash scripts/svtrv2/run_semantic_direction_tmux.sh --preflight-only
tmux attach -t semantic_direction_v1
```

Require `SEMANTIC_DIRECTION_COMPONENTS_OK` (two-device checks), real-batch
forward/backward and full-epoch sampler audits, then
`SEMANTIC_DIRECTION_PREFLIGHT_OK` and `SEMANTIC_DIRECTION_EXIT_CODE=0`.

```bash
bash scripts/svtrv2/run_semantic_direction_tmux.sh
tmux attach -t semantic_direction_v1
```

Use the same command to resume. The fixed log directory preserves the
per-epoch metrics needed by the existing stage resume validator. A live
session is never killed. Dead panes retain their exit status and output.
There is no `--replace` option on this entry point.

```bash
tail -F 04_model_training/logs/semantic_direction_v1/study.log
```

Final outputs are under `04_model_training/eval_reports/semantic_direction_v1/`:
`structure_selection.json`, `structure_selection.csv`, `d2_decision.json`,
and, when D2 runs, `d2_learned_gates.json`. Synchronize this directory together
with the three variants' per-epoch/final reports, run selection JSONs,
configs, prepare summaries and `logs/semantic_direction_v1/`.

Preflight deliberately fails on stale references (for example, the previous
M2 alpha=0.30 best folder and final summary disagreeing). Reconcile those
with the actual selected checkpoint; never edit a metric to make it match.

## Verification Scope

Local CPU tests cover exact initial identity, direct CTC gradient isolation,
CTC output invariance after changing direction parameters, nonzero learnable
gate gradients after opening the residual, inference skipping the module,
and CER selection with the Line Accuracy guardrail. The server entry point
also performs a three-step two-GPU component reducer test and real-data model
forward/backward and sampler audits. Local tests do not certify server CUDA,
data availability or future training outcomes.
