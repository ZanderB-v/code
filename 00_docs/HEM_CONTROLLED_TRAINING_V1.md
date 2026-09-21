# HEM Controlled Training V1

## Research role

HEM V1 is a controlled training-strategy experiment, not a fourth SOAR-SVTR
architectural contribution. The architectural method remains Dual-Order Semantic
Guidance, Cross-Order Consistency, and Script-Conditioned Adaptation.

## Immutable inputs

Before training, `freeze_hem_v1.py` verifies and snapshots:

- the completed 165-row manual review;
- the authorized 66,854-row HEM weight manifest;
- the frozen target-domain Train metadata.

It reports review acceptance by `language x ED`, verifies that all 49 manually
rejected candidates retain weight 1.0, and records SHA-256 for all three files.
The freeze directory is immutable and has no replace mode.

Mining inference is aspect-ratio grouped, so its source manifest is not assumed
to share the frozen Train row order. Freezing verifies one-to-one sample-ID
membership and label/image equality, then writes a canonical HEM manifest in
the exact frozen Train and LMDB file-index order. Both the audited source hash
and canonical training hash are retained as provenance.

## Sampling

Weights are fixed at ED0=1.0, ED1=2.0, ED2=1.5, and ED>=3=1.0. Weighted
sampling occurs with replacement inside each aspect-ratio bucket. The sampler
emulates the original two-rank partition, then concatenates each pair of
equal-width rank batches into one physical-GPU batch. It therefore preserves
the control's per-bucket sample count, dynamic-width schedule, and exactly 3,065
optimizer updates per epoch, matching the original global-batch-32 B1 and M3
target runs.

## GPU and batch policy

- Physical GPU: GPU1 only.
- Distributed world size: 1.
- Batch per card: 32.
- Global batch: 32, unchanged from the original two-card 16+16 controls.
- P1/MSR dynamic width remains active, so long-line batches shrink according to
  the frozen ratio rule.
- FP32, optimizer, scheduler, warmup, epoch limit, augmentation, seed, and early
  stopping are unchanged.

Batch 32 is the largest scientifically matched batch. Increasing it further only
to occupy memory would change the optimization protocol. Pinned memory, six
workers, persistent workers, and prefetch factor three are used to reduce GPU
input stalls. Actual memory use is reported by the smoke test.

## Preflight

Both B1+HEM and M3+HEM must pass:

1. configuration and initialization-hash validation;
2. HEM sampler DDP/control parity and deterministic replay;
3. 200 real training steps on physical GPU1;
4. increased hard-sample frequency;
5. finite loss and gradients;
6. no checkpoint writing and no Test evaluation.

Smoke results are operational evidence and are excluded from reported results.

## Formal runs

Only two new target-domain runs are allowed:

- B1+HEM, initialized from the same frozen B1 S50 checkpoint as B1;
- M3/SOAR-SVTR+HEM, initialized from the same frozen M3 S50 checkpoint as M3.

Both use the same frozen HEM manifest. Checkpoint selection is exclusively the
minimum Clean Dev Macro CER. Line Accuracy, Corrupted Dev, and Test cannot select
a checkpoint. Test remains prohibited.

Additional seeds are allowed only if the canonical M3+HEM run reaches both
Macro CER <= 2.25% and Macro Line Accuracy >= 89.2%.

The final controlled comparison is written to both JSON and CSV under
`04_model_training/eval_reports/hem_v1_controlled_comparison.*`. It includes
all four metrics overall, all four metrics for each language, control deltas,
the predeclared retention gates, and the additional-seed decision.
