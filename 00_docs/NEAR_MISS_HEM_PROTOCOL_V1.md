# Near-Miss HEM Protocol V1

## Eligibility

Formal HEM is authorized only when the completed Stage 6 ED=1 review contains
at least 70% `genuine_ocr_error` rows. The historical Stage 6 V1 result and its
50% diagnostic gate remain unchanged for provenance; this protocol adds the
stricter training-authorization gate before any HEM experiment.

If the gate fails, do not train HEM models. Correct target-train labels and the
normalization policy first, then create a new versioned audit. Do not relabel or
delete Dev/Test examples to improve the eligibility result.

## Frozen HEM design

- Mine only the frozen target-domain Train split with M3 / SOAR-SVTR alpha 0.15.
- Compute exact character edit-distance buckets: ED=0, ED=1, ED=2, ED>=3.
- Exclude known or automatically detected label, Unicode, crop, visibility, and
  illegal-character risks before assigning extra sampling weight.
- Use weights ED=0: 1.0, ED=1: 2.0, ED=2: 1.5, ED>=3: 1.0.
- Use physical GPUs 0 and 1 with DDP world size 2.
- Keep batch size 16 per card and global batch size 32, matching the frozen B1
  and SOAR-SVTR controls. Do not increase batch size merely to occupy memory;
  improve input-pipeline throughput without changing optimization semantics.
- Preserve P1_MSR_V3, dictionary, U2 rules, optimizer, epoch length, and the
  Clean Dev Macro CER checkpoint-selection rule.
- Corrupted Dev is diagnostic only. Test remains unopened.

## Controlled experiment matrix

Only two new runs are allowed after eligibility and target-train quality gates:

1. B1 + HEM
2. SOAR-SVTR + HEM

B1 and SOAR-SVTR without HEM are the frozen controls. Do not change alpha,
architecture, augmentation, or direction conditioning in the HEM experiment.

HEM is retained only if SOAR-SVTR + HEM improves both Clean Dev Macro CER and
Macro Line Accuracy over SOAR-SVTR, with per-language results disclosed. The
predeclared practical target is Macro CER <= 2.25% and Macro Line Accuracy >=
89.2%; three-seed confirmation is required before a paper claim.
