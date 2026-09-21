# M3 to SCDL Diagnostic V1

This is a diagnostic only. M3 / SOAR-SVTR remains the frozen candidate.
SLDR full-S50 is closed; no new model, hyperparameter search, Corrupted Dev,
or Test evaluation is authorized by this protocol.

## Inputs

- Frozen Clean Dev V4: 940 evaluated lines, normalization_v2.
- M3 alpha=0.15, V4 CER-selected epoch 34 predictions.
- M3+SCDL full-S50 selected target checkpoint predictions.
- For internal analysis only: the selected SCDL target checkpoint and a fixed
  sample of target Train batches on physical GPU0.

## Outputs

The local paired analysis writes `05_evaluation/m3_scdl_diagnostic_v1/`:

- `migration_by_language.csv`: Recovered, Regression, both-wrong movements.
- `edit_operation_changes.csv`: deterministic minimum-edit Sub/Del/Ins totals.
- `changed_error_rows.csv`: line-level GT, both predictions, ED, edit counts.
- `ug_regressions_review.csv`: four UG regressions awaiting image-based review.
- `summary.json`: source hashes and V4 protocol binding.

The server-only, read-only internal audit writes `internal_v2/`:

- `script_diagnostics.csv`: per-script character loss, CTC target probability,
  temporal concentration, positive/top-K negative similarity, bank coverage.
- `hard_negative_examples.csv`: up to 100 deterministic reservoir samples per
  script, with Unicode character identities.
- `summary.json`: checkpoint/config hashes and fixed Train sampling settings.

The internal metrics are a **snapshot at the selected checkpoint** on a fixed
Train sample. They are not a reconstruction of the historical training loss or
proof of a causal mechanism. Prototype counts are token observations, not
unique training samples. The forward pass uses training mode to expose SCDL
features but never updates weights or prototype buffers.

The first `internal/` run computed `mean_ctc_target_probability` with a
prototype class ID instead of CTC class ID. Since CTC class 0 is blank, its
target-probability field is invalid and must not be interpreted. All other
statistics in that run remain valid. `internal_v2/` fixes the offset and keeps
the first result untouched for provenance.

## Interpretation gate

1. Review all four UG Regression images before assigning a mechanism. Edit
   operations are automatic candidates, not image-grounded labels.
2. Compare Han, Arabic, and Cyrillic only after confirming adequate usable
   token counts and prototype coverage for each script.
3. A reliability-gated variant is justified only if low CTC target probability
   or diffuse alignment is concentrated in the relevant UG failures.
4. A hard-negative-filtering variant requires concrete negative-character
   evidence; do not hand-select characters from Dev.
5. If no coherent pattern emerges, close SCDL instead of tuning lambda.

The 4891-character dictionary contains no Arabic Presentation Forms codepoints.
Joining variants of the same base Unicode character therefore share a class
prototype; they cannot directly be distinct hard negatives in SCDL V1.

## Completed review and internal V2 readout

The user's four-line UG review is recorded in
`05_evaluation/m3_scdl_diagnostic_v1/ug_manual_review_notes_v1.md`:
two confirmed OCR regressions (one character substitution, one extra colon)
and two visually unresolved lines. The unresolved lines are not evidence of GT
errors; frozen Clean Dev V4 and the published paired metrics remain unchanged.

At the selected target checkpoint, `internal_v2/` sampled 64 target-Train
batches (336 lines, 259 with valid CTC alignment). Corrected mean aligned CTC
target probabilities are Han 0.9865, Arabic 0.9795, and Cyrillic 0.9565.
Arabic alignment peak concentration is 0.8543, versus 0.8495 for Han and
0.8361 for Cyrillic. Arabic prototype coverage is 40/47 characters. These
aggregate Train-only snapshots do not show a uniquely low-confidence or
diffuse-alignment Arabic mechanism. They also do not resolve the two ambiguous
Dev images or establish a causal explanation for SCDL's UG regression.

On frozen Clean Dev V4, SCDL changes Macro CER from 1.5059% to 1.4916%, but
Macro Line Accuracy from 92.0084% to 91.9842%. UG CER worsens from 0.9609%
to 1.0225%, and UG Line Accuracy falls from 93.7931% to 93.1034%. This is a
weak aggregate CER gain with a language-specific guardrail failure. Keep M3 as
the current model; do not promote or tune SCDL based on this diagnostic.
