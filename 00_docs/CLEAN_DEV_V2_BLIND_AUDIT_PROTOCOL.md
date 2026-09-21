# Clean Dev V2 Model-Blind Audit Protocol

## Purpose

The 24 corrections discovered through frozen M3 errors are diagnostically
valuable but selection-biased. They cannot independently define Clean Dev V2.
Clean Dev V2 therefore requires a model-blind review of all 951 original Dev
rows.

## Blindness rule

The review package is built only from the original Dev metadata and line crops.
It displays sample ID, language, image, and current GT. It does not load or
display predictions, model names, confidence, edit distance, or error type.

Allowed decisions are:

- `gt_correct`
- `gt_annotation_error`
- `normalization_issue`
- `ambiguous_image`
- `crop_issue`

`gt_annotation_error` requires a changed, non-empty corrected GT. Other
decisions must not silently create per-sample text patches. Normalization issues
are resolved only by a globally shared rule applied identically to GT and
prediction.

## Immutability

The original 951-row Dev data are retained as `clean_dev_v1_raw`. The audit
creates a candidate patch table without modifying the original files.
Ambiguous or cropped rows remain present unless a model-independent exclusion
rule is declared and applied to all 951 rows before any model is evaluated.

## Development gate

No Clean Dev V2 manifest is generated and no new module training is authorized
until:

1. all 951 rows have valid blind decisions;
2. every GT correction is complete;
3. normalization V2 is implemented, tested, and frozen;
4. the V2 manifest and all source artifacts are hashed.

After freezing, all saved B1/M1/M2/M3 checkpoints, the fixed M3 alpha grid, and
the controlled S25/S50 checkpoints are re-evaluated. Checkpoints remain selected
only by minimum Clean Dev V2 Macro CER. Corrupted Dev and Test remain prohibited.

Normalization V2 is deliberately minimal and model-independent. It uses NFC,
maps only `U+FF5E FULLWIDTH TILDE` to ASCII tilde, canonicalizes whitespace,
and preserves internal spaces and all other punctuation. It does not use global
NFKC, punctuation removal, Han-space removal, or silent bidi-control deletion.
