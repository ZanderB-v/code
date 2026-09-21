# Clean Dev V2 M3 Error Profile V1

## Fixed source

- Model: M3 / SOAR-SVTR
- Synthetic scale: S50
- Cross-order consistency alpha: 0.15
- Selected checkpoint: target fine-tuning epoch 34
- Selection metric: minimum Clean Dev V2 Macro CER
- Clean Dev V2 rows: 951 (ZH 346, UG 295, KK 310)
- Corrupted Dev and Test are prohibited.

The profile reuses the immutable per-sample predictions written by the Clean
Dev V2 checkpoint reselection. It does not run inference or change labels.

## Manual categories

Chinese errors distinguish similar Han characters, local stroke/component
differences, small or blurred text, outline/shadow/artistic fonts, other
substitutions, deletions, insertions, punctuation/normalization, ambiguous or
cropped images, and other errors.

Uyghur errors distinguish local glyph confusion, joining-form confusion,
order-related errors, other substitutions, deletions, insertions,
punctuation/normalization, ambiguous or cropped images, and other errors.

Kazakh errors distinguish glyph confusion, other substitutions, deletions,
insertions, punctuation/normalization, ambiguous or cropped images, and other
errors. An automatic adjacent duplicate candidate is not treated as proof of a
duplicate-alignment failure.

## Frozen gates

SLDR is authorized only after every error line is reviewed, there are at least
40 genuine ZH+UG error lines, local-glyph or joining-form errors account for at
least 40% of those genuine errors, and local glyph or joining form is the
largest error family.

Segment-count or duplicate-alignment work remains stopped. It may be reopened
only if the new manual review confirms genuine duplicate errors and a separate
CTC raw-path audit confirms blank-separated repeated peaks.

M3-noCOC may start after the error-profile review is complete. It changes only
the consistency weight and does not alter alpha search, data scale,
augmentation, labels, or checkpoint selection.
