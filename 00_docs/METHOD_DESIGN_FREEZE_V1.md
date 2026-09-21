# Method Design Freeze V1

## Decision

This protocol closes clean-dev architecture and hyperparameter development for
the Chinese, Uyghur, and Cyrillic Kazakh meme-line recognizer.

The primary proposed model is `M3`:

- RCTC/CTC supervision: U2 visual order;
- SGM supervision: Unicode logical order;
- permutation-transported cross-order consistency;
- image-inferred script-conditioned visual adaptation;
- no local direction conditioning.

The trained M3 checkpoint uses `consistency_weight = 0.15`. It must not be
described as an alpha-0.30 model.

The official M2 consistency ablation uses `consistency_weight = 0.30`, selected
before this freeze from clean target-domain dev. A matched M2 alpha-0.15 result
is retained only to isolate the Script Adapter when comparing M2 with M3.

`Full` uses the same alpha-0.15 base as M3 and adds local direction
conditioning. Because it underperforms M3 on the frozen clean dev benchmark,
local direction conditioning is a negative ablation and is not part of the
primary proposed model.

## Frozen Matrix

| ID | CTC | SGM | Consistency | Script adapter | Local direction | Role |
| --- | --- | --- | ---: | --- | --- | --- |
| B0 | U2 visual | None | 0.00 | No | No | RCTC baseline |
| B1 | U2 visual | U2 visual | 0.00 | No | No | Full SVTRv2 baseline |
| B2 | Logical | Logical | 0.00 | No | No | Direction-conflict diagnostic |
| M1 | U2 visual | Logical | 0.00 | No | No | Dual-order semantic guidance |
| M2 | U2 visual | Logical | 0.30 | No | No | Official consistency ablation |
| M2-control | U2 visual | Logical | 0.15 | No | No | Matched control for M3 attribution |
| M3 | U2 visual | Logical | 0.15 | Yes | No | Primary proposed model |
| Full | U2 visual | Logical | 0.15 | Yes | Yes | Negative direction ablation |

## Selection And Test Policy

- Every checkpoint was selected only by clean target-domain dev macro CER.
- `P1_MSR_V3`, Protocol V2 data, the unified dictionary, and U2 conversion are
  inherited unchanged.
- Test data were not evaluated during method development.
- After this freeze, clean dev may be used for reporting and diagnosis only.
  It must not be used to alter architecture, loss weights, preprocessing,
  training data, or checkpoint selection rules.
- Any future architecture or hyperparameter change requires a new protocol ID
  and cannot replace V1 results silently.

## Interpretation Boundary

The primary method claim is supported by M3. M2 alpha 0.30 measures the best
frozen consistency-only setting, while M2-control alpha 0.15 provides a
one-factor comparison for the Script Adapter. Full is reported as a negative
ablation; it is not renamed or omitted because its result is unfavorable.

