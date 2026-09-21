# Multi-Script Dual-Order Semantic Guidance V1

## Scope

This protocol defines the proposed method for Chinese, Uyghur, and
Cyrillic Kazakh meme-line recognition. It addresses a specific conflict:

- CTC requires a sequence that is monotonic in image scan order.
- Unicode language modeling requires logical reading order.
- In Uyghur lines, the two orders can differ locally because Arabic,
  Latin, digits, punctuation, and mirrored symbols may form mixed bidi
  runs.

The implementation does not reverse a complete sentence or a complete
feature map. It preserves local Unicode bidi runs.

Protocol identifiers:

- Preprocessing: `P1_MSR_V3`
- Dual-order labels: `P1_MSR_V3_DUAL_ORDER_V2`
- Method: `MULTISCRIPT_DUAL_ORDER_SGM_V1`

## Notation

For a logical Unicode label of length `L`,

`y_log = [y_log_1, ..., y_log_L]`,

the Unicode bidi algorithm provides a visual-order label

`y_vis = [y_vis_1, ..., y_vis_L]`

and a bijection `pi`, where `pi(i)` is the visual position of logical
position `i`. Chinese and Kazakh use the identity permutation. Uyghur
uses the exact permutation returned by `python-bidi` with RTL paragraph
direction.

### OCR label normalization

The immutable source metadata is preserved verbatim. Before constructing the
dual-order supervision, non-glyph Unicode format controls (`U+200B`, `U+200C`,
`U+200D`, `U+2060`, and `U+FEFF`) are removed from both logical and visual OCR
labels. These controls can affect shaping but are not independently observable
OCR characters, and `python-bidi` removes them while resolving visual order.
Any other Unicode `Cf` character is rejected instead of silently normalized.
The LMDB manifest records the removed code points and occurrence counts. This
keeps the logical-to-visual transport bijective without modifying the frozen
dataset or the 4891-class checkpoint-compatible dictionary.

The dataset stores both labels and `pi`. It never reconstructs `pi` by
matching repeated characters.

## 1. Dual-Order Semantic Guidance

The RCTC branch is supervised with `y_vis`:

`L_ctc = CTC(P_ctc, y_vis)`.

The SGM branch is supervised with `y_log`:

`L_sgm = CE(P_sgm, y_log)`.

This division gives each objective the order that matches its role:

- RCTC retains monotonic visual alignment.
- SGM models Unicode logical context.

`B1` is not the proposed dual-order method because both of its branches
use `y_vis`. `M1` is the first dual-order model.

## 2. Permutation-Transported Posterior Consistency

Directly comparing CTC frames and SGM tokens is invalid because they
have different lengths and orders. The implementation therefore uses a
target-conditioned soft monotonic transport.

For visual token `j` and CTC frame `t`, the transport score is:

`a_jt = log P_ctc(t, y_vis_j) + G(j, t)`,

where `G` is a Gaussian monotonic prior centered at the expected visual
position of token `j`. Softmax over frames gives a token-to-frame
transport matrix `A`.

`P_ctc(t, y_vis_j)` is the absolute CTC posterior before character-only
renormalization, so a high-blank frame cannot become artificial character
evidence merely because the blank class was removed.

The visual token posterior is:

`Q_vis = A P_ctc`.

The exact bidi permutation transports it into logical order:

`Q_log_i = Q_vis_pi(i)`.

The consistency objective is symmetric Jensen-Shannon divergence:

`L_xorder = weighted_mean_i JS(Q_log_i, P_sgm_i)`.

The detached reliability weight is the geometric mean of the two
normalized inverse entropies. It is near zero while the randomly
initialized SGM posterior is uniform, then increases as both branches
become informative. This prevents symmetric consistency from pulling a
pretrained CTC branch toward an untrained semantic branch at the start
of optimization without introducing a dataset-specific epoch warmup.

The comparison is restricted to the common real-character vocabulary.
CTC blank and SGM EOS/BOS classes are excluded. Positions changed by
Unicode mirroring are masked because their visual and logical code
points are intentionally different.

This objective transfers uncertainty distributions rather than only
hard argmax labels, while preserving mixed-direction local runs.

## 3. Script-Conditioned Visual Adaptation

The model predicts a four-way soft script field for every visual column:

`g_t = softmax(h_script(F_t))`.

Four lightweight residual experts specialize in Han, Arabic, Cyrillic,
and common/Latin visual statistics. Their outputs are mixed locally:

`F_script_t = F_t + alpha * sum_s g_ts E_s(F)_t`.

Frame-level soft script targets are transported from visual-order
character labels through the same CTC alignment used by the consistency
term. Column losses are weighted by detached CTC non-blank probability,
so background and padding columns are not forced to carry a script label.
The gate is supervised during training, but inference uses only
the image. No language ID or ground-truth text is supplied at test time.
The common/Latin state allows local switching inside Uyghur lines with
digits, Latin abbreviations, or punctuation.

The experts are zero-initialized at their final projection, so this
component begins as an identity mapping and does not destabilize a
pretrained encoder.

## 4. Local Direction Conditioning

The full model predicts an LTR/RTL distribution for every visual
column. Frame-level soft direction targets are obtained from the same
target-conditioned visual-token alignment used by the consistency
term and are likewise weighted by CTC non-blank probability. The
resulting direction field modulates RCTC features with a
small learned scale and bias.

This component represents local bidi runs. It is not a global Uyghur
feature reversal.

## Objective

`L = 0.1 L_ctc + L_sgm + lambda_x L_xorder`

`    + lambda_s L_script + lambda_d L_direction`.

V1 weights:

- `lambda_x = 0.15`
- `lambda_s = 0.10`
- `lambda_d = 0.05`

These values are fixed before method training. Any later tuning must
use clean target-domain dev only and must be reported as a separate
protocol revision.

## Controlled Experiment Matrix

| ID | CTC order | SGM order | Cross-order | Script adapter | Local direction |
| --- | --- | --- | --- | --- | --- |
| B0 | Visual U2 | None | No | No | No |
| B1 | Visual U2 | Visual U2 | No | No | No |
| B2 | Logical | Logical | No | No | No |
| M1 | Visual U2 | Logical | No | No | No |
| M2 | Visual U2 | Logical | Yes | No | No |
| M3 | Visual U2 | Logical | Yes | Yes | No |
| Full | Visual U2 | Logical | Yes | Yes | Yes |

For `B2` through `Full`, the following are controlled:

- frozen S50 images and texts;
- frozen target train/dev;
- P1/MSR preprocessing;
- SVTRv2-S encoder, RCTC decoder, and SMTR decoder;
- dictionary and maximum text length;
- starting RCTC checkpoint;
- optimizer, learning rate, B1 seed (`20260731`), batch size, and
  early-stop policy;
- clean target dev macro CER for checkpoint selection;
- no test evaluation during model development.

## Hypotheses and Falsification

- `M1 < B1` in Uyghur CER supports logical-order semantic supervision.
- `M2 < M1` supports posterior transport rather than extra supervision
  alone.
- `M3 < M2` with gains in all or most scripts supports shared
  script-aware adaptation.
- `Full < M3`, especially on mixed Uyghur runs, supports local direction
  conditioning.

Here `<` means lower clean-dev CER under the same checkpoint-selection
protocol. A component is not retained merely because training loss
decreases. It must improve the predefined dev metrics and should not
cause a material regression in another language.

## Reporting

Report:

- CER, WER, 1-NED, and line accuracy per language;
- macro CER;
- short, medium, long, and ultra-wide buckets;
- pure Uyghur versus Uyghur with digits or Latin runs;
- Kazakh-specific character recall;
- parameters, FLOPs, FPS, and peak memory;
- clean and frozen-corruption dev results.

The final selected methods are evaluated once on frozen clean and
corrupted test data.

## Novelty Boundary

This is an independently specified project method, not a claim that
every individual operation is unprecedented. The paper contribution
must be stated as the complete problem formulation and controlled
mechanism:

1. separate visual and Unicode logical supervision for mixed-direction
   CTC recognition;
2. exact bidi-permutation posterior transport between CTC and SGM;
3. image-inferred script routing and local direction conditioning.

A formal related-work search is still required before claiming novelty
relative to all published or concurrent work.
