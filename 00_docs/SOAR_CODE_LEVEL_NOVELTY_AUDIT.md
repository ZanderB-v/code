# SOAR-SVTR Code-Level Novelty Audit

Scope: frozen S50, Clean Dev V4, M3 alpha=0.15. This is an attribution and
claim-boundary audit, not a new experiment. No Test data were used.

## 1. Baseline ownership

The checked-in OpenOCR SVTRv2 config already contains
`SVTRv2LNConvTwo33` (encoder), `GTCDecoder`, `SMTRDecoder` (semantic branch),
`RCTCDecoder` (visual CTC branch), `GTCLoss`, and the existing resize/data
pipeline. See `third_party/OpenOCR/configs/rec/svtrv2/svtrv2_smtr_gtc_rctc.yml`
and the original classes in `openrec/modeling/decoders/__init__.py` and
`openrec/losses/__init__.py`. The SVTRv2 paper owns MSR, FRM, SGM, and the
training-time CTC+SGM recipe. B1 is a task adaptation of this baseline; the
4891-character vocabulary and frozen U2 labels are not new model modules.
`git diff` shows no changes to the checked-in original SVTRv2 encoder or
official config. The three new method implementation files below are currently
untracked, while the framework's preprocess/decoder/loss registries each have
one added mapping. Add the new files to version control before release.

## 2. Actual implementation delta

| Area | Original | Added here | Evidence |
| --- | --- | --- | --- |
| Target representation | One label sequence is passed to CTC and SGM | Versioned JSON payload with logical and visual strings, exact bidi index map, script and direction metadata; invisible format controls normalized out | `scripts/svtrv2/dual_order_protocol.py:262,284,329` |
| Label encoding | Standard CTC/SMTR encoders | CTC takes visual/U2 text; SGM takes Unicode logical text; exact permutation and consistency mask are checked | `third_party/OpenOCR/openrec/preprocess/dual_order_gtc_label_encode.py:30,134` |
| Decoder | Shared encoder -> GTCDecoder -> CTC + SMTR | DualOrderGTCDecoder retains original RCTC/SMTR subdecoders, optionally adds script residual adaptation before both | `third_party/OpenOCR/openrec/modeling/decoders/dual_order_gtc_decoder.py:271,385` |
| COC | Weighted CTC + SMTR loss only | Construct soft target-conditioned CTC token/frame alignment, transport token posteriors to logical positions, then confidence-weighted JS against SGM character distributions | `third_party/OpenOCR/openrec/losses/dual_order_gtc_loss.py:79,106,134` |
| Script adapter | None | Learned column-level script posterior softly mixes four zero-output-initialized bottleneck/depthwise-convolution residual experts (Han, Arabic, Cyrillic, common) | `third_party/OpenOCR/openrec/modeling/decoders/dual_order_gtc_decoder.py:94,119,141` |
| Script supervision | None | Unicode-derived token script IDs are softly projected to visual frames with CTC-based alignment; auxiliary loss weight 0.1 | `third_party/OpenOCR/openrec/preprocess/dual_order_gtc_label_encode.py:107,175`; `third_party/OpenOCR/openrec/losses/dual_order_gtc_loss.py:257` |
| Framework registration | Existing GTC/SMTR mappings | Three one-line mappings for the new encoder, decoder and loss; no replacement of original implementations | `third_party/OpenOCR/openrec/preprocess/__init__.py:101`; `third_party/OpenOCR/openrec/modeling/decoders/__init__.py:24`; `third_party/OpenOCR/openrec/losses/__init__.py:26` |

The frozen M3 target config sets `ctc_order: visual`, `sgm_order: logical`,
`consistency_weight: 0.15`, `script_weight: 0.1`, and
`use_script_adapter: True` in
`04_model_training/configs/svtrv2_s_m3_dual_order_s50_to_target.yml`.

The accurate COC description is **not** `JS(raw CTC frames, permuted SGM)`.
CTC has many frames and SGM has character positions. The code first computes
target-aware soft frame alignment, obtains visual token posteriors, indexes
them using the logical-to-visual map, then compares aligned distributions.
Confidence weighting is detached. Mirrored symbols that change codepoints
under bidi are excluded by `consistency_mask`; the exact permutation claim is
limited to normalized retained characters. The algorithm does not merely
reverse an RTL string.

The script gate is inferred from image features at inference. It does not use
the GT language ID or select a separate recognition head. All four experts
are evaluated before soft mixing; the adapter therefore has nonzero compute
and parameter cost. SGM, COC, and the auxiliary script loss can be omitted
from CTC-only inference; the script adapter cannot.
The evaluation script explicitly sets `decoder.infer_gtc = False` for the CTC
branch (`scripts/svtrv2/infer_label_file_metrics.py:201-207`). The training
config itself has `infer_gtc: True`, so a speed claim requires a benchmark of
the actual CTC-only deployment path.

## 3. Related-work boundary

- SVTRv2: CTC+SGM, MSR, FRM and SGM removal at inference are prior work.
  https://openaccess.thecvf.com/content/ICCV2025/papers/Du_SVTRv2_CTC_Beats_Encoder-Decoder_Models_in_Scene_Text_Recognition_ICCV_2025_paper.pdf
- GTC (2020): a guidance branch for CTC is prior work; do not claim to invent
  guided CTC training. https://arxiv.org/abs/2002.01276
- PARSeq (ECCV 2022) and CGDS (2025): permutation or multi-decoder semantic
  modeling is prior work. Their published objectives are not, on the checked
  descriptions, an explicit visual/U2-CTC versus Unicode-logical-SGM target
  split with bidi-transported posterior consistency.
  https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880177.pdf
  https://link.springer.com/article/10.1007/s40747-024-01689-5
- Visual-Semantic Dual-Decoder Collaboration (PRCV 2025, online 2026):
  dual-decoder KL-guided collaboration exists. Do not claim the first
  inter-decoder consistency loss. Its published abstract does not identify
  an explicit Unicode logical/visual order transport.
  https://link.springer.com/chapter/10.1007/978-981-95-5676-2_42
- DFETransformer (2026): CTC plus Transformer/attention optimization for
  Uyghur is also prior work, but its publisher abstract does not describe a
  visual-order/logical-order target split with bidi index transport.
  https://www.sciencedirect.com/science/article/pii/S095219762601033X
- SARN (2024): local/global script-aware character features already exist.
  The adapter's use of script information or soft script mixing alone is not
  novel. https://www.sciencedirect.com/science/article/pii/S0957417424006195
- Multiplexed Multilingual OCR (CVPR 2021): script identification plus
  separate recognition heads already exists. SOAR instead keeps a shared
  CTC/SMTR recognizer with softly mixed residual experts.
  https://openaccess.thecvf.com/content/CVPR2021/html/Huang_A_Multiplexed_Network_for_End-to-End_Multilingual_OCR_CVPR_2021_paper.html
- Unicode UAX #9 defines logical-to-visual bidi ordering. Using python-bidi
  correctly is protocol engineering, not invention of the bidi algorithm.
  https://www.unicode.org/reports/tr9/

Literature conclusion: in these representative sources, no exact same
combination was identified. This is a scoped observation, not a priority or
patent novelty assertion. A broader systematic literature review may change
it.

## 4. Evidence and attribution limits

Clean Dev V4 CER-selected results: B1 1.7092%, M1 1.6428%, M2 alpha=.30
1.5888%, M3 alpha=.30 1.5332%, final M3 alpha=.15 1.5059%. Thus B1->M1
supports the dual-order target change, M1->M2 supports COC, and the matched
alpha=.30 M2->M3 comparison supports a 0.0556 percentage-point CER gain from
adding the script adapter and its auxiliary loss as a **bundle**. It does
not isolate the residual experts from script supervision. M3 alpha=.15 versus
M2 alpha=.30 additionally changes alpha; do not attribute that entire gap
to the adapter. The matched-alpha line accuracy slightly decreases from
91.2455% to 91.2065%, so do not claim every metric improves with the adapter.

The quoted M3-noCOC result in prior notes is not present in the currently
synced local result directory. Do not use its numerical delta as verified
evidence until the formal summary and checkpoint provenance are available.
The V4 main table is single-seed selected-checkpoint evidence; statistical
reliability requires consistent multi-seed evaluation on the same frozen V4.

## 5. Paper wording

Use: "We adapt SVTRv2's training-time semantic guidance to mixed writing
orders by supervising CTC in visual/U2 order and SGM in Unicode logical
order. A normalized bidi index map aligns soft CTC token posteriors with SGM
character distributions for confidence-weighted cross-order consistency."

Use: "We add an image-inferred, column-wise script-conditioned residual
adapter before the shared recognition branches, with auxiliary script
supervision. This component complements order-aware guidance in our
multiscript setting."

Avoid: "We propose CTC+SGM", "first bidirectional/RTL OCR", "first
script-aware recognition", "first decoder consistency", "JS directly
between CTC frames and SGM outputs", "GT-script routing at inference",
"zero inference overhead for the entire M3", "all tokens have exact bidi
posterior alignment", and "script adaptation alone caused the full M2->M3
alpha=.15 improvement".

## 6. Highest-value follow-up audits (no new module)

1. Report fraction of characters used by COC after format-control removal
   and mirrored-symbol masking, by language and mixed-script subset.
2. Add independent toggles for residual experts versus script auxiliary
   supervision if claiming an adapter mechanism rather than bundle effect.
3. Verify and archive M3-noCOC's formal S50->target provenance; use the same
   V4 selection rule and alpha=.15.
4. Measure parameter count, FLOPs, and latency for B1/M1/M2/M3 through the
   actual CTC-only inference path.
5. Re-evaluate the frozen seed set on the same V4 protocol before writing
   significance language. Keep Test closed until final model freeze.
