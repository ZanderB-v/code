# Clean Dev V2 Frozen Protocol

## Status

- Protocol: `CLEAN_DEV_V2_VERIFIED`
- Rows: 951 (`zh=346`, `ug=295`, `kk=310`)
- Model-blind review coverage: 951/951
- GT patches: 15
- Ambiguous or crop rows excluded: no
- Original Clean Dev modified: no
- Corrupted Dev used for selection: no
- Test evaluated: no

The model-selected error review remains diagnostic evidence only. It is not a
source of final Dev labels because it conditions annotation discovery on M3
errors. The final patch set comes only from the full 951-row model-blind audit.

## Normalization V2

The metric applies the same normalization to GT and prediction:

1. Reject zero-width and explicit bidi control characters.
2. Apply Unicode NFC.
3. Map `U+FF5E FULLWIDTH TILDE` to ASCII `~`.
4. Canonicalize Unicode whitespace to one ASCII space and trim outer spaces.
5. Preserve internal spaces and all other punctuation.

Global NFKC, punctuation removal, and Han-space removal are forbidden.

## Checkpoint Reselection

Existing per-sample predictions are immutable and are rescored against Clean
Dev V2. No inference or training is rerun. For every model or hyperparameter
arm, checkpoint selection remains:

```text
argmin_epoch Clean Dev V2 Macro CER
```

Macro CER is the arithmetic mean of the `zh`, `ug`, and `kk` CER values. Ties
are resolved by the earliest epoch. WER, 1-NED, and Line Accuracy are reported
from that same selected epoch but do not participate in selection.

Run on the server:

```bash
cd /path/to/svtrv2_line_recognition
bash scripts/svtrv2/run_clean_dev_v2_reselection_tmux.sh --replace
tmux attach -t clean_dev_v2_reselection
```

The preflight must find every saved epoch prediction report for B1, M1, M2,
M3, the fixed M3 alpha grid, and the controlled S25/S50 comparison. A missing
arm or epoch stops the run instead of producing a partial scientific table.

No M3-noCOC, SLDR, or other new training is allowed until the V2 reselection
manifest has been inspected and the final S50/M3/alpha decisions are confirmed.
