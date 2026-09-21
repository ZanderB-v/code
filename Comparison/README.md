# Public pretrained baseline comparison

This directory implements the fail-closed preflight for CRNN, SVTR-Base,
PARSeq and ABINet before any formal comparison training starts.

## Scientific contract

- The same frozen 4,891-symbol dictionary and S50 records are used.
- Uyghur uses `ctc_text_u2`; Chinese and Kazakh use `ctc_text`.
- Public checkpoints are loaded only into architecture-compatible parameters.
- Vocabulary-dependent and length-dependent modules are reset by an explicit
  model-specific whitelist.
- A loading audit fails on any unexpected key, non-whitelisted shape mismatch,
  missing preserved parameter or insufficient backbone coverage.
- A real three-language S50 batch must complete forward, finite loss, backward,
  and non-zero finite gradients in both the retained backbone and reset modules.
- PARSeq uses the released six-permutation mirrored training objective; it is
  not reduced to ordinary left-to-right teacher forcing.
- The same source samples are used, then each architecture applies its released
  native height and normalization. CRNN widens to 32x512 for line capacity;
  PARSeq/ABINet remain 32x128; MMOCR SVTR remains 64x256 before its STN.
  P1/MSR is not injected into these external architectures.
- Frozen corruptions must be applied to the source image before the model-native
  resize. This keeps corruption content identical without changing architectures.
- These scripts never read or evaluate Test.

The dictionary file has 4,891 entries. The frozen OpenOCR-compatible runtime
also has `use_space_char: true`, so every model recognizes 4,892 shared symbols
including space. CTC then adds one blank; autoregressive models add their
required EOS/BOS/PAD tokens. These special tokens do not alter the dictionary
file.

## Server commands

```bash
cd /data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition

bash Comparison/scripts/install_pretrained_baseline_runtime.sh

bash Comparison/scripts/run_pretrained_baselines_preflight_tmux.sh --replace
tmux attach -t pretrained_baselines_v1_preflight
```

Formal training is allowed only when the final line is:

```text
ALL_PRETRAINED_BASELINES_V1_PREFLIGHTS_OK
```

Reports and adapted initializations are written to
`Comparison/pretrained_baselines_v1/outputs/`.

Sync the entire `Comparison/` directory before running the commands. It now
contains the four checkpoints, exact tagged source snapshots, source hashes and
the Linux offline wheels required by the server environment.

## Fairness disclosure

This is a controlled downstream comparison, not a pure architecture-only
comparison. Initialization sources differ: MMOCR SVTR-Base uses MJ+ST, while
the PARSeq model-hub checkpoints follow their own public recipes. The paper
must disclose each source and must not describe the initializations as equal.
Likewise, native input preprocessing is disclosed and the resulting table is a
controlled downstream comparison, not a claim that only architecture differs.

## Formal line-recognition gate

The released word-recognition defaults are too short for the frozen line
labels. The formal protocol uses a 128-character capacity without changing any
pretrained backbone tensors:

- CRNN uses `32x512`, producing 129 CTC steps.
- MMOCR SVTR changes only its parameter-free final adaptive pooling length from
  40 to 128.
- PARSeq and ABINet explicitly reset vocabulary- and length-dependent
  positional modules to 128 characters.
- Silent truncation and hiding impossible CTC alignments with
  `zero_infinity=True` are forbidden.
- The MMOCR SVTR STN/TPS rectification block runs in FP32 inside the outer
  AMP context. This is a numerical-stability guard for TPS/grid sampling, not
  an architecture change; the SVTR encoder and replacement CTC head still use
  the common AMP training path.
- The CRNN recurrent CTC path runs in FP32 because its FP16 backward becomes
  non-finite with the expanded 4891-class head on long labels. The other
  baselines retain AMP, and CTC log-probabilities are always computed in
  FP32. This is a numerical precision guard, not an architecture change.

After syncing code, rerun the enhanced gate. The earlier short-batch result is
not authorization for formal training:

```bash
bash Comparison/scripts/run_pretrained_baselines_preflight_tmux.sh --replace
tmux attach -t pretrained_baselines_v1_preflight
```

The JSON summary must report `formal_protocol_audit: passed`, a passed
`formal_ddp_preflight` for every model, `formal_training_allowed: true`, and no
Test access. The gate also binds the current S50 manifest, target metadata,
dictionary, frozen corruption protocol, source provenance, public weights, and
training/evaluation code by SHA-256. Then run:

```bash
bash Comparison/scripts/run_pretrained_baselines_formal_tmux.sh --replace
tmux attach -t pretrained_baselines_v1_formal
```

Use `--replace` only for a new formal run. If the server interrupts after at
least one `latest.pth` was written, restart without `--replace`; the chain will
resume incomplete stages and skip completed stages:

```bash
bash Comparison/scripts/run_pretrained_baselines_formal_tmux.sh
tmux attach -t pretrained_baselines_v1_formal
```

The formal chain is public initialization -> S50 adaptation -> target-domain
fine-tuning -> Clean Dev checkpoint selection -> frozen Corrupted Dev
diagnostics. Test remains blocked.
