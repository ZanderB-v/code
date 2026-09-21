# P1/MSR Formal RCTC Baseline

## Scope

Protocol V2 remains the frozen data and metric protocol. It freezes the repaired
target-domain train/dev/test split, nested S10/S25/S50 synthetic manifests,
character dictionary, Uyghur U2 conversion, metrics, and data hashes.

`P1_MSR_V3` is the selected input and RCTC training protocol for the new formal
experiments. Old P0 fixed `48 x 640` results are not part of the formal result
chain.

## Frozen RCTC Input Protocol

- dataset: `RatioDataSetTVResize`
- sampler: `RatioSampler`
- base shapes: `[[64,64], [96,48], [112,40], [128,32]]`
- base height: `32`
- sampler scale: `[[128,32]]`
- divided factor: `[4,16]`
- dynamic width: enabled
- padding: disabled
- maximum width-height ratio: `40`
- first batch size per card: `32`
- maximum label length: `120`
- external Clean Dev inference batch size: `1` (per-sample dynamic width)

External model selection never batches different aspect ratios together. This
prevents batch-neighbor padding from changing a sample's effective width and
keeps Clean Dev identical to the frozen corruption evaluator.

The label limit is 120 because the frozen target benchmark contains lines longer
than 60 characters and its maximum observed length is 101. This change prevents
valid long target lines from being silently rejected by `CTCLabelEncode`; it
does not alter any image or label.

All formal preparation scripts use `scripts/svtrv2/p1_msr_protocol.py`.
`scripts/svtrv2/validate_p1_msr_config.py` rejects:

- fixed `RecTVResize`;
- `48 x 640` image shapes;
- padded P0 input;
- a missing or stale LMDB manifest;
- the wrong initialization source;
- a nonnumeric optimizer learning rate;
- a config containing a test reference.

## RCTC Baseline Chain

| ID | Initialization | Training data | Selection |
| --- | --- | --- | --- |
| E0 | random | target-domain train | clean dev macro CER |
| E1 | Union14M SVTRv2-S | target-domain train | clean dev macro CER |
| D2 | Union14M SVTRv2-S | S50 synthetic | clean target dev macro CER |
| E5 | D2 best checkpoint | target-domain train | clean dev macro CER |

The shared architecture is:

`SVTRv2LNConvTwo33 -> RCTCDecoder -> CTCLoss`

The class dictionary, U2 labels, optimizer, learning rate, input protocol, and
checkpoint selection rule are held constant. Test is not run during model
development.

External evaluation and dual-GPU DDP never run concurrently. At each evaluation
boundary, the stage runner:

1. validates a complete checkpoint containing model, optimizer, scheduler,
   epoch, and global step;
2. stops DDP and releases both GPUs;
3. evaluates clean target dev macro CER;
4. resumes from the same checkpoint if early stopping has not triggered.

This preserves the OneCycleLR and optimizer states while avoiding GPU contention
and the DDP/internal-evaluation stall seen in earlier runs.

The OpenOCR checkpoint loader filters pretrained tensors by both name and shape.
The 95-class Union14M output tensors are skipped explicitly when the 4,891-class
multilingual head is built, while compatible encoder and RCTC tensors are
loaded. The formal validator also checks the fixed Union14M SHA-256 before
training. Resumed `RatioSampler` instances receive the actual epoch number so
their shuffle order does not restart at epoch zero after each external
evaluation boundary.

## Scale Ablation

The scale study uses nested data and the same pipeline:

`S10/S25/S50 pretraining -> target-domain fine-tuning`

S50 reuses D2/E5 only when their final summaries declare
`preprocess_protocol = P1_MSR_V3`. It is otherwise rejected or rerun explicitly.

## Commands

Formal baseline chain:

```bash
cd /data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition
bash scripts/svtrv2/run_rctc_baseline_chain_tmux.sh \
  --replace \
  --rebuild-msr-lmdb \
  --preflight-only
```

This first pass builds the V3 LMDB caches and performs every CPU/GPU smoke test
without starting formal optimization. Start training only after it reports
`RCTC_BASELINE_PREFLIGHT_ONLY_OK`:

```bash
bash scripts/svtrv2/run_rctc_baseline_chain_tmux.sh --replace
```

The launcher also runs a synchronous environment preflight. A missing
dependency, GPU, frozen input, or mismatched Union14M checkpoint is reported
before tmux is created. Rebuild LMDB again only after a verified data or
protocol change:

```bash
bash scripts/svtrv2/run_rctc_baseline_chain_tmux.sh \
  --replace \
  --rebuild-msr-lmdb
```

Scale ablation after the baseline chain:

```bash
bash scripts/svtrv2/run_rctc_scale_ablation_tmux.sh \
  --replace \
  --preflight-only
bash scripts/svtrv2/run_rctc_scale_ablation_tmux.sh --replace
```

The first command validates S10/S25/S50 synthetic configs, real LMDB batches,
forward/loss/backward, and complete two-rank sampler schedules without training.
Target fine-tune configs are checkpoint-dependent and are validated immediately
after their corresponding synthetic pretrain stage.

Full SVTRv2 B1 after scale selection:

```bash
bash scripts/svtrv2/run_b1_full_svtrv2_tmux.sh \
  --replace \
  --preflight-only
bash scripts/svtrv2/run_b1_full_svtrv2_tmux.sh --replace
```

The B1 preflight rejects stale D2 checkpoints by checking the P1/MSR protocol,
Protocol V2 fingerprint, external-evaluation batch size, config hash, and
checkpoint hash. Legacy one-off launchers are disabled for formal experiments.

## Next Gates

Do not implement the proposed method before these gates pass:

1. E0/E1/D2/E5 configs and summaries all declare `P1_MSR_V3`.
2. E0/E1 differ only in initialization.
3. E1/E5 differ only in initialization checkpoint.
4. S10/S25/S50 use nested manifests and identical training settings.
5. Clean dev macro CER is the only checkpoint selection signal.

After the scale is selected, implement the full official SVTRv2-S GTC/SGM
baseline. The method sequence remains:

| ID | CTC order | SGM order | Additional component |
| --- | --- | --- | --- |
| B0 | U2 visual | none | RCTC baseline |
| B1 | U2 visual | U2 visual | official full SVTRv2 |
| B2 | logical | logical | direction-conflict diagnostic |
| M1 | U2 visual | logical | dual-order semantic guidance |
| M2 | U2 visual | logical | cross-order consistency |
| M3 | U2 visual | logical | script-conditioned adaptation |
| Full | U2 visual | logical | M3 plus local bidi-run conditioning |

B1 and the proposed method require separate GTC/SGM data encoders and losses.
They must not be approximated by renaming the RCTC configuration.
