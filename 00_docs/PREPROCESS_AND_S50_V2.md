# Preprocessing Ablation and S50 Generation V2

## Gate 1: freeze the target benchmark

Run `freeze_target_benchmark_v2.py` on the complete server copy. Continue only
when `00_docs/frozen_target_benchmark_v2/verification_report.json` reports
`status: passed`.

This gate freezes target membership, source-image splits, labels, dictionary,
U2, metrics, and target image hashes. It does not freeze synthetic data or
model preprocessing.

## Gate 2: generate the preprocessing pilot

`run_preprocess_pilot_tmux.sh` generates exactly 5,000 diverse samples per
language at a retained final height of 64 pixels. The pilot is experimental
and is not part of S50.

## Gate 3: compare P0 and P1

`run_preprocess_ablation_tmux.sh` trains the same pretrained SVTRv2-S + RCTC
model on the same 15,000 pilot samples:

- P0: aspect-preserving `48x640` resize and right padding.
- P1: official `RatioDataSetTVResize` MSR shapes and dynamic long width.

Both are evaluated only on the frozen target dev split. The summary reports
macro CER, language CER, length buckets, source-aspect buckets including
`aspect>=12`, training throughput, inference timing, and GPU memory.

Target test is not used.

## Gate 4: generate S50

After P0/P1 is selected, `run_s50_generation_tmux.sh` generates five sequential
formal shards. Each shard contains exactly 10,000 samples per language.

The final layout is:

```text
03_synthetic_generation/synthetic_formal_v2/
  synthetic_shard_0000_parallel/
  synthetic_shard_0001_parallel/
  synthetic_shard_0002_parallel/
  synthetic_shard_0003_parallel/
  synthetic_shard_0004_parallel/
  subsets/
    s10/
    s25/
    s50/
```

The subset builder enforces `S10 subset S25 subset S50`, dictionary coverage,
strict UG U2 labels, Kazakh-specific character coverage, no duplicate image
bytes across shards, and at most two occurrences of one logical string.
