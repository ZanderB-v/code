# Method Robustness and Multi-Seed Protocol V1

## Frozen scope

- Models: B0, B1, B2, M1, M2 alpha=0.30, M3, and Full.
- Checkpoints and configurations come from Method Design Freeze V1.
- Corrupted Dev uses the immutable Corruption Protocol V1.
- Test inference remains prohibited during model development.

## Corrupted Dev

All seven models use the same manifest, image variants, P1/MSR inference path,
character dictionary, U2 conversion, and metric implementation. The immutable
corruption transformation protocol remains V1. A separate Method Evaluation
Binding V1 freezes the current P1_MSR_V3 inference code and method registry;
this avoids silently reusing reports produced by the obsolete evaluator hash.
Protocol V2 verification must pass immediately before the binding is created.
Report clean
macro CER, mean corrupted macro CER, absolute and relative CER increase,
per-corruption CER, and per-language CER.

## Multi-seed scope

The synthetic pretraining checkpoint is fixed for every method. Only the target
domain finetuning stage is repeated. This is the minimum-cost protocol and must
not be described as three full end-to-end training seeds.

- Existing frozen seed: 20260731.
- New finetuning seeds: 20260811 and 20260812.
- Models repeated: B1, M1, M2 alpha=0.30, and M3.
- Selection metric: Clean Dev Macro CER.
- Training control: maximum 50 epochs, evaluation every 2 epochs, patience 5
  evaluations, minimum 10 epochs, and minimum delta 1e-4.

Only `Global.seed`, `Global.output_dir`, `Global.save_res_path`, and
`Global.project_name` may differ from the frozen canonical target-finetuning
configuration. The seed controls model RNG, augmentation/worker RNG, and the
LMDB traversal sampled before loader construction. The ratio sampler keeps an
epoch-deterministic batch shuffle so interrupted/resumed training is equivalent;
this common ordering rule is intentionally held constant across seeds.

Report sample mean and sample standard deviation across the three finetuning
seeds. Compare B1 and M3 using a paired, language-stratified bootstrap over the
same Dev lines. The primary bootstrap statistic is the seed-averaged difference
`M3 macro CER - B1 macro CER`; negative values favor M3.
