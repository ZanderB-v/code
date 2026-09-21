# M3 alpha verification V1

This closes the only remaining consistency-weight question before phase 6.
Direction experiments are not rerun. D1, D2, and the old LDC remain diagnostic
results, while M3 / SOAR-SVTR remains the final model family.

The candidate grid is permanently limited to `0.15, 0.20, 0.25, 0.30`.
Only `0.20` and `0.25` receive new training. Each new point independently runs
the complete S50 synthetic stage followed by target-domain fine-tuning. All
four S50 stages begin from the same converted RCTC checkpoint; every target
stage begins from its matching S50 checkpoint.

All data, P1_MSR_V3 preprocessing, Protocol V2, dictionary, U2/logical labels,
seed, optimizer, scheduler, epoch horizon, early stopping, and external Clean
Dev evaluator are held fixed. Every saved target epoch is evaluated, and the
checkpoint with minimum Clean Dev Macro CER is selected. Corrupted Dev and Test
are not used.

The existing alpha=0.15 checkpoints are re-evaluated under this same all-epoch
rule. The alpha=0.30 all-epoch reports from `semantic_direction_v2` are reused
only after their checkpoint, config, evaluator, and report hashes pass. This
keeps selection granularity identical without retraining either existing model.

The module-attribution table keeps `alpha=0.30` because that comparison changes
only Script-Conditioned Adaptation from M2 to M3. The final SOAR-SVTR alpha is
selected independently from the four-point grid and then written to an
immutable decision artifact. No additional alpha values may be introduced.

## Manual synchronization

Synchronize these files to the same paths on the server:

- `scripts/svtrv2/dual_order_protocol.py`
- `scripts/svtrv2/run_dual_order_method_suite.py`
- `scripts/svtrv2/test_semantic_direction.py`
- `scripts/svtrv2/run_m3_alpha_verification.py`
- `scripts/svtrv2/run_m3_alpha_verification_tmux.sh`
- `scripts/svtrv2/test_m3_alpha_verification.py`
- `00_docs/M3_ALPHA_VERIFICATION_V1.md`

No synchronization archive is required or generated.

## Execution

Run the no-training preflight first:

```bash
bash scripts/svtrv2/run_m3_alpha_verification_tmux.sh --preflight-only
tmux attach -t m3_alpha_verification_v1
```

Require `M3_ALPHA_VERIFICATION_PREFLIGHT_OK` and exit code zero. Then run:

```bash
bash scripts/svtrv2/run_m3_alpha_verification_tmux.sh
tmux attach -t m3_alpha_verification_v1
```

The same command resumes an interrupted run. Never use `--replace`.

Completion requires `M3_ALPHA_VERIFICATION_COMPLETE` and exit code zero.
Outputs are written under
`04_model_training/eval_reports/m3_alpha_verification_v1/`.
