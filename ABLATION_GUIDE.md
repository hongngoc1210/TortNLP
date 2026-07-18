# RE–TP Ablation Guide

## 1. Baselines

```bash
python src/majority_baseline.py --config config/config.yaml
python src/ablation_main.py --experiment tp_only
python src/ablation_main.py --experiment re_only
python src/ablation_main.py --experiment joint_no_rationale
python src/ablation_main.py --experiment joint_predicted
```

Interpretation:

- `tp_only > joint_no_rationale`: RE gradients are harming the shared encoder.
- `joint_no_rationale > joint_predicted`: the rationale pooling path is noisy.
- `re_only > joint_predicted` on RE F1: TP causes negative transfer to RE.

## 2. Oracle and faithfulness controls

Every rationale-based run automatically evaluates its best checkpoint with:

- `predicted`
- `gold`
- `no_rationale`
- `random`

The results are written to:

```text
outputs/ablations/<experiment>/results.json
```

You can evaluate an existing checkpoint without retraining:

```bash
python src/evaluate_checkpoint_ablation.py \
  --checkpoint outputs/run_01/best_model.pt \
  --split dev
```

Interpretation:

- `gold >> predicted`: RE is the bottleneck.
- `no_rationale > predicted`: predicted rationales hurt TP.
- `random ≈ predicted`: TP is not using the rationale scores faithfully.
- `gold ≈ no_rationale`: the TP fusion/head cannot exploit rationale.

## 3. Coupling and specialization

```bash
python src/ablation_main.py --experiment detach_rationale
python src/ablation_main.py --experiment task_adapters
python src/ablation_main.py --experiment adapters_global_residual
```

- `detach_rationale` prevents TP loss from directly changing RE logits.
- `task_adapters` adds a small RE adapter and TP adapter.
- `adapters_global_residual` starts TP close to the global representation and
  lets the rationale path enter through a learned conservative scale.

## 4. Gradient conflict

First measure it:

```bash
python src/ablation_main.py --experiment gradient_diagnostics
```

Check `grad_cosine` in `training_log.json`:

- often negative: gradient conflict is plausible;
- near zero: tasks are mostly independent;
- one gradient norm much larger: loss/gradient imbalance.

Then test PCGrad:

```bash
python src/ablation_main.py --experiment pcgrad
```

The reference PCGrad implementation uses `grad_accum_steps=1` and disables AMP.

## 5. Teacher forcing

Compare:

```bash
python src/ablation_main.py --experiment joint_predicted,joint_teacher_forcing,joint_gold_train
```

`joint_gold_train` is diagnostic only. It creates a train–inference gap and is
not a recommended final system.

## 6. Important corrections included

1. The cosine scheduler now counts optimizer updates rather than raw batches.
   With 326 batches and gradient accumulation 4, one epoch has about 82
   scheduler steps, not 326.
2. Plaintiff and defendant RE losses are averaged by default. Previously they
   were summed, so `0.33 * L_RE` could have roughly twice the intended scale.
3. Best-checkpoint selection is configurable: `mean`, `tp`, or `re`.
4. RE and TP losses, per-side RE F1, gate means, gate saturation, gradient norms,
   and gradient cosine are logged separately.

## Recommended run order

```text
majority
→ tp_only
→ re_only
→ joint_no_rationale
→ joint_predicted
→ oracle controls from results.json
→ task_adapters
→ adapters_global_residual
→ gradient diagnostics
→ PCGrad only when diagnostics support it
```
