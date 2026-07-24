# Final V2 Architecture

## Architecture

```text
Shared ModernBERT + claim/fact evidence fusion
                │
        ┌───────┴────────┐
        │                │
 Exact-identity      Exact-identity
  RE adapter          TP adapter
        │                │
 Plaintiff/Defendant     │
      RE heads            │
        │                 │
  detached probabilities │
        └────────┬────────┘
                 │
   rationale pool + content fallback
      (fallback-dominant initialization)
                 │
 plaintiff/defendant interaction reasoner
                 │
       rationale residual logit
                 │
phase-1 global logit + sigmoid(scale) × residual
                 │
              verdict
```

The objective remains:

```text
L = 0.33 L_RE + 0.67 L_TP
```

No rationale statistics or extra auxiliary losses are added.

## Why this version

- RE probabilities are detached before TP pooling, so TP cannot rewrite the RE
  head into a latent gate that no longer matches rationale labels.
- RE and TP receive separate exact-identity adapters.
- Phase 2 preserves the phase-1 global verdict classifier as a frozen,
  deterministic anchor.
- The rationale branch starts as a zero correction, scaled by
  `sigmoid(-1.5) ≈ 0.18`.
- Rationale/fallback pooling also starts fallback-dominant.
- Phase-2 checkpoint selection penalizes an RE drop larger than the configured
  tolerance.

## Recommended command

Reuse the completed `joint_no_rationale` checkpoint:

```bash
python src/main.py \
  --config config/final_v2.yaml \
  --phase1-checkpoint outputs/ablations/joint_no_rationale/best_model.pt
```

Train both phases from scratch:

```bash
python src/main.py --config config/final_v2.yaml
```

## GPU configuration

The default profile targets a 16 GB GPU:

```yaml
training:
  batch_size: 1
  grad_accum_steps: 64

model:
  claim_chunk_size: 4
```

This preserves an effective batch size of 64 cases while reducing peak memory.

## Phase 2 schedule

By default:

1. Epochs 1–2: train only Stage 3 and the non-anchor parts of Stage 4.
2. Afterwards: unfreeze Stage-1 fusion and the final two encoder blocks.
3. Keep RE heads/adapters frozen.
4. Keep the global verdict classifier frozen.
5. RE loss continues to regularize the partially unfrozen shared Stage 1.

## Output

```text
outputs/final_v2/
├── phase1_global/
│   ├── best_model.pt
│   ├── resolved_config.yaml
│   └── training_log.json
└── phase2_final/
    ├── best_model.pt
    ├── phase1_load_report.json
    ├── resolved_config.yaml
    ├── training_log.json
    ├── results.json
    └── test_predictions.jsonl
```

`results.json` includes dev evaluation with predicted, gold, no-rationale, and
random rationale controls.

## Important interpretation

The final version should only be considered successful when:

- TP remains above the TP-only/global-only baseline;
- RE stays within the configured drop tolerance from phase 1;
- predicted rationale is consistently better than random rationale across
  multiple seeds.
