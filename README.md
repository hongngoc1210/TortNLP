# UIT_TortNLP — COLIEE 2026 Pilot Task

Joint Rationale Extraction and Tort Determination for Japanese civil law cases.

---

## Project structure

```
.
├── config/
│   └── config.yaml            # All hyperparameters
├── src/
│   ├── main.py                # Single-GPU training & evaluation
│   ├── main_2x.py             # 2-GPU DDP training
│   ├── inference.py           # Standalone inference on test set
│   ├── models/
│   │   ├── shared_encoder.py         # Stage 1: encoder + FiLM + attention
│   │   ├── claim_self_attention.py   # [A1] Intra-party self-attention
│   │   ├── cross_attention.py        # P↔D cross-attention
│   │   ├── conditioning.py           # FiLM conditioner
│   │   ├── encode_text.py            # Shared ModernBERT encoder
│   │   ├── re_module.py              # Stage 2: rationale extraction heads
│   │   ├── pooling.py                # Stage 3: rationale pooling + learned fallback
│   │   └── td_head.py                # Stage 4: rationale-aligned TP head
│   ├── losses/
│   │   ├── multitask_loss.py         # 0.33 L_RE + 0.67 L_TP
│   │   ├── re_loss.py                # BCE loss for rationale extraction
│   │   └── td_loss.py                # BCE loss for tort prediction
│   ├── trainer/
│   │   ├── engine.py                 # Trainer class (train / eval / predict)
│   │   ├── metrics.py                # F1, accuracy, optimal threshold search
│   │   ├── scheduler.py              # Teacher forcing scheduler
│   │   ├── train_pipeline.py         # Single-GPU build helpers
│   │   └── train_pipeline2x.py       # 2-GPU build helpers
│   └── data_utils/
│       ├── __init__.py               # collate_fn, seed_worker
│       ├── dataset.py                # LegalDataset
│       ├── dataloader.py             # build_dataloader
│       ├── preprocessing.py          # load_jsonl, normalize_case
│       └── split.py                  # train/dev/test split
└── outputs/
    └── run_01/
        ├── best_model.pt             # Best checkpoint (weights only)
        ├── last_checkpoint.pt        # Full checkpoint (resume)
        ├── training_log.json         # Per-epoch metrics history
        └── test_predictions.jsonl    # Inference output
```

---

## Requirements

- Python ≥ 3.10
- CUDA ≥ 11.8 (for GPU training)
- 1× or 2× NVIDIA GPU with ≥ 16 GB VRAM recommended

```bash
pip install -r requirements.txt
```
---

## Configuration

All hyperparameters are controlled from `config/config.yaml`.

---

## Training (single GPU)

```bash
# Train from scratch
python src/main.py

# Resume from last checkpoint (auto-detect outputs/run_01/last_checkpoint.pt)
python src/main.py --resume

# Resume from a specific checkpoint file
python src/main.py --resume --ckpt outputs/run_01/last_checkpoint.pt

# Evaluate best model on dev + test (no training)
python src/main.py --eval-only

# Generate predictions on test set using best model (no training)
python src/main.py --predict-only
```

All runs load config from `config/config.yaml` by default.
To use a different config file:

```bash
python src/main.py --config config/config_run2.yaml
```

---

## Inference on test set

Run inference on the competition test file (no labels required):

```bash
python src/inference.py \
  --test-path  datasets/LJPJT26-test001.jsonl \
  --ckpt       outputs/run_01/best_model.pt
```


Additional options:

```bash
python src/inference.py \
  --test-path       datasets/LJPJT26-test001.jsonl \
  --ckpt            outputs/run_01/best_model.pt \
  --output-path     submissions/run01_full.jsonl \
  --submission-path submissions/run01_submit.jsonl \
  --threshold       0.5 \
  --batch-size      8 \
  --device          cuda
```

To run on CPU (e.g., for debugging):

```bash
python src/inference.py \
  --test-path datasets/LJPJT26-test001.jsonl \
  --ckpt      outputs/run_01/best_model.pt \
  --device    cpu \
  --no-amp
```

---

## Output format

Each line in `submission.jsonl` is a JSON object:

```json
{
  "tort_id": "case_001",
  "T_pred":  1,
  "rP_pred": [1, 0, 1],
  "rD_pred": [0, 1]
}
```

Each line in `inference_output.jsonl` additionally contains:

```json
{
  "tort_id":  "case_001",
  "T_hat":    0.831,
  "T_pred":   1,
  "rP_hat":   [0.912, 0.124, 0.783],
  "rP_pred":  [1, 0, 1],
  "rD_hat":   [0.051, 0.674],
  "rD_pred":  [0, 1],
  "P_claims": ["...", "...", "..."],
  "D_claims": ["...", "..."]
}
```

---

## Checkpoints

| File | Description |
|---|---|
| `best_model.pt` | Model weights only. Used for inference and `--eval-only`. Loaded with `weights_only=True`. |
| `last_checkpoint.pt` | Full training state: weights + optimizer + lr scheduler + AMP scaler + history + epoch counter. Used for `--resume`. |

To manually load the best model for custom evaluation:

```python
import torch
from src.models.shared_encoder import Stage1Encoder
# ... (build all 4 stages matching config)

ckpt = torch.load("outputs/run_01/best_model.pt",
                  map_location="cuda", weights_only=True)
stage1.load_state_dict(ckpt["stage1"])
stage2.load_state_dict(ckpt["stage2"])
stage3.load_state_dict(ckpt["stage3"])
stage4.load_state_dict(ckpt["stage4"])
```

---

## Model architecture summary

```
Input: U (undisputed facts), P = {p_1…p_M}, D = {d_1…d_N}

Stage 1 — Contextual Claim Encoding
  Shared ModernBERT encoder -> H_u, h_P, h_D
  Claim-specific fact retrieval + opposing-claim attention + gated fusion

Stage 2 — Rationale Extraction (RE)
  REHead_P: MLP -> logits/probabilities for plaintiff claims
  REHead_D: MLP -> logits/probabilities for defendant claims
  L_RE: BCE for plaintiff claims + BCE for defendant claims

Stage 3 — Rationale Pooling
  Rationale-guided attention pooling
  Learned fallback attention pooling
  Representation-based mixing gate
  Top-k claim tokens and rationale scores are passed to Stage 4
  No handcrafted rationale statistics are produced or consumed

Stage 4 — Tort Prediction (TP)
  Dual verdict queries
  Rationale-biased cross-attention over top-k claims
  Interaction feature construction and transformer reasoning
  Verdict MLP -> T_logit, T_hat
  L_TP: BCEWithLogitsLoss

Main objective
  L_total = 0.33 * L_RE + 0.67 * L_TP
  No alignment, consistency, uncertainty-weighting, contrastive,
  teacher-forcing KL, or other auxiliary loss terms are used.
```

> **Checkpoint note:** Stage 3 and Stage 4 parameter shapes changed after removing
> rationale statistics. Retrain from scratch; older checkpoints are not directly
> compatible with this architecture.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{UIT_TortNLP_COLIEE2026,
  title     = {Joint Rationale Extraction and Tort Determination
               for Japanese Legal Judgment Prediction},
  author    = {UIT TortNLP Team},
  booktitle = {Proceedings of COLIEE 2026},
  year      = {2026}
}
```
---

## Final V2 (recommended)

The default `src/main.py` now runs the two-phase detached-rationale residual
architecture. See [`FINAL_V2_GUIDE.md`](FINAL_V2_GUIDE.md).

```bash
python src/main.py \
  --config config/final_v2.yaml \
  --phase1-checkpoint outputs/ablations/joint_no_rationale/best_model.pt
```

The previous single-phase entry point is retained as `src/main_legacy.py`.
