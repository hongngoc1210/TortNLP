"""
ModernBERT baseline for COLIEE Rationale Extraction (RE).

This script intentionally uses the existing MTL data_utils:
    - data_utils.preprocessing.build_dataset
    - data_utils.preprocessing.normalize_case

Task:
    claim-level binary classification: whether each P/D claim is accepted.

Important difference from the MTL architecture:
    The MTL model encodes U, P, D separately and models interactions with FiLM,
    self-attention, cross-attention, and pooling. This baseline is intentionally
    simpler: each claim becomes one sequence-classification example for ModernBERT.
"""

import argparse
import inspect
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data_utils.preprocessing import build_dataset


# -----------------------------------------------------------------------------
# Text construction
# -----------------------------------------------------------------------------


def _lines(prefix: str, texts: List[str]) -> List[str]:
    return [f"{prefix}{i}: {text}" for i, text in enumerate(texts)]


def format_re_input(sample: dict, side: str, claim_idx: int) -> str:
    """
    Convert one claim into one ModernBERT input.

    side:
        "P" for plaintiff claim, "D" for defendant claim.
    """
    p_claims = sample.get("P", [])
    d_claims = sample.get("D", [])

    if side == "P":
        target_side = "PLAINTIFF"
        target_claim = f"P{claim_idx}: {p_claims[claim_idx]}"
    elif side == "D":
        target_side = "DEFENDANT"
        target_claim = f"D{claim_idx}: {d_claims[claim_idx]}"
    else:
        raise ValueError(f"Unknown side: {side}")

    return (
        "[UNDISPUTED FACTS]\n"
        f"{sample.get('U', '')}\n\n"
        "[TARGET SIDE]\n"
        f"{target_side}\n\n"
        "[TARGET CLAIM]\n"
        f"{target_claim}\n\n"
        "[ALL PLAINTIFF CLAIMS]\n"
        f"{chr(10).join(_lines('P', p_claims))}\n\n"
        "[ALL DEFENDANT CLAIMS]\n"
        f"{chr(10).join(_lines('D', d_claims))}"
    )


@dataclass
class REExample:
    text: str
    label: int
    tort_id: str
    side: str
    claim_idx: int


class REClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[REExample], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        encoded = self.tokenizer(
            ex.text,
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = int(ex.label)
        return encoded


def load_re_examples(jsonl_path: str) -> List[REExample]:
    """
    Build claim-level RE examples from the existing MTL normalized dataset.
    Claims with missing is_accepted label (-1) are skipped.
    """
    samples = build_dataset(jsonl_path)
    examples: List[REExample] = []

    for sample in samples:
        tort_id = str(sample.get("tort_id", ""))

        for i, label in enumerate(sample.get("R_P", [])):
            if label is None or int(label) < 0:
                continue
            examples.append(
                REExample(
                    text=format_re_input(sample, "P", i),
                    label=int(label),
                    tort_id=tort_id,
                    side="P",
                    claim_idx=i,
                )
            )

        for i, label in enumerate(sample.get("R_D", [])):
            if label is None or int(label) < 0:
                continue
            examples.append(
                REExample(
                    text=format_re_input(sample, "D", i),
                    label=int(label),
                    tort_id=tort_id,
                    side="D",
                    claim_idx=i,
                )
            )

    if not examples:
        raise ValueError(f"No RE training examples with valid claim labels found in: {jsonl_path}")
    return examples


# -----------------------------------------------------------------------------
# Trainer helpers
# -----------------------------------------------------------------------------


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedCETrainer(Trainer):
    """Trainer with optional class-weighted cross entropy for imbalanced labels."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    f1 = f1_score(labels, preds, zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def build_training_args(args, train_dataset_size: int) -> TrainingArguments:
    """Handle transformers version differences and avoid deprecated warmup_ratio."""
    effective_batch_size = max(1, args.batch_size * args.grad_accum)
    steps_per_epoch = int(np.ceil(train_dataset_size / effective_batch_size))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = int(total_steps * args.warmup_ratio)

    kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
        save_total_limit=2,
        seed=args.seed,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"

    return TrainingArguments(**kwargs)

def build_trainer_kwargs(tokenizer, **kwargs):
    """
    Make Trainer compatible with different transformers versions.
    New versions use processing_class instead of tokenizer.
    Some versions accept neither, so we skip it.
    """
    sig = inspect.signature(Trainer.__init__)
    params = sig.parameters

    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer

    return kwargs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--dev_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/modernbert_re_baseline")
    parser.add_argument("--model_name", type=str, default="sbintuitions/modernbert-ja-310m")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    train_examples = load_re_examples(args.train_path)
    dev_examples = load_re_examples(args.dev_path)

    print(f"Loaded RE train claim examples: {len(train_examples)}")
    print(f"Loaded RE dev claim examples:   {len(dev_examples)}")
    print(f"Train label counts: {np.bincount([x.label for x in train_examples], minlength=2).tolist()}")
    print(f"Dev label counts:   {np.bincount([x.label for x in dev_examples], minlength=2).tolist()}")
    print(f"Train side counts:  P={sum(x.side == 'P' for x in train_examples)}, D={sum(x.side == 'D' for x in train_examples)}")
    print(f"Dev side counts:    P={sum(x.side == 'P' for x in dev_examples)}, D={sum(x.side == 'D' for x in dev_examples)}")

    train_ds = REClassificationDataset(train_examples, tokenizer, args.max_length)
    dev_ds = REClassificationDataset(dev_examples, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "REJECTED", 1: "ACCEPTED"},
        label2id={"REJECTED": 0, "ACCEPTED": 1},
    )

    class_weights = compute_class_weights([x.label for x in train_examples]) if args.use_class_weights else None
    if class_weights is not None:
        print(f"Using class weights: {class_weights.tolist()}")

    trainer = WeightedCETrainer(
    **build_trainer_kwargs(
        tokenizer,
        model=model,
        args=build_training_args(args, train_dataset_size=len(train_ds)),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        )
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(Path(args.output_dir) / "dev_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
