import argparse
import gc
import inspect
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)



def load_jsonl(path: str) -> List[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def normalize_case(case: dict) -> dict:
    facts = [f.get("description", "") for f in case.get("undisputed_facts", [])]
    p_claims = []
    r_p = []
    for c in case.get("plaintiff_claims", []):
        p_claims.append(c.get("description", ""))
        label = c.get("is_accepted")
        r_p.append(-1 if label is None else int(label))

    d_claims = []
    r_d = []
    for c in case.get("defendant_claims", []):
        d_claims.append(c.get("description", ""))
        label = c.get("is_accepted")
        r_d.append(-1 if label is None else int(label))

    t = case.get("court_decision")
    t = -1 if t is None else int(t)

    return {
        "tort_id": case.get("tort_id", ""),
        "U": " ".join(facts),
        "P": p_claims,
        "D": d_claims,
        "R_P": r_p,
        "R_D": r_d,
        "T": t,
    }


def fallback_build_dataset(jsonl_path: str) -> List[dict]:
    dataset = []
    for case in load_jsonl(jsonl_path):
        try:
            dataset.append(normalize_case(case))
        except Exception as e:
            print("Skip corrupted case:", e)
    return dataset


def build_dataset(jsonl_path: str) -> List[dict]:
    return fallback_build_dataset(jsonl_path)


# -----------------------------------------------------------------------------
# Text construction
# -----------------------------------------------------------------------------


def _lines(prefix: str, texts: List[str]) -> List[str]:
    return [f"{prefix}{i}: {text}" for i, text in enumerate(texts)]


def format_tp_input(sample: dict) -> str:
    p_lines = _lines("P", sample.get("P", []))
    d_lines = _lines("D", sample.get("D", []))
    return (
        "[UNDISPUTED FACTS]\n"
        f"{sample.get('U', '')}\n\n"
        "[PLAINTIFF CLAIMS]\n"
        f"{chr(10).join(p_lines)}\n\n"
        "[DEFENDANT CLAIMS]\n"
        f"{chr(10).join(d_lines)}"
    )


def format_re_input(sample: dict, side: str, claim_idx: int) -> str:
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


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------


@dataclass
class TPExample:
    text: str
    label: int
    tort_id: str


@dataclass
class REExample:
    text: str
    label: int
    tort_id: str
    side: str
    claim_idx: int
    case_idx: int = -1


class SequenceClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        encoded = self.tokenizer(ex.text, truncation=True, max_length=self.max_length)
        encoded["labels"] = int(ex.label)
        return encoded


class PredictionTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length)


def load_tp_examples(jsonl_path: str) -> List[TPExample]:
    examples = []
    for sample in build_dataset(jsonl_path):
        label = sample.get("T", -1)
        if label is None or int(label) < 0:
            continue
        examples.append(TPExample(format_tp_input(sample), int(label), str(sample.get("tort_id", ""))))
    if not examples:
        raise ValueError(f"No TP examples with valid T labels found in: {jsonl_path}")
    return examples


def load_re_examples(jsonl_path: str) -> List[REExample]:
    examples = []
    for case_idx, sample in enumerate(build_dataset(jsonl_path)):
        tort_id = str(sample.get("tort_id", ""))
        for i, label in enumerate(sample.get("R_P", [])):
            if label is None or int(label) < 0:
                continue
            examples.append(REExample(format_re_input(sample, "P", i), int(label), tort_id, "P", i, case_idx))
        for i, label in enumerate(sample.get("R_D", [])):
            if label is None or int(label) < 0:
                continue
            examples.append(REExample(format_re_input(sample, "D", i), int(label), tort_id, "D", i, case_idx))
    if not examples:
        raise ValueError(f"No RE examples with valid claim labels found in: {jsonl_path}")
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


def compute_tp_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, _, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision,
        "recall": recall,
    }


def compute_re_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, _, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision,
        "recall": recall,
    }


def build_training_args(args, output_dir: str, train_dataset_size: int, task: str) -> TrainingArguments:
    batch_size = args.tp_batch_size if task == "tp" else args.re_batch_size
    eval_batch_size = args.tp_eval_batch_size if task == "tp" else args.re_eval_batch_size
    grad_accum = args.tp_grad_accum if task == "tp" else args.re_grad_accum
    lr = args.tp_lr if task == "tp" else args.re_lr
    epochs = args.tp_epochs if task == "tp" else args.re_epochs

    effective_batch_size = max(1, batch_size * grad_accum)
    steps_per_epoch = int(np.ceil(train_dataset_size / effective_batch_size))
    total_steps = max(1, int(steps_per_epoch * epochs))
    warmup_steps = int(total_steps * args.warmup_ratio)

    kwargs = dict(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
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
    sig = inspect.signature(Trainer.__init__)
    params = sig.parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    return kwargs


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Train functions
# -----------------------------------------------------------------------------


def train_tp(args, tokenizer) -> str:
    tp_output_dir = str(Path(args.output_root) / "modernbert_tp_baseline")
    train_examples = load_tp_examples(args.train_path)
    dev_examples = load_tp_examples(args.dev_path)

    print("\n========== TP TRAINING ==========")
    print(f"Loaded TP train examples: {len(train_examples)}")
    print(f"Loaded TP dev examples:   {len(dev_examples)}")
    print(f"Train label counts: {np.bincount([x.label for x in train_examples], minlength=2).tolist()}")
    print(f"Dev label counts:   {np.bincount([x.label for x in dev_examples], minlength=2).tolist()}")

    train_ds = SequenceClassificationDataset(train_examples, tokenizer, args.tp_max_length)
    dev_ds = SequenceClassificationDataset(dev_examples, tokenizer, args.tp_max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "NOT_AFFIRMED", 1: "AFFIRMED"},
        label2id={"NOT_AFFIRMED": 0, "AFFIRMED": 1},
    )

    class_weights = compute_class_weights([x.label for x in train_examples]) if args.use_class_weights else None
    if class_weights is not None:
        print(f"Using TP class weights: {class_weights.tolist()}")

    trainer = WeightedCETrainer(
        **build_trainer_kwargs(
            tokenizer,
            model=model,
            args=build_training_args(args, tp_output_dir, len(train_ds), task="tp"),
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_tp_metrics,
            class_weights=class_weights,
        )
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("TP dev metrics:", metrics)
    trainer.save_model(tp_output_dir)
    tokenizer.save_pretrained(tp_output_dir)
    with open(Path(tp_output_dir) / "dev_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    del trainer, model
    clear_cuda()
    return tp_output_dir


def train_re(args, tokenizer) -> str:
    re_output_dir = str(Path(args.output_root) / "modernbert_re_baseline")
    train_examples = load_re_examples(args.train_path)
    dev_examples = load_re_examples(args.dev_path)

    print("\n========== RE TRAINING ==========")
    print(f"Loaded RE train claim examples: {len(train_examples)}")
    print(f"Loaded RE dev claim examples:   {len(dev_examples)}")
    print(f"Train label counts: {np.bincount([x.label for x in train_examples], minlength=2).tolist()}")
    print(f"Dev label counts:   {np.bincount([x.label for x in dev_examples], minlength=2).tolist()}")
    print(f"Train side counts:  P={sum(x.side == 'P' for x in train_examples)}, D={sum(x.side == 'D' for x in train_examples)}")
    print(f"Dev side counts:    P={sum(x.side == 'P' for x in dev_examples)}, D={sum(x.side == 'D' for x in dev_examples)}")

    train_ds = SequenceClassificationDataset(train_examples, tokenizer, args.re_max_length)
    dev_ds = SequenceClassificationDataset(dev_examples, tokenizer, args.re_max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "REJECTED", 1: "ACCEPTED"},
        label2id={"REJECTED": 0, "ACCEPTED": 1},
    )

    class_weights = compute_class_weights([x.label for x in train_examples]) if args.use_class_weights else None
    if class_weights is not None:
        print(f"Using RE class weights: {class_weights.tolist()}")

    trainer = WeightedCETrainer(
        **build_trainer_kwargs(
            tokenizer,
            model=model,
            args=build_training_args(args, re_output_dir, len(train_ds), task="re"),
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_re_metrics,
            class_weights=class_weights,
        )
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("RE dev metrics:", metrics)
    trainer.save_model(re_output_dir)
    tokenizer.save_pretrained(re_output_dir)
    with open(Path(re_output_dir) / "dev_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    del trainer, model
    clear_cuda()
    return re_output_dir


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


@torch.no_grad()
def predict_texts(
    model_dir: str,
    texts: List[str],
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[List[int], List[float]]:
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    ds = PredictionTextDataset(texts, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    preds = []
    probs_1 = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        preds.extend(pred.detach().cpu().numpy().astype(int).tolist())
        probs_1.extend(probs[:, 1].detach().cpu().numpy().astype(float).tolist())

    del model
    clear_cuda()
    return preds, probs_1


def predict_test(args, tp_model_dir: str, re_model_dir: str, tokenizer):
    print("\n========== TEST PREDICTION ==========")
    samples = build_dataset(args.test_path)
    raw_cases = load_jsonl(args.test_path)
    if len(raw_cases) != len(samples):
        print("Warning: raw case count and normalized sample count differ; output will use normalized samples only.")

    device = resolve_device(args.device)
    print(f"Prediction device: {device}")

    # TP predictions, one per case.
    tp_texts = [format_tp_input(s) for s in samples]
    tp_preds, tp_probs = predict_texts(
        tp_model_dir,
        tp_texts,
        tokenizer,
        args.tp_max_length,
        args.predict_batch_size,
        device,
    )

    outputs = []
    for i, sample in enumerate(samples):
        outputs.append(
            {
                "tort_id": str(sample.get("tort_id", "")),
                "court_decision": bool(tp_preds[i]),
                "court_decision_label": int(tp_preds[i]),
                "court_decision_prob": float(tp_probs[i]),
                "plaintiff_claims": [
                    {"id": f"P{j}", "description": text, "is_accepted": False, "accepted_prob": 0.0}
                    for j, text in enumerate(sample.get("P", []))
                ],
                "defendant_claims": [
                    {"id": f"D{j}", "description": text, "is_accepted": False, "accepted_prob": 0.0}
                    for j, text in enumerate(sample.get("D", []))
                ],
                "accepted_plaintiff_claims": [],
                "accepted_defendant_claims": [],
            }
        )

    # RE predictions, one per claim.
    re_texts = []
    re_meta = []
    for case_idx, sample in enumerate(samples):
        for j in range(len(sample.get("P", []))):
            re_texts.append(format_re_input(sample, "P", j))
            re_meta.append((case_idx, "P", j))
        for j in range(len(sample.get("D", []))):
            re_texts.append(format_re_input(sample, "D", j))
            re_meta.append((case_idx, "D", j))

    if re_texts:
        re_preds, re_probs = predict_texts(
            re_model_dir,
            re_texts,
            tokenizer,
            args.re_max_length,
            args.predict_batch_size,
            device,
        )
        for (case_idx, side, claim_idx), pred, prob in zip(re_meta, re_preds, re_probs):
            accepted = bool(pred)
            if side == "P":
                outputs[case_idx]["plaintiff_claims"][claim_idx]["is_accepted"] = accepted
                outputs[case_idx]["plaintiff_claims"][claim_idx]["accepted_prob"] = float(prob)
                if accepted:
                    outputs[case_idx]["accepted_plaintiff_claims"].append(f"P{claim_idx}")
            else:
                outputs[case_idx]["defendant_claims"][claim_idx]["is_accepted"] = accepted
                outputs[case_idx]["defendant_claims"][claim_idx]["accepted_prob"] = float(prob)
                if accepted:
                    outputs[case_idx]["accepted_defendant_claims"].append(f"D{claim_idx}")

    output_path = Path(args.pred_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved predictions to: {output_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--dev_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="outputs/modernbert_baseline")
    parser.add_argument("--pred_output_path", type=str, default="outputs/modernbert_baseline/test_predictions.jsonl")
    parser.add_argument("--model_name", type=str, default="sbintuitions/modernbert-ja-310m")

    parser.add_argument("--do_train", action="store_true", help="Train TP and RE sequentially.")
    parser.add_argument("--do_predict", action="store_true", help="Predict on test after training or using provided model dirs.")
    parser.add_argument("--tp_model_dir", type=str, default=None, help="Existing TP model dir for prediction if not training.")
    parser.add_argument("--re_model_dir", type=str, default=None, help="Existing RE model dir for prediction if not training.")

    parser.add_argument("--tp_max_length", type=int, default=4096)
    parser.add_argument("--re_max_length", type=int, default=4096)
    parser.add_argument("--tp_batch_size", type=int, default=1)
    parser.add_argument("--re_batch_size", type=int, default=2)
    parser.add_argument("--tp_eval_batch_size", type=int, default=1)
    parser.add_argument("--re_eval_batch_size", type=int, default=2)
    parser.add_argument("--tp_grad_accum", type=int, default=8)
    parser.add_argument("--re_grad_accum", type=int, default=8)
    parser.add_argument("--tp_epochs", type=float, default=3)
    parser.add_argument("--re_epochs", type=float, default=3)
    parser.add_argument("--tp_lr", type=float, default=2e-5)
    parser.add_argument("--re_lr", type=float, default=2e-5)

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--predict_batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    # Default behavior: if neither flag is specified, do both.
    if not args.do_train and not args.do_predict:
        args.do_train = True
        args.do_predict = True

    if args.do_train and (args.train_path is None or args.dev_path is None):
        raise ValueError("--train_path and --dev_path are required when --do_train is enabled.")

    return args


def main():
    args = parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    tp_model_dir = args.tp_model_dir
    re_model_dir = args.re_model_dir

    if args.do_train:
        tp_model_dir = train_tp(args, tokenizer)
        re_model_dir = train_re(args, tokenizer)

    if args.do_predict:
        if tp_model_dir is None:
            tp_model_dir = str(Path(args.output_root) / "modernbert_tp_baseline")
        if re_model_dir is None:
            re_model_dir = str(Path(args.output_root) / "modernbert_re_baseline")
        predict_test(args, tp_model_dir, re_model_dir, tokenizer)


if __name__ == "__main__":
    main()
