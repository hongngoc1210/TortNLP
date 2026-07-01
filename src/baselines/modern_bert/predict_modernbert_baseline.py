"""
Inference script for the simple ModernBERT TP + RE baselines.

It reuses data_utils.preprocessing.build_dataset to normalize the COLIEE JSONL.
The output is JSONL with:
    tort_id
    court_decision / tort_affirmed
    plaintiff_claims:  [{id, is_accepted, score}]
    defendant_claims:  [{id, is_accepted, score}]
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_utils.preprocessing import build_dataset
from train_modernbert_tp_baseline import format_tp_input
from train_modernbert_re_baseline import format_re_input


@torch.no_grad()
def predict_positive_probs(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> List[float]:
    model.eval()
    probs: List[float] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Predict", leave=False):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**encoded).logits
        batch_probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
        probs.extend(float(x) for x in batch_probs)

    return probs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--tp_model_dir", type=str, required=True)
    parser.add_argument("--re_model_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="outputs/modernbert_baseline_predictions.jsonl")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--tp_threshold", type=float, default=0.5)
    parser.add_argument("--re_threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    samples = build_dataset(args.input_path)
    print(f"Loaded cases: {len(samples)}")

    # ------------------------- TP prediction -------------------------
    tp_tokenizer = AutoTokenizer.from_pretrained(args.tp_model_dir, use_fast=True)
    tp_model = AutoModelForSequenceClassification.from_pretrained(args.tp_model_dir).to(device)

    tp_texts = [format_tp_input(sample) for sample in samples]
    tp_probs = predict_positive_probs(
        tp_model, tp_tokenizer, tp_texts, device, args.max_length, args.batch_size
    )

    # ------------------------- RE prediction -------------------------
    re_tokenizer = AutoTokenizer.from_pretrained(args.re_model_dir, use_fast=True)
    re_model = AutoModelForSequenceClassification.from_pretrained(args.re_model_dir).to(device)

    re_texts: List[str] = []
    re_index = []  # (case_idx, side, claim_idx)

    for case_idx, sample in enumerate(samples):
        for i in range(len(sample.get("P", []))):
            re_texts.append(format_re_input(sample, "P", i))
            re_index.append((case_idx, "P", i))
        for i in range(len(sample.get("D", []))):
            re_texts.append(format_re_input(sample, "D", i))
            re_index.append((case_idx, "D", i))

    re_probs = predict_positive_probs(
        re_model, re_tokenizer, re_texts, device, args.max_length, args.batch_size
    )

    # Allocate back to cases.
    p_scores = [[] for _ in samples]
    d_scores = [[] for _ in samples]
    for (case_idx, side, claim_idx), score in zip(re_index, re_probs):
        if side == "P":
            p_scores[case_idx].append(score)
        else:
            d_scores[case_idx].append(score)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for case_idx, sample in enumerate(samples):
            tp_score = tp_probs[case_idx]
            out = {
                "tort_id": sample.get("tort_id", ""),
                "court_decision": bool(tp_score >= args.tp_threshold),
                "tort_affirmed": bool(tp_score >= args.tp_threshold),
                "tort_score": tp_score,
                "plaintiff_claims": [
                    {
                        "id": f"P{i}",
                        "is_accepted": bool(score >= args.re_threshold),
                        "score": score,
                    }
                    for i, score in enumerate(p_scores[case_idx])
                ],
                "defendant_claims": [
                    {
                        "id": f"D{i}",
                        "is_accepted": bool(score >= args.re_threshold),
                        "score": score,
                    }
                    for i, score in enumerate(d_scores[case_idx])
                ],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Saved predictions to: {args.output_path}")


if __name__ == "__main__":
    main()
