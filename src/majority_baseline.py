"""Compute majority baselines on the same deterministic split."""

from __future__ import annotations

import argparse
import json
import yaml

from data_utils.preprocessing import build_dataset
from data_utils.split import train_dev_test_split


def majority_accuracy(labels):
    valid = [int(label) for label in labels if label >= 0]
    if not valid:
        return float("nan")
    positive = sum(valid)
    negative = len(valid) - positive
    return max(positive, negative) / len(valid)


def positive_f1_for_constant(labels, prediction):
    valid = [int(label) for label in labels if label >= 0]
    tp = sum(label == 1 and prediction == 1 for label in valid)
    fp = sum(label == 0 and prediction == 1 for label in valid)
    fn = sum(label == 1 and prediction == 0 for label in valid)
    return 2 * tp / (2 * tp + fp + fn + 1e-8)


def summarize(samples):
    tp_labels = [sample["T"] for sample in samples]
    re_labels = [
        label
        for sample in samples
        for label in list(sample["R_P"]) + list(sample["R_D"])
    ]
    re_zero = positive_f1_for_constant(re_labels, 0)
    re_one = positive_f1_for_constant(re_labels, 1)
    return {
        "num_cases": len(samples),
        "tp_majority_accuracy": majority_accuracy(tp_labels),
        "re_all_zero_positive_f1": re_zero,
        "re_all_one_positive_f1": re_one,
        "re_best_constant_positive_f1": max(re_zero, re_one),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    samples = build_dataset(cfg["data"]["train_path"])
    train, dev, test = train_dev_test_split(samples, seed=int(cfg.get("seed", 42)))
    print(json.dumps({"train": summarize(train), "dev": summarize(dev), "test": summarize(test)}, indent=2))


if __name__ == "__main__":
    main()
