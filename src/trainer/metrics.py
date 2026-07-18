from __future__ import annotations

import torch


def compute_re_f1(preds, labels, threshold: float = 0.5):
    if preds.numel() == 0:
        return float("nan")
    valid = labels >= 0
    if not valid.any():
        return float("nan")

    preds = (preds[valid] > threshold).float()
    labels = labels[valid].float()

    tp = torch.sum((preds == 1) & (labels == 1))
    fp = torch.sum((preds == 1) & (labels == 0))
    fn = torch.sum((preds == 0) & (labels == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.item()


def compute_td_accuracy(preds, labels, threshold: float = 0.5):
    if preds.numel() == 0:
        return float("nan")
    valid = labels >= 0
    if not valid.any():
        return float("nan")

    predictions = (preds[valid] > threshold).float()
    return (predictions == labels[valid].float()).float().mean().item()
