"""Rationale extraction loss."""

from __future__ import annotations

import torch
import torch.nn as nn


class RELoss(nn.Module):
    """BCE loss for plaintiff and defendant rationales.

    ``side_reduction='mean'`` is recommended.  The old implementation summed
    the two side losses, making the effective RE contribution roughly twice as
    large as expected from a nominal 0.33 loss weight.
    """

    def __init__(self, side_reduction: str = "mean") -> None:
        super().__init__()
        if side_reduction not in {"mean", "sum"}:
            raise ValueError("side_reduction must be 'mean' or 'sum'")
        self.side_reduction = side_reduction
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _side_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        if logits.numel() == 0:
            return None
        labels = labels.float()
        mask = labels >= 0
        if not mask.any():
            return None
        return self.loss_fn(logits[mask], labels[mask])

    def forward(self, outputs, batch):
        side_losses = []
        loss_p = self._side_loss(outputs["logits_P"], batch["R_P"])
        loss_d = self._side_loss(outputs["logits_D"], batch["R_D"])
        if loss_p is not None:
            side_losses.append(loss_p)
        if loss_d is not None:
            side_losses.append(loss_d)

        if not side_losses:
            reference = outputs["logits_P"]
            return reference.sum() * 0.0

        stacked = torch.stack(side_losses)
        loss = stacked.mean() if self.side_reduction == "mean" else stacked.sum()
        return torch.nan_to_num(loss)
