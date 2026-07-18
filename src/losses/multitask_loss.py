"""Configurable objective for RE/TP ablations."""

from __future__ import annotations

import torch
import torch.nn as nn

from .re_loss import RELoss
from .td_loss import TDLoss


class MultiTaskLoss(nn.Module):
    """Compute joint, RE-only, or TP-only objectives.

    Joint mode keeps exactly two losses:
        L_total = weight_re * L_RE + weight_tp * L_TP
    """

    def __init__(
        self,
        weight_re: float = 0.33,
        weight_tp: float = 0.67,
        task_mode: str = "joint",
        re_side_reduction: str = "mean",
    ) -> None:
        super().__init__()
        if task_mode not in {"joint", "re_only", "tp_only"}:
            raise ValueError(f"Unknown task_mode={task_mode!r}")
        if weight_re < 0 or weight_tp < 0:
            raise ValueError("Loss weights must be non-negative.")
        if task_mode == "joint" and abs((weight_re + weight_tp) - 1.0) > 1e-6:
            raise ValueError("In joint mode, weight_re + weight_tp must equal 1.0.")

        self.re_loss = RELoss(side_reduction=re_side_reduction)
        self.tp_loss = TDLoss()
        self.weight_re = float(weight_re)
        self.weight_tp = float(weight_tp)
        self.task_mode = task_mode

    def forward(self, re_outputs, tp_outputs, batch):
        reference = None
        if re_outputs is not None:
            reference = re_outputs["logits_P"]
        elif tp_outputs is not None:
            reference = tp_outputs["T_logit"]
        else:
            raise ValueError("At least one task output is required")

        zero = reference.sum() * 0.0
        loss_re = self.re_loss(re_outputs, batch) if re_outputs is not None else zero
        loss_tp = self.tp_loss(tp_outputs, batch) if tp_outputs is not None else zero

        if self.task_mode == "re_only":
            loss = loss_re
        elif self.task_mode == "tp_only":
            loss = loss_tp
        else:
            loss = self.weight_re * loss_re + self.weight_tp * loss_tp

        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        return loss, loss_re, loss_tp

    def task_objectives(self, re_outputs, tp_outputs, batch):
        """Return weighted objectives separately for PCGrad/diagnostics."""
        loss_re = self.re_loss(re_outputs, batch)
        loss_tp = self.tp_loss(tp_outputs, batch)
        return self.weight_re * loss_re, self.weight_tp * loss_tp, loss_re, loss_tp
