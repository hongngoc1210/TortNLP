"""
re_loss.py  —  Focal Loss cho RE (fixed)

Fixes so với version trước:
  1. alpha=0.25 → alpha=0.75: trong legal RE, positives (R=1) là minority
     → cần UP-WEIGHT positives, không down-weight. alpha=0.25 là sai hướng
     (paper RetinaNet dùng 0.25 vì object detection có rất nhiều easy negatives
      chiếm 99%+; trong legal RE tỉ lệ thường 20-40% positives).
  2. gamma: giảm từ 2.0 → 1.0 để tránh scale down loss_re quá nhiều so với
     loss_td, gây mất cân bằng gradient khi kết hợp với uncertainty weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 1.0, alpha: float = 0.75):
        """
        gamma : focusing exponent.
                1.0 thay vì 2.0 — tránh scale loss_re xuống quá thấp
                so với loss_td khi dùng uncertainty weighting.
        alpha : weight cho POSITIVE class.
                0.75 → positives được weight 3x so với negatives.
                Tune theo tỉ lệ thực tế:
                  alpha ≈ N_neg / (N_pos + N_neg)
                Ví dụ 30% positive → alpha = 0.7
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        bce     = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        probs   = torch.sigmoid(logits)
        p_t     = probs * labels + (1 - probs) * (1 - labels)
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)

        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class RELoss(nn.Module):

    def __init__(self, gamma: float = 1.0, alpha: float = 0.75):
        super().__init__()
        self.loss_fn = FocalLoss(gamma=gamma, alpha=alpha)

    def forward(self, outputs: dict, batch: dict) -> torch.Tensor:

        logits_P = outputs["logits_P"]
        logits_D = outputs["logits_D"]
        labels_P = batch["R_P"].float()
        labels_D = batch["R_D"].float()

        loss_P = torch.tensor(0.0, device=logits_P.device)
        loss_D = torch.tensor(0.0, device=logits_D.device)

        if logits_P.numel() > 0:
            mask_P = labels_P >= 0
            if mask_P.any():
                loss_P = self.loss_fn(logits_P[mask_P], labels_P[mask_P])

        if logits_D.numel() > 0:
            mask_D = labels_D >= 0
            if mask_D.any():
                loss_D = self.loss_fn(logits_D[mask_D], labels_D[mask_D])

        return torch.nan_to_num(loss_P + loss_D)