import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,   # giảm từ 4.0 → 2.0: tránh loss_td → 0
        clip:      float = 0.0,   # tắt clip: tránh easy negatives mất gradient
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip      = clip

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        probs     = torch.sigmoid(logits)
        probs_neg = (probs + self.clip).clamp(max=1.0)

        loss_pos  = -labels       * torch.log(probs.clamp(min=1e-8))
        loss_neg  = -(1 - labels) * torch.log((1 - probs_neg).clamp(min=1e-8))

        loss_pos  = loss_pos * (1 - probs)   ** self.gamma_pos
        loss_neg  = loss_neg * probs_neg     ** self.gamma_neg

        return (loss_pos + loss_neg).mean()


class TDLoss(nn.Module):

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        clip:      float = 0.0,
    ):
        super().__init__()
        self.loss_fn = AsymmetricLoss(gamma_pos, gamma_neg, clip)

    def forward(self, outputs: dict, batch: dict) -> torch.Tensor:

        logits = outputs["T_logit"]
        labels = batch["T"].float()

        mask = labels >= 0
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)

        return torch.nan_to_num(self.loss_fn(logits[mask], labels[mask]))