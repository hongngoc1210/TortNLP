import torch
import torch.nn as nn


class TDLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, outputs, batch):

        logits = outputs["T_logit"]
        labels = batch["T"].float()

        mask = labels >= 0
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)

        loss = self.loss_fn(logits[mask], labels[mask])

        loss = torch.nan_to_num(loss)

        return loss