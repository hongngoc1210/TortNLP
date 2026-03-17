import torch
import torch.nn as nn


class REHead(nn.Module):

    def __init__(self, hidden):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden // 2, 1)
        )

    def forward(self, h):

        """
        h: [num_claims, hidden]
        """

        if h.size(0) == 0:
            return (
                torch.zeros(0, device=h.device),
                torch.zeros(0, device=h.device)
            )

        logits = self.classifier(h)

        logits = logits.squeeze(-1)

        # ---- clamp logits to avoid overflow ----
        logits = torch.clamp(logits, -20, 20)

        probs = torch.sigmoid(logits)

        probs = torch.nan_to_num(probs)

        return logits, probs


class RationableExtraction(nn.Module):

    def __init__(self, hidden):

        super().__init__()

        self.re_plaintiff = REHead(hidden)
        self.re_defendant = REHead(hidden)

    def forward(self, stage1_output):

        hP = stage1_output["hP_cond"]
        hD = stage1_output["hD_cond"]

        logits_P, probs_P = self.re_plaintiff(hP)
        logits_D, probs_D = self.re_defendant(hD)

        return {

            "logits_P": logits_P,
            "logits_D": logits_D,

            "rP_hat": probs_P,
            "rD_hat": probs_D
        }   