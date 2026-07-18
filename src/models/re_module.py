"""Stage 2: claim-level rationale extraction."""

from __future__ import annotations

import torch
import torch.nn as nn

from .adapters import build_adapter


class REHead(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, h: torch.Tensor):
        """Args: h: [num_claims, hidden]."""
        if h.size(0) == 0:
            return h.new_zeros(0), h.new_zeros(0)

        logits = self.classifier(h).squeeze(-1)
        logits = torch.clamp(logits, -20, 20)
        probs = torch.nan_to_num(torch.sigmoid(logits))
        return logits, probs


class RationableExtraction(nn.Module):
    """Two RE heads with an optional RE-specific bottleneck adapter."""

    def __init__(
        self,
        hidden: int,
        use_task_adapter: bool = False,
        adapter_bottleneck: int = 128,
        adapter_dropout: float = 0.1,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.re_adapter = build_adapter(
            hidden_size=hidden,
            enabled=use_task_adapter,
            bottleneck_size=adapter_bottleneck,
            dropout=adapter_dropout,
        )
        self.re_plaintiff = REHead(hidden, dropout=head_dropout)
        self.re_defendant = REHead(hidden, dropout=head_dropout)

    def forward(self, stage1_output):
        hP = self.re_adapter(stage1_output["hP_cond"])
        hD = self.re_adapter(stage1_output["hD_cond"])

        logits_P, probs_P = self.re_plaintiff(hP)
        logits_D, probs_D = self.re_defendant(hD)

        return {
            "logits_P": logits_P,
            "logits_D": logits_D,
            "rP_hat": probs_P,
            "rD_hat": probs_D,
            "hP_re": hP,
            "hD_re": hD,
        }
