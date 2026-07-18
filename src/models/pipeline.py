"""Convenience wrapper around the four-stage legal prediction model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .pooling import RationalePooling
from .re_module import RationableExtraction
from .shared_encoder import Stage1Encoder
from .td_head import TDHead


class LegalPipeline(nn.Module):
    """Run Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 in one module.

    The wrapper keeps optional rationale teacher forcing, but it does not
    compute rationale statistics or auxiliary losses.
    """

    def __init__(
        self,
        model_name: str = "sbintuitions/modernbert-ja-310m",
        claim_chunk_size: int = 64,
        fusion_heads: int = 8,
        topk_fact_tokens: int = 16,
        topk_opponents: int = 3,
        fusion_dropout: float = 0.1,
        topk_claims: int = 5,
        aggregation_dropout: float = 0.1,
        td_num_heads: int = 8,
        td_dropout: float = 0.2,
        eta: float = 1.0,
    ) -> None:
        super().__init__()

        self.stage1 = Stage1Encoder(
            model_name=model_name,
            claim_chunk_size=claim_chunk_size,
            num_heads=fusion_heads,
            topk_fact_tokens=topk_fact_tokens,
            topk_opponents=topk_opponents,
            dropout=fusion_dropout,
        )
        hidden = self.stage1.encoder.hidden_size

        self.stage2 = RationableExtraction(hidden)
        self.stage3 = RationalePooling(
            hidden=hidden,
            topk_claims=topk_claims,
            dropout=aggregation_dropout,
        )
        self.stage4 = TDHead(
            hidden=hidden,
            num_heads=td_num_heads,
            dropout=td_dropout,
        )

        self.eta = float(eta)

    def _mixed_rationale(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor | None,
    ) -> torch.Tensor:
        if label is None or not self.training or self.eta <= 0.0:
            return prediction

        label = label.to(device=prediction.device, dtype=prediction.dtype)
        valid = label >= 0
        gold_or_prediction = torch.where(valid, label, prediction.detach())
        return self.eta * gold_or_prediction + (1.0 - self.eta) * prediction

    def forward(
        self,
        batch: dict,
        gt_rP: torch.Tensor | None = None,
        gt_rD: torch.Tensor | None = None,
    ) -> dict:
        s1 = self.stage1(batch)
        s2 = self.stage2(s1)

        s2_for_pool = dict(s2)
        s2_for_pool["rP_for_pool"] = self._mixed_rationale(s2["rP_hat"], gt_rP)
        s2_for_pool["rD_for_pool"] = self._mixed_rationale(s2["rD_hat"], gt_rD)

        s3 = self.stage3(s1, s2_for_pool, batch)
        s4 = self.stage4(s1, s3)

        return {
            **s2,
            **s4,
            "H_re_p": s3["H_re_p"],
            "H_re_d": s3["H_re_d"],
            "top_indices_P": s3["top_indices_P"],
            "top_indices_D": s3["top_indices_D"],
        }
