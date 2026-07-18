"""Stage 4: tort prediction with explicit ablation modes."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import RationaleBiasedCrossAttention, valid_num_heads


class TDHead(nn.Module):
    """Verdict head.

    ``input_mode`` controls the TP-path ablation:
      - ``rationale``: full current architecture.
      - ``global_only``: only the global undisputed-facts representation.

    ``use_global_residual`` adds a conservative residual route from global
    context to the full rationale-based interaction representation.
    """

    def __init__(
        self,
        hidden: int,
        num_heads: int = 8,
        dropout: float = 0.2,
        input_mode: str = "rationale",
        use_global_residual: bool = False,
        rationale_scale_init: float = -1.5,
    ) -> None:
        super().__init__()

        if input_mode not in {"rationale", "global_only"}:
            raise ValueError(f"Unknown TDHead input_mode={input_mode!r}")
        self.input_mode = input_mode
        self.use_global_residual = bool(use_global_residual)

        heads = valid_num_heads(hidden, num_heads)

        self.global_verdict_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

        self.plaintiff_query = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.defendant_query = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.plaintiff_cross_attention = RationaleBiasedCrossAttention(
            hidden, heads, dropout=dropout
        )
        self.defendant_cross_attention = RationaleBiasedCrossAttention(
            hidden, heads, dropout=dropout
        )

        self.feature_type_embedding = nn.Parameter(torch.empty(8, hidden))
        nn.init.normal_(self.feature_type_embedding, mean=0.0, std=0.02)

        reasoner_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.interaction_reasoner = nn.TransformerEncoder(
            reasoner_layer, num_layers=1
        )

        self.feature_pool_query = nn.Parameter(torch.empty(hidden))
        nn.init.normal_(self.feature_pool_query, mean=0.0, std=0.02)

        self.global_residual_norm = nn.LayerNorm(hidden)
        self.rationale_scale = nn.Parameter(torch.tensor(float(rationale_scale_init)))

        self.verdict_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def _empty_diagnostics(H_u: torch.Tensor) -> dict:
        batch_size = H_u.size(0)
        return {
            "verdict_attention_P": H_u.new_zeros((batch_size, 0)),
            "verdict_attention_D": H_u.new_zeros((batch_size, 0)),
            "feature_attention": H_u.new_zeros((batch_size, 0)),
            "Z_p": H_u.new_zeros(H_u.shape),
            "Z_d": H_u.new_zeros(H_u.shape),
        }

    def forward(self, stage1_out, stage3_out=None, input_mode: str | None = None):
        mode = input_mode or self.input_mode
        if mode not in {"rationale", "global_only"}:
            raise ValueError(f"Unknown TP input mode={mode!r}")

        if stage3_out is not None and "H_u_tp" in stage3_out:
            H_u = stage3_out["H_u_tp"]
        else:
            H_u = stage1_out["H_u"]

        if mode == "global_only":
            T_logit = self.global_verdict_mlp(H_u).squeeze(-1)
            T_hat = torch.sigmoid(T_logit)
            return {
                "T_logit": T_logit,
                "T_hat": T_hat,
                **self._empty_diagnostics(H_u),
            }

        if stage3_out is None:
            raise ValueError("stage3_out is required for rationale TP mode")

        H_p = stage3_out["H_re_p"]
        H_d = stage3_out["H_re_d"]

        q_p = self.plaintiff_query(H_u)
        q_d = self.defendant_query(H_u)

        Z_p, verdict_attention_P = self.plaintiff_cross_attention(
            query=q_p,
            tokens=stage3_out["top_tokens_P"],
            token_mask=stage3_out["top_mask_P"],
            rationale_scores=stage3_out["top_scores_P"],
        )
        Z_d, verdict_attention_D = self.defendant_cross_attention(
            query=q_d,
            tokens=stage3_out["top_tokens_D"],
            token_mask=stage3_out["top_mask_D"],
            rationale_scores=stage3_out["top_scores_D"],
        )

        delta = H_p - H_d
        absolute_delta = torch.abs(delta)
        product = H_p * H_d

        feature_tokens = torch.stack(
            [H_u, H_p, H_d, Z_p, Z_d, delta, absolute_delta, product], dim=1
        )
        feature_tokens = feature_tokens + self.feature_type_embedding.unsqueeze(0)
        reasoned_tokens = self.interaction_reasoner(feature_tokens)

        pool_logits = torch.einsum("bkh,h->bk", reasoned_tokens, self.feature_pool_query)
        pool_weights = torch.softmax(pool_logits, dim=-1)
        z = torch.sum(pool_weights.unsqueeze(-1) * reasoned_tokens, dim=1)

        if self.use_global_residual:
            alpha = torch.sigmoid(self.rationale_scale)
            z = self.global_residual_norm(H_u + alpha * z)

        T_logit = self.verdict_mlp(z).squeeze(-1)
        T_hat = torch.sigmoid(T_logit)

        return {
            "T_logit": T_logit,
            "T_hat": T_hat,
            "verdict_attention_P": verdict_attention_P,
            "verdict_attention_D": verdict_attention_D,
            "feature_attention": pool_weights,
            "Z_p": Z_p,
            "Z_d": Z_d,
            "rationale_scale": torch.sigmoid(self.rationale_scale).detach(),
        }
