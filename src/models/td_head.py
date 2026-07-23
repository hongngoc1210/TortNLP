"""Stage 4: globally anchored tort prediction.

The phase-1 global classifier remains an explicit anchor in the final model.
The rationale branch predicts a bounded residual correction:

    T_logit = global_logit + sigmoid(scale) * rationale_delta

This makes phase-2 initialization conservative and prevents noisy rationale
pooling from immediately replacing a useful global verdict representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import RationaleBiasedCrossAttention, valid_num_heads


class TDHead(nn.Module):
    """Verdict head with a global anchor and rationale residual.

    Modes
    -----
    ``global_only``
        Uses only the undisputed-facts/global representation.  This is the
        phase-1 objective and the TP-only/no-rationale ablation.

    ``rationale``
        Uses the same trained global logit, then adds a small learned correction
        from rationale-guided plaintiff/defendant interactions.
    """

    def __init__(
        self,
        hidden: int,
        num_heads: int = 8,
        dropout: float = 0.2,
        input_mode: str = "rationale",
        use_global_residual: bool = True,
        rationale_scale_init: float = -1.5,
    ) -> None:
        super().__init__()

        hidden = int(hidden)
        if input_mode not in {"rationale", "global_only"}:
            raise ValueError(
                f"Unknown TDHead input_mode={input_mode!r}"
            )

        self.hidden = hidden
        self.input_mode = input_mode
        self.use_global_residual = bool(use_global_residual)
        heads = valid_num_heads(hidden, num_heads)

        # Keep this module name compatible with phase-1 checkpoints.
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
            hidden,
            heads,
            dropout=dropout,
        )
        self.defendant_cross_attention = RationaleBiasedCrossAttention(
            hidden,
            heads,
            dropout=dropout,
        )

        self.feature_type_embedding = nn.Parameter(
            torch.empty(8, hidden)
        )
        nn.init.normal_(
            self.feature_type_embedding,
            mean=0.0,
            std=0.02,
        )

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
            reasoner_layer,
            num_layers=1,
        )

        self.feature_pool_query = nn.Parameter(
            torch.empty(hidden)
        )
        nn.init.normal_(
            self.feature_pool_query,
            mean=0.0,
            std=0.02,
        )

        self.rationale_norm = nn.LayerNorm(hidden)

        # This predicts a correction to the phase-1 global logit, not a full
        # replacement verdict.  It is intentionally initialized at zero.
        self.verdict_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.verdict_mlp[-1].weight)
        nn.init.zeros_(self.verdict_mlp[-1].bias)

        self.rationale_scale = nn.Parameter(
            torch.tensor(float(rationale_scale_init))
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen phase-1 anchor must remain deterministic during phase 2.
        if mode and not any(
            parameter.requires_grad
            for parameter in self.global_verdict_mlp.parameters()
        ):
            self.global_verdict_mlp.eval()
        return self

    @staticmethod
    def _empty_diagnostics(H_u: torch.Tensor) -> dict:
        batch_size = H_u.size(0)
        return {
            "verdict_attention_P": H_u.new_zeros(
                (batch_size, 0)
            ),
            "verdict_attention_D": H_u.new_zeros(
                (batch_size, 0)
            ),
            "feature_attention": H_u.new_zeros(
                (batch_size, 0)
            ),
            "Z_p": H_u.new_zeros(H_u.shape),
            "Z_d": H_u.new_zeros(H_u.shape),
            "rationale_delta_logit": H_u.new_zeros(
                batch_size
            ),
            "rationale_scale": torch.sigmoid(
                H_u.new_tensor(-100.0)
            ),
        }

    def _interaction_representation(
        self,
        H_u: torch.Tensor,
        stage3_out: dict,
    ) -> tuple[torch.Tensor, dict]:
        H_p = stage3_out["H_re_p"]
        H_d = stage3_out["H_re_d"]

        q_p = self.plaintiff_query(H_u)
        q_d = self.defendant_query(H_u)

        Z_p, verdict_attention_P = (
            self.plaintiff_cross_attention(
                query=q_p,
                tokens=stage3_out["top_tokens_P"],
                token_mask=stage3_out["top_mask_P"],
                rationale_scores=stage3_out[
                    "top_scores_P"
                ],
            )
        )
        Z_d, verdict_attention_D = (
            self.defendant_cross_attention(
                query=q_d,
                tokens=stage3_out["top_tokens_D"],
                token_mask=stage3_out["top_mask_D"],
                rationale_scores=stage3_out[
                    "top_scores_D"
                ],
            )
        )

        delta = H_p - H_d
        absolute_delta = torch.abs(delta)
        product = H_p * H_d

        feature_tokens = torch.stack(
            [
                H_u,
                H_p,
                H_d,
                Z_p,
                Z_d,
                delta,
                absolute_delta,
                product,
            ],
            dim=1,
        )
        feature_tokens = (
            feature_tokens
            + self.feature_type_embedding.unsqueeze(0)
        )
        reasoned_tokens = self.interaction_reasoner(
            feature_tokens
        )

        pool_logits = torch.einsum(
            "bkh,h->bk",
            reasoned_tokens,
            self.feature_pool_query,
        )
        pool_weights = torch.softmax(
            pool_logits,
            dim=-1,
        )
        z = torch.sum(
            pool_weights.unsqueeze(-1)
            * reasoned_tokens,
            dim=1,
        )
        z = self.rationale_norm(z)

        diagnostics = {
            "verdict_attention_P": verdict_attention_P,
            "verdict_attention_D": verdict_attention_D,
            "feature_attention": pool_weights,
            "Z_p": Z_p,
            "Z_d": Z_d,
        }
        return z, diagnostics

    def forward(
        self,
        stage1_out,
        stage3_out=None,
        input_mode: str | None = None,
    ):
        mode = input_mode or self.input_mode
        if mode not in {"rationale", "global_only"}:
            raise ValueError(
                f"Unknown TP input mode={mode!r}"
            )

        if (
            stage3_out is not None
            and "H_u_tp" in stage3_out
        ):
            H_u = stage3_out["H_u_tp"]
        else:
            H_u = stage1_out["H_u"]

        global_logit = self.global_verdict_mlp(
            H_u
        ).squeeze(-1)

        if mode == "global_only":
            T_logit = global_logit
            return {
                "T_logit": T_logit,
                "T_hat": torch.sigmoid(T_logit),
                "global_T_logit": global_logit,
                **self._empty_diagnostics(H_u),
            }

        if stage3_out is None:
            raise ValueError(
                "stage3_out is required for rationale mode"
            )

        z, diagnostics = self._interaction_representation(
            H_u,
            stage3_out,
        )
        rationale_delta = self.verdict_mlp(
            z
        ).squeeze(-1)

        if self.use_global_residual:
            alpha = torch.sigmoid(
                self.rationale_scale
            )
            T_logit = (
                global_logit
                + alpha * rationale_delta
            )
        else:
            alpha = global_logit.new_tensor(1.0)
            T_logit = rationale_delta

        T_hat = torch.sigmoid(T_logit)

        return {
            "T_logit": T_logit,
            "T_hat": T_hat,
            "global_T_logit": global_logit,
            "rationale_delta_logit": rationale_delta,
            "rationale_scale": alpha.detach(),
            **diagnostics,
        }
