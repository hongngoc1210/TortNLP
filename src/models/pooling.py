"""Stage 3: rationale-guided evidence aggregation with ablation switches."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .adapters import build_adapter


class RationalePooling(nn.Module):
    """Adaptive rationale attention with a learned fallback.

    Supported pooling modes:
      - ``rationale``: current rationale-guided path.
      - ``fallback_only``: ignores RE probabilities and selects/pools claims
        using only content and global context.

    ``fallback_only`` is the clean no-rationale ablation.  It does not use a
    constant fake rationale inside the rationale scorer.
    """

    def __init__(
        self,
        hidden: int,
        topk_claims: int = 5,
        dropout: float = 0.1,
        use_task_adapter: bool = False,
        adapter_bottleneck: int = 128,
        adapter_dropout: float = 0.1,
        detach_rationale_for_tp: bool = False,
    ) -> None:
        super().__init__()

        self.hidden = int(hidden)
        self.topk_claims = max(1, int(topk_claims))
        self.detach_rationale_for_tp = bool(detach_rationale_for_tp)

        self.tp_adapter = build_adapter(
            hidden_size=hidden,
            enabled=use_task_adapter,
            bottleneck_size=adapter_bottleneck,
            dropout=adapter_dropout,
        )

        self.claim_proj = nn.Linear(hidden, hidden)
        self.context_proj = nn.Linear(hidden, hidden, bias=False)
        self.score_proj = nn.Linear(1, hidden, bias=False)
        self.rationale_score = nn.Linear(hidden, 1, bias=False)

        self.fallback_claim_proj = nn.Linear(hidden, hidden)
        self.fallback_context_proj = nn.Linear(hidden, hidden, bias=False)
        self.fallback_score = nn.Linear(hidden, 1, bias=False)

        gate_hidden = max(32, hidden // 4)
        self.mix_gate = nn.Sequential(
            nn.Linear(hidden * 3, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

        self.output_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def _empty_topk(
        self,
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            reference.new_zeros((self.topk_claims, self.hidden)),
            torch.zeros(self.topk_claims, dtype=torch.bool, device=reference.device),
            reference.new_full((self.topk_claims,), 0.5),
            torch.full(
                (self.topk_claims,), -1, dtype=torch.long, device=reference.device
            ),
        )

    def pool_side(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        sample_map: torch.Tensor,
        global_context: torch.Tensor,
        batch_size: int,
        pooling_mode: str = "rationale",
    ) -> Dict[str, torch.Tensor]:
        if pooling_mode not in {"rationale", "fallback_only"}:
            raise ValueError(f"Unknown pooling_mode={pooling_mode!r}")

        pooled_cases = []
        mix_values = []
        top_tokens = []
        top_masks = []
        top_scores = []
        top_indices = []

        flat_attention = torch.zeros_like(r)
        flat_fallback_attention = torch.zeros_like(r)

        for case_id in range(batch_size):
            idx = torch.nonzero(sample_map == case_id, as_tuple=True)[0]

            if idx.numel() == 0:
                pooled_cases.append(global_context.new_zeros(self.hidden))
                mix_values.append(global_context.new_zeros(1))
                empty = self._empty_topk(global_context)
                top_tokens.append(empty[0])
                top_masks.append(empty[1])
                top_scores.append(empty[2])
                top_indices.append(empty[3])
                continue

            h_case = h[idx]
            r_case = torch.nan_to_num(r[idx], nan=0.5, posinf=1.0, neginf=0.0)
            r_case = r_case.clamp(1e-5, 1.0 - 1e-5)
            context = global_context[case_id]

            fallback_hidden = (
                self.fallback_claim_proj(h_case)
                + self.fallback_context_proj(context).unsqueeze(0)
            )
            fallback_logits = self.fallback_score(
                torch.tanh(fallback_hidden)
            ).squeeze(-1)
            fallback_weights = torch.softmax(fallback_logits, dim=0)
            fallback_pool = torch.sum(
                self.dropout(fallback_weights).unsqueeze(-1) * h_case, dim=0
            )

            if pooling_mode == "fallback_only":
                rationale_weights = torch.zeros_like(fallback_weights)
                rationale_pool = torch.zeros_like(fallback_pool)
                mixing = context.new_zeros(1)
                pooled = self.output_norm(fallback_pool)
                selection_weights = fallback_weights
                selected_scores = torch.full_like(r_case, 0.5)
            else:
                rationale_hidden = (
                    self.claim_proj(h_case)
                    + self.context_proj(context).unsqueeze(0)
                    + self.score_proj(r_case.unsqueeze(-1))
                )
                rationale_logits = self.rationale_score(
                    torch.tanh(rationale_hidden)
                ).squeeze(-1)
                rationale_logits = rationale_logits + torch.logit(r_case)
                rationale_weights = torch.softmax(rationale_logits, dim=0)
                rationale_pool = torch.sum(
                    self.dropout(rationale_weights).unsqueeze(-1) * h_case, dim=0
                )

                gate_input = torch.cat(
                    [rationale_pool, fallback_pool, context], dim=-1
                )
                mixing = self.mix_gate(gate_input.unsqueeze(0)).squeeze(0)
                pooled = mixing * rationale_pool + (1.0 - mixing) * fallback_pool
                pooled = self.output_norm(pooled)
                selection_weights = rationale_weights
                selected_scores = r_case

            flat_attention = flat_attention.index_copy(0, idx, rationale_weights)
            flat_fallback_attention = flat_fallback_attention.index_copy(
                0, idx, fallback_weights
            )

            k = min(self.topk_claims, idx.numel())
            selected_local = torch.topk(selection_weights, k=k, dim=0).indices
            selected_global = idx[selected_local]

            case_top_tokens = h.new_zeros((self.topk_claims, self.hidden))
            case_top_mask = torch.zeros(
                self.topk_claims, dtype=torch.bool, device=h.device
            )
            case_top_scores = r.new_full((self.topk_claims,), 0.5)
            case_top_indices = torch.full(
                (self.topk_claims,), -1, dtype=torch.long, device=h.device
            )

            case_top_tokens[:k] = h_case[selected_local]
            case_top_mask[:k] = True
            case_top_scores[:k] = selected_scores[selected_local]
            case_top_indices[:k] = selected_global

            pooled_cases.append(pooled)
            mix_values.append(mixing)
            top_tokens.append(case_top_tokens)
            top_masks.append(case_top_mask)
            top_scores.append(case_top_scores)
            top_indices.append(case_top_indices)

        return {
            "pooled": torch.stack(pooled_cases, dim=0),
            "mix_gate": torch.stack(mix_values, dim=0),
            "attention": flat_attention,
            "fallback_attention": flat_fallback_attention,
            "top_tokens": torch.stack(top_tokens, dim=0),
            "top_mask": torch.stack(top_masks, dim=0),
            "top_scores": torch.stack(top_scores, dim=0),
            "top_indices": torch.stack(top_indices, dim=0),
        }

    def forward(
        self,
        stage1_out,
        stage2_out,
        batch,
        pooling_mode: str = "rationale",
    ):
        batch_size = batch["U_input_ids"].size(0)

        # TP-specific representations.  The RE head receives its own adapter in
        # Stage 2, so the two tasks can specialize without duplicating encoder.
        hP = self.tp_adapter(stage1_out["hP_cond"])
        hD = self.tp_adapter(stage1_out["hD_cond"])
        H_u_tp = self.tp_adapter(stage1_out["H_u"])

        rP = stage2_out.get("rP_for_pool", stage2_out["rP_hat"])
        rD = stage2_out.get("rD_for_pool", stage2_out["rD_hat"])
        if self.detach_rationale_for_tp:
            rP = rP.detach()
            rD = rD.detach()

        side_P = self.pool_side(
            h=hP,
            r=rP,
            sample_map=batch["sample_map_P"].long(),
            global_context=H_u_tp,
            batch_size=batch_size,
            pooling_mode=pooling_mode,
        )
        side_D = self.pool_side(
            h=hD,
            r=rD,
            sample_map=batch["sample_map_D"].long(),
            global_context=H_u_tp,
            batch_size=batch_size,
            pooling_mode=pooling_mode,
        )

        return {
            "H_u_tp": H_u_tp,
            "H_re_p": side_P["pooled"],
            "H_re_d": side_D["pooled"],
            "mix_gate_P": side_P["mix_gate"],
            "mix_gate_D": side_D["mix_gate"],
            "claim_attention_P": side_P["attention"],
            "claim_attention_D": side_D["attention"],
            "fallback_attention_P": side_P["fallback_attention"],
            "fallback_attention_D": side_D["fallback_attention"],
            "top_tokens_P": side_P["top_tokens"],
            "top_tokens_D": side_D["top_tokens"],
            "top_mask_P": side_P["top_mask"],
            "top_mask_D": side_D["top_mask"],
            "top_scores_P": side_P["top_scores"],
            "top_scores_D": side_D["top_scores"],
            "top_indices_P": side_P["top_indices"],
            "top_indices_D": side_D["top_indices"],
        }
