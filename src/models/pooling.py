from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .adapters import build_adapter


class RationalePooling(nn.Module):
    """Adaptive rationale attention with a learned content fallback.

    Supported pooling modes
    -----------------------
    ``rationale``
        Pool claims with predicted/gold rationale probabilities and blend this
        representation with a content-only fallback representation.

    ``fallback_only``
        Ignore rationale probabilities when pooling.  This is the clean
        no-rationale ablation.
    """

    VALID_POOLING_MODES = {"rationale", "fallback_only"}

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
            hidden_size=self.hidden,
            enabled=use_task_adapter,
            bottleneck_size=adapter_bottleneck,
            dropout=adapter_dropout,
        )

        # Rationale-guided scorer.
        self.claim_proj = nn.Linear(self.hidden, self.hidden)
        self.context_proj = nn.Linear(
            self.hidden,
            self.hidden,
            bias=False,
        )
        self.score_proj = nn.Linear(1, self.hidden, bias=False)
        self.rationale_score = nn.Linear(
            self.hidden,
            1,
            bias=False,
        )

        # Content-only fallback scorer.
        self.fallback_claim_proj = nn.Linear(
            self.hidden,
            self.hidden,
        )
        self.fallback_context_proj = nn.Linear(
            self.hidden,
            self.hidden,
            bias=False,
        )
        self.fallback_score = nn.Linear(
            self.hidden,
            1,
            bias=False,
        )

        gate_hidden = max(32, self.hidden // 4)
        self.mix_gate = nn.Sequential(
            nn.Linear(self.hidden * 3, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

        self.output_norm = nn.LayerNorm(self.hidden)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Validation and tensor helpers
    # ------------------------------------------------------------------

    def _empty_topk(
        self,
        reference: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return a correctly shaped empty top-k result."""
        return (
            reference.new_zeros(
                (self.topk_claims, self.hidden)
            ),
            torch.zeros(
                self.topk_claims,
                dtype=torch.bool,
                device=reference.device,
            ),
            reference.new_full(
                (self.topk_claims,),
                0.5,
            ),
            torch.full(
                (self.topk_claims,),
                -1,
                dtype=torch.long,
                device=reference.device,
            ),
        )

    @staticmethod
    def _safe_index_copy(
        destination: torch.Tensor,
        index: torch.Tensor,
        source: torch.Tensor,
        dim: int = 0,
    ) -> torch.Tensor:
        """AMP-safe ``index_copy`` with explicit dtype/device alignment."""
        index = index.reshape(-1).to(
            device=destination.device,
            dtype=torch.long,
        )
        source = source.to(
            device=destination.device,
            dtype=destination.dtype,
        )

        if source.ndim == 0:
            source = source.reshape(1)

        if source.size(dim) != index.numel():
            raise RuntimeError(
                "index_copy shape mismatch: "
                f"destination={tuple(destination.shape)}, "
                f"index={tuple(index.shape)}, "
                f"source={tuple(source.shape)}, "
                f"dim={dim}"
            )

        return destination.index_copy(
            dim=dim,
            index=index,
            source=source,
        )

    def _validate_side_inputs(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        sample_map: torch.Tensor,
        global_context: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if h.ndim != 2:
            raise ValueError(
                "Claim representations must have shape [num_claims, hidden], "
                f"received {tuple(h.shape)}."
            )
        if h.size(-1) != self.hidden:
            raise ValueError(
                f"Expected hidden size {self.hidden}, received {h.size(-1)}."
            )
        if global_context.ndim != 2:
            raise ValueError(
                "global_context must have shape [batch_size, hidden], "
                f"received {tuple(global_context.shape)}."
            )
        if global_context.size(0) != batch_size:
            raise ValueError(
                "batch_size does not match global_context: "
                f"batch_size={batch_size}, "
                f"global_context.size(0)={global_context.size(0)}."
            )
        if global_context.size(-1) != self.hidden:
            raise ValueError(
                f"Expected global hidden size {self.hidden}, "
                f"received {global_context.size(-1)}."
            )

        r = r.reshape(-1)
        sample_map = sample_map.reshape(-1).to(
            device=h.device,
            dtype=torch.long,
        )

        num_claims = int(h.size(0))
        if r.numel() != num_claims:
            raise ValueError(
                "Rationale-score count does not match claim count: "
                f"scores={r.numel()}, claims={num_claims}."
            )
        if sample_map.numel() != num_claims:
            raise ValueError(
                "sample_map count does not match claim count: "
                f"sample_map={sample_map.numel()}, claims={num_claims}."
            )

        if sample_map.numel() > 0:
            min_case = int(sample_map.min().item())
            max_case = int(sample_map.max().item())
            if min_case < 0 or max_case >= batch_size:
                raise ValueError(
                    "sample_map contains an invalid case index: "
                    f"min={min_case}, max={max_case}, "
                    f"batch_size={batch_size}."
                )

        return r, sample_map

    @staticmethod
    def _normalise_rationale_scores(
        scores: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Sanitise probabilities and align them with AMP compute dtype."""
        scores = scores.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        scores = torch.nan_to_num(
            scores,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        )
        return scores.clamp(1e-5, 1.0 - 1e-5)

    @staticmethod
    def _pad_topk_tensor(
        selected: torch.Tensor,
        size: int,
        fill_value: float,
    ) -> torch.Tensor:
        """Pad first dimension without severing gradients from ``selected``."""
        missing = size - int(selected.size(0))
        if missing <= 0:
            return selected

        padding_shape = (missing, *selected.shape[1:])
        padding = selected.new_full(
            padding_shape,
            fill_value,
        )
        return torch.cat([selected, padding], dim=0)

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def pool_side(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        sample_map: torch.Tensor,
        global_context: torch.Tensor,
        batch_size: int,
        pooling_mode: str = "rationale",
    ) -> Dict[str, torch.Tensor]:
        if pooling_mode not in self.VALID_POOLING_MODES:
            raise ValueError(
                f"Unknown pooling_mode={pooling_mode!r}. "
                f"Expected one of {sorted(self.VALID_POOLING_MODES)}."
            )

        r, sample_map = self._validate_side_inputs(
            h=h,
            r=r,
            sample_map=sample_map,
            global_context=global_context,
            batch_size=batch_size,
        )

        pooled_cases = []
        mix_values = []
        top_tokens = []
        top_masks = []
        top_scores = []
        top_indices = []

        # Attention is consumed together with TP hidden states, so use ``h`` as
        # the dtype/device reference.  This avoids Float/Half index_copy errors
        # when RE probabilities remain FP32 under AMP.
        flat_attention = h.new_zeros((h.size(0),))
        flat_fallback_attention = h.new_zeros((h.size(0),))

        for case_id in range(batch_size):
            idx = torch.nonzero(
                sample_map == case_id,
                as_tuple=True,
            )[0]

            context = global_context[case_id]

            if idx.numel() == 0:
                # Keep dtype/device consistent with TP representations.
                pooled_cases.append(context.new_zeros(self.hidden))
                mix_values.append(context.new_zeros(1))

                empty = self._empty_topk(context)
                top_tokens.append(empty[0])
                top_masks.append(empty[1])
                top_scores.append(empty[2])
                top_indices.append(empty[3])
                continue

            h_case = h.index_select(0, idx)
            r_case = self._normalise_rationale_scores(
                r.index_select(0, idx),
                reference=h_case,
            )

            # ----------------------------------------------------------
            # Content-only fallback attention
            # ----------------------------------------------------------
            fallback_hidden = (
                self.fallback_claim_proj(h_case)
                + self.fallback_context_proj(context).unsqueeze(0)
            )
            fallback_logits = self.fallback_score(
                torch.tanh(fallback_hidden)
            ).squeeze(-1)
            fallback_weights = torch.softmax(
                fallback_logits,
                dim=0,
            )
            fallback_pool = torch.sum(
                self.dropout(fallback_weights).unsqueeze(-1)
                * h_case,
                dim=0,
            )

            # ----------------------------------------------------------
            # Rationale-guided attention or clean no-rationale path
            # ----------------------------------------------------------
            if pooling_mode == "fallback_only":
                rationale_weights = torch.zeros_like(
                    fallback_weights
                )
                rationale_pool = torch.zeros_like(
                    fallback_pool
                )
                mixing = context.new_zeros(1)
                pooled = self.output_norm(fallback_pool)
                selection_weights = fallback_weights
                selected_scores = torch.full_like(
                    r_case,
                    0.5,
                )
            else:
                rationale_hidden = (
                    self.claim_proj(h_case)
                    + self.context_proj(context).unsqueeze(0)
                    + self.score_proj(r_case.unsqueeze(-1))
                )
                rationale_logits = self.rationale_score(
                    torch.tanh(rationale_hidden)
                ).squeeze(-1)

                # Explicit cast protects against promotion when probabilities
                # originate from an FP32 sigmoid while the scorer runs FP16.
                prior_logits = torch.logit(r_case).to(
                    dtype=rationale_logits.dtype,
                    device=rationale_logits.device,
                )
                rationale_logits = rationale_logits + prior_logits
                rationale_weights = torch.softmax(
                    rationale_logits,
                    dim=0,
                )
                rationale_pool = torch.sum(
                    self.dropout(rationale_weights).unsqueeze(-1)
                    * h_case,
                    dim=0,
                )

                gate_input = torch.cat(
                    [
                        rationale_pool,
                        fallback_pool,
                        context.to(dtype=rationale_pool.dtype),
                    ],
                    dim=-1,
                )
                mixing = self.mix_gate(
                    gate_input.unsqueeze(0)
                ).squeeze(0)
                pooled = (
                    mixing * rationale_pool
                    + (1.0 - mixing) * fallback_pool
                )
                pooled = self.output_norm(pooled)
                selection_weights = rationale_weights
                selected_scores = r_case

            # ----------------------------------------------------------
            # Store per-claim diagnostics with AMP-safe indexed writes
            # ----------------------------------------------------------
            flat_attention = self._safe_index_copy(
                destination=flat_attention,
                index=idx,
                source=rationale_weights.reshape(-1),
            )
            flat_fallback_attention = self._safe_index_copy(
                destination=flat_fallback_attention,
                index=idx,
                source=fallback_weights.reshape(-1),
            )

            # ----------------------------------------------------------
            # Top-k claim diagnostics
            # ----------------------------------------------------------
            k = min(
                self.topk_claims,
                int(idx.numel()),
            )
            selected_local = torch.topk(
                selection_weights,
                k=k,
                dim=0,
            ).indices
            selected_global = idx.index_select(
                0,
                selected_local,
            )

            selected_tokens = h_case.index_select(
                0,
                selected_local,
            )
            case_top_tokens = self._pad_topk_tensor(
                selected_tokens,
                self.topk_claims,
                fill_value=0.0,
            )

            selected_top_scores = selected_scores.index_select(
                0,
                selected_local,
            ).to(
                device=h.device,
                dtype=h.dtype,
            )
            case_top_scores = self._pad_topk_tensor(
                selected_top_scores,
                self.topk_claims,
                fill_value=0.5,
            )

            case_top_mask = torch.zeros(
                self.topk_claims,
                dtype=torch.bool,
                device=h.device,
            )
            case_top_mask[:k] = True

            case_top_indices = torch.full(
                (self.topk_claims,),
                -1,
                dtype=torch.long,
                device=h.device,
            )
            case_top_indices[:k] = selected_global.to(
                device=h.device,
                dtype=torch.long,
            )

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
        stage1_out: Dict[str, torch.Tensor],
        stage2_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        pooling_mode: str = "rationale",
    ) -> Dict[str, torch.Tensor]:
        if stage2_out is None:
            raise ValueError(
                "RationalePooling requires Stage 2 outputs. "
                "Use the Stage 4 global-only path for TP-only ablations."
            )

        batch_size = int(batch["U_input_ids"].size(0))

        # TP-specific representations.  The RE head can use its own adapter in
        # Stage 2, allowing task specialisation without duplicating the encoder.
        hP = self.tp_adapter(stage1_out["hP_cond"])
        hD = self.tp_adapter(stage1_out["hD_cond"])
        H_u_tp = self.tp_adapter(stage1_out["H_u"])

        rP = stage2_out.get(
            "rP_for_pool",
            stage2_out["rP_hat"],
        )
        rD = stage2_out.get(
            "rD_for_pool",
            stage2_out["rD_hat"],
        )

        if self.detach_rationale_for_tp:
            rP = rP.detach()
            rD = rD.detach()

        side_P = self.pool_side(
            h=hP,
            r=rP,
            sample_map=batch["sample_map_P"],
            global_context=H_u_tp,
            batch_size=batch_size,
            pooling_mode=pooling_mode,
        )
        side_D = self.pool_side(
            h=hD,
            r=rD,
            sample_map=batch["sample_map_D"],
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