"""Reusable attention blocks for the CAER-MTL architecture."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def valid_num_heads(hidden_size: int, requested: int) -> int:
    """Return the largest valid head count not exceeding ``requested``."""

    requested = max(1, min(int(requested), int(hidden_size)))
    for heads in range(requested, 0, -1):
        if hidden_size % heads == 0:
            return heads
    return 1


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Softmax that returns zero for rows containing no valid item."""

    mask = mask.bool()
    safe_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    no_valid = ~mask.any(dim=dim, keepdim=True)
    safe_logits = torch.where(no_valid, torch.zeros_like(safe_logits), safe_logits)

    weights = torch.softmax(safe_logits, dim=dim)
    weights = weights * mask.to(weights.dtype)
    normalizer = weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    weights = torch.where(no_valid, torch.zeros_like(weights), weights / normalizer)
    return weights


class AdditiveAttentionPool(nn.Module):
    """Context-conditioned additive attention pooling."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.item_proj = nn.Linear(hidden_size, hidden_size)
        self.context_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        items: torch.Tensor,
        mask: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            items: ``[B, K, H]``
            mask: ``[B, K]``
            context: optional ``[B, H]``
        """

        scores_hidden = self.item_proj(items)
        if context is not None:
            scores_hidden = scores_hidden + self.context_proj(context).unsqueeze(1)

        scores = self.score(torch.tanh(scores_hidden)).squeeze(-1)
        weights = masked_softmax(scores, mask, dim=-1)
        pooled = torch.sum(self.dropout(weights).unsqueeze(-1) * items, dim=1)
        return pooled, weights


class RationaleBiasedCrossAttention(nn.Module):
    """
    Multi-head query-to-claims attention with an additive rationale bias.

    The bias makes the TP head prefer claims that Stage 2 considers accepted,
    while content-based query/key similarity can still override a weak RE score.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_heads = valid_num_heads(hidden_size, num_heads)
        self.head_dim = hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Positive by construction through softplus in forward.
        self.rationale_bias_raw = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        rationale_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: ``[B, H]``
            tokens: ``[B, K, H]``
            token_mask: ``[B, K]``
            rationale_scores: ``[B, K]`` in ``[0, 1]``
        Returns:
            attended: ``[B, H]``
            mean_attention: ``[B, K]``
        """

        batch_size, num_tokens, hidden_size = tokens.shape

        q = self.q_proj(query).view(batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(tokens).view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(tokens).view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

        logits = torch.einsum("bhd,bhkd->bhk", q, k) * self.scale

        eps = 1e-5
        rationale_scores = rationale_scores.clamp(eps, 1.0 - eps)
        rationale_logit = torch.logit(rationale_scores)
        bias_strength = F.softplus(self.rationale_bias_raw)
        logits = logits + bias_strength * rationale_logit.unsqueeze(1)

        expanded_mask = token_mask.bool().unsqueeze(1).expand_as(logits)
        weights = masked_softmax(logits, expanded_mask, dim=-1)
        weights = self.dropout(weights)

        attended = torch.einsum("bhk,bhkd->bhd", weights, v)
        attended = attended.reshape(batch_size, hidden_size)
        attended = self.out_proj(attended)

        valid_row = token_mask.any(dim=-1, keepdim=True)
        attended = torch.where(valid_row, attended, torch.zeros_like(attended))

        return attended, weights.mean(dim=1)
