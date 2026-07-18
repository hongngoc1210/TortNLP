"""Shared Hugging Face encoder used by all four stages."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel


class SharedEncoder(nn.Module):
    """
    A thin wrapper around a pretrained Transformer.

    The old project only exposed a CLS vector.  The new Stage 1 also needs
    token-level representations of the undisputed facts so each claim can
    attend to the fact tokens that are most relevant to it.  Returning both
    values lets us keep the original ``data_utils`` unchanged.
    """

    def __init__(
        self,
        model_name: str = "sbintuitions/modernbert-ja-310m",
        enable_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        if (
            enable_gradient_checkpointing
            and hasattr(self.encoder, "gradient_checkpointing_enable")
        ):
            self.encoder.gradient_checkpointing_enable()

        self.hidden_size = int(self.encoder.config.hidden_size)

    @staticmethod
    def masked_mean(
        token_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool valid tokens; safe for an all-padding row."""

        mask = attention_mask.to(token_states.dtype).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (token_states * mask).sum(dim=1) / denominator

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_tokens: bool = False,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        token_states = outputs.last_hidden_state

        # ModernBERT supports a CLS-like first token, matching the old model.
        pooled = token_states[:, 0]

        if return_tokens:
            return {
                "pooled": pooled,
                "tokens": token_states,
                "attention_mask": attention_mask.bool(),
            }

        return pooled
