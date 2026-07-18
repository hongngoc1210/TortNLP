"""Small task-specific residual adapters for MTL ablations."""

from __future__ import annotations

import torch
import torch.nn as nn


class IdentityAdapter(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class TaskAdapter(nn.Module):
    """A light bottleneck adapter with a residual connection.

    It gives RE and TP a small amount of task-specific capacity while keeping
    the expensive Transformer encoder shared.
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        bottleneck_size = max(1, min(int(bottleneck_size), int(hidden_size)))
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

        # Start close to identity so that adding adapters does not immediately
        # destroy the pretrained/shared representation.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        update = self.up(self.dropout(self.activation(self.down(hidden_states))))
        return self.norm(hidden_states + update)


def build_adapter(
    hidden_size: int,
    enabled: bool,
    bottleneck_size: int = 128,
    dropout: float = 0.1,
) -> nn.Module:
    if not enabled:
        return IdentityAdapter()
    return TaskAdapter(
        hidden_size=hidden_size,
        bottleneck_size=bottleneck_size,
        dropout=dropout,
    )
