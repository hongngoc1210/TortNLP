"""Exact-identity task adapters for the final MTL architecture.

The adapter is initialized as an exact identity mapping.  This matters when a
phase-1 checkpoint trained without adapters is loaded into the phase-2 model:
adding the adapter must not immediately change the RE representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IdentityAdapter(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class TaskAdapter(nn.Module):
    """Small bottleneck residual adapter.

    ``up`` is zero-initialized, therefore the initial forward pass is exactly
    ``hidden_states`` rather than merely approximately identical.
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        hidden_size = int(hidden_size)
        bottleneck_size = max(
            1,
            min(int(bottleneck_size), hidden_size),
        )

        self.pre_norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(bottleneck_size, hidden_size)

        # Exact identity at initialization.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        update = self.pre_norm(hidden_states)
        update = self.down(update)
        update = self.activation(update)
        update = self.dropout(update)
        update = self.up(update)
        return hidden_states + update


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
