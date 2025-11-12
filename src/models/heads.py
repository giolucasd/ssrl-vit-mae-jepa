from __future__ import annotations

import torch


class MLPHead(torch.nn.Module):
    """Compact MLP head with GELU, LayerNorm, dropout, and light residual connection."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.fc1 = torch.nn.Linear(input_dim, input_dim)
        self.act = torch.nn.GELU()
        self.norm = torch.nn.LayerNorm(input_dim)
        self.drop = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.norm(h)
        h = self.drop(h)
        h = h + x  # residual connection
        out = self.fc2(h)
        return out
