from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn


class SimpleHead(torch.nn.Module):
    """Simple classification head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.classification = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classification(x)


class ViTClassifier(nn.Module):
    """Classifier built on top of a pretrained ViT encoder."""

    def __init__(
        self,
        pretrained_encoder: VisionTransformer,
        num_classes: int = 10,
        head_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.encoder = pretrained_encoder
        head_cfg = head_cfg or {}

        embed_dim = head_cfg.get("embed_dim", pretrained_encoder.embed_dim)
        pool_type = head_cfg.get("pool", "cls")  # or "mean"

        self.pool_type = pool_type
        self.head = SimpleHead(
            input_dim=embed_dim,
            output_dim=num_classes,
        )

    def forward(self, x: torch.Tensor):
        feats = self.encoder.forward_features(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]

        if self.pool_type == "cls":
            pooled = feats[:, 0]
        else:
            pooled = feats.mean(dim=1)

        return self.head(pooled)
