from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn


class LinearProbe(nn.Module):
    """
    Linear probe classifier on top of a frozen Vision Transformer encoder.

    This module is used to evaluate the quality of representations learned
    during self-supervised pretraining (e.g., MAE). The encoder is kept frozen,
    and only a single linear layer is trained.

    Args:
        encoder (VisionTransformer): Pretrained encoder backbone (frozen).
        num_classes (int): Number of target classes for classification.
        head_cfg (dict, optional): Config for head with optional keys:
            - "emb_dim": embedding dimension of encoder output (default: encoder.embed_dim)
            - "bias": whether to include bias in the linear layer (default: True)
            - "pool": pooling strategy ('cls' or 'mean', default: 'mean')
    """

    def __init__(
        self,
        encoder: VisionTransformer,
        num_classes: int = 10,
        head_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.head_cfg = head_cfg or {}

        # Freeze encoder (no gradients)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.embed_dim = self.head_cfg.get("emb_dim", encoder.embed_dim)
        self.bias = self.head_cfg.get("bias", True)
        self.pool_type = self.head_cfg.get("pool", "mean")

        # Simple linear head (strictly linear)
        self.head = nn.Linear(self.embed_dim, num_classes, bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        - Extract encoder features (frozen)
        - Apply pooling (mean or CLS token)
        - Apply linear classification head
        """
        feats = self.encoder.forward_features(x)  # shape: [B, N, D]

        if self.pool_type == "cls" and hasattr(self.encoder, "cls_token"):
            # CLS token is typically at index 0
            pooled = feats[:, 0]
        else:
            # Default: average across patch tokens
            pooled = feats.mean(dim=1)

        logits = self.head(pooled)
        return logits
