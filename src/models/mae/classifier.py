"""
Vision Transformer (ViT) Classifier using a pretrained MAE encoder.

Wraps the encoder's transformer backbone for downstream classification tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from src.models.mae.encoder import MAEEncoder


class ViTClassifier(nn.Module):
    """Vision Transformer classifier built from a pretrained MAE encoder."""

    def __init__(self, encoder: MAEEncoder, num_classes: int = 10) -> None:
        """
        Args:
            encoder: A pretrained MAEEncoder instance (frozen or fine-tunable).
            num_classes: Number of output classes.
        """
        super().__init__()

        # Reuse encoder components
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.norm = encoder.norm

        # Classification head
        emb_dim = self.pos_embedding.shape[-1]
        self.head = nn.Linear(emb_dim, num_classes)

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (useful for linear probing)."""
        for param in (
            self.cls_token,
            self.pos_embedding,
            self.patchify.parameters(),
            self.transformer.parameters(),
            self.norm.parameters(),
        ):
            if isinstance(param, torch.nn.Parameter):
                param.requires_grad = False
            else:
                for p in param:
                    p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters (for full fine-tuning)."""
        for param in (
            self.cls_token,
            self.pos_embedding,
            self.patchify.parameters(),
            self.transformer.parameters(),
            self.norm.parameters(),
        ):
            if isinstance(param, torch.nn.Parameter):
                param.requires_grad = True
            else:
                for p in param:
                    p.requires_grad = True

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ViT classifier.

        Args:
            imgs: Input tensor of shape (B, 3, H, W).

        Returns:
            logits: Class logits of shape (B, num_classes).
        """
        # Convert image into patch embeddings
        patches = self.patchify(imgs)  # (B, C, H', W')
        patches = rearrange(patches, "b c h w -> (h w) b c")
        patches = patches + self.pos_embedding

        # Prepend CLS token
        cls = self.cls_token.expand(-1, patches.shape[1], -1)
        tokens = torch.cat([cls, patches], dim=0)

        # Transformer encoding
        tokens = rearrange(tokens, "t b c -> b t c")
        encoded = self.transformer(tokens)
        encoded = self.norm(encoded)
        encoded = rearrange(encoded, "b t c -> t b c")

        # Classification via CLS token
        cls_repr = encoded[0]  # (B, emb_dim)
        logits = self.head(cls_repr)

        return logits
