"""
Vision Transformer (ViT) Classifier using a pretrained MAE encoder.

Wraps the encoder's transformer backbone for downstream classification tasks.
"""

from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange

from src.models.mae.encoder import MAEEncoder


class ViTClassifier(torch.nn.Module):
    """
    Simple ViT classifier that reuses a pretrained MAEEncoder's patch embedding
    and transformer backbone.

    Notes:
      - This class intentionally reuses the encoder's learnable parameters
        (cls_token, pos_embedding, patchify, transformer, layer_norm).
      - Input images are expected as (B, 3, H, W) and H/W must be compatible
        with the encoder's patch_size used to build its patchify conv.
    """

    def __init__(self, encoder: MAEEncoder, num_classes: int = 10) -> None:
        """
        Args:
            encoder: a pretrained MAEEncoder instance whose backbone will be reused.
            num_classes: number of output classes for the classifier head.
        """
        super().__init__()
        # Reuse encoder parameters/modules for consistent embeddings and transformer
        self.cls_token: torch.nn.Parameter = encoder.cls_token
        self.pos_embedding: torch.nn.Parameter = encoder.pos_embedding
        self.patchify: torch.nn.Conv2d = encoder.patchify
        self.transformer: torch.nn.Sequential = encoder.transformer
        self.layer_norm: torch.nn.LayerNorm = encoder.layer_norm

        # Classification head
        emb_dim = int(self.pos_embedding.shape[-1])
        self.head: torch.nn.Linear = torch.nn.Linear(
            in_features=emb_dim, out_features=int(num_classes)
        )

    def forward(
        self, img: torch.Tensor, return_features: Optional[bool] = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            img: input image tensor of shape (B, 3, H, W).
            return_features: if True, returns the pre-logit features along with logits
                             as a tuple (logits, features). Default False.

        Returns:
            logits: Tensor of shape (B, num_classes) or (logits, features) if
                    return_features is True.
        """
        if img.ndim != 4:
            raise ValueError(
                f"img must be a 4D tensor (B, C, H, W), got shape {img.shape}"
            )
        if img.shape[1] != 3:
            raise ValueError(
                f"img must have 3 channels in dimension 1, got {img.shape[1]}"
            )

        # Patchify: (B, C, H_p, W_p) -> (T, B, C)
        patches = self.patchify(img)
        patches = rearrange(patches, "b c h w -> (h w) b c")

        # Add positional embedding (pos_embedding shape: (T, 1, C))
        patches = patches + self.pos_embedding

        # Prepend cls token and run transformer: -> (B, T+1, C)
        batch_size = patches.shape[1]
        cls_expanded = self.cls_token.expand(-1, batch_size, -1)
        tokens = torch.cat(tensors=[cls_expanded, patches], dim=0)
        tokens = rearrange(tokens, "t b c -> b t c")

        # Transformer + LayerNorm
        features = self.layer_norm(self.transformer(tokens))

        # features back to (t, b, c) for compatibility with original API
        features = rearrange(features, "b t c -> t b c")

        # Classification head on the cls token (first token)
        logits = self.head(features[0])

        if return_features:
            return logits, features[0]

        return logits
