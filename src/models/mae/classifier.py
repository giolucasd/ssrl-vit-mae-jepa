"""
Vision Transformer (ViT) Classifier using a pretrained MAE encoder.

Wraps the encoder's transformer backbone for downstream classification tasks.
"""

from __future__ import annotations

import torch
from einops import rearrange

from src.models.mae.encoder import MAEEncoder


class ViTClassifier(torch.nn.Module):
    def __init__(self, encoder: MAEEncoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, "b c h w -> (h w) b c")
        patches = patches + self.pos_embedding
        patches = torch.cat(
            [self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0
        )
        patches = rearrange(patches, "t b c -> b t c")
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, "b t c -> t b c")
        logits = self.head(features[0])
        return logits
