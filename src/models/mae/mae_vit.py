"""
Masked Autoencoder Vision Transformer (MAE-ViT).

Combines an encoder and decoder for self-supervised pretraining following
the MAE architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.mae.decoder import MAEDecoder
from src.models.mae.encoder import MAEEncoder


class MAEViT(nn.Module):
    """Full Masked Autoencoder Vision Transformer (MAE-ViT) model."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 2,
        emb_dim: int = 192,
        encoder_layers: int = 12,
        encoder_heads: int = 3,
        decoder_layers: int = 4,
        decoder_heads: int = 3,
        mask_ratio: float = 0.75,
    ) -> None:
        """
        Args:
            image_size: Input image size (assumed square).
            patch_size: Size of each patch.
            emb_dim: Embedding dimension for encoder and decoder interface.
            encoder_layers: Number of Transformer blocks in the encoder.
            encoder_heads: Number of attention heads in the encoder.
            decoder_layers: Number of Transformer blocks in the decoder.
            decoder_heads: Number of attention heads in the decoder.
            mask_ratio: Fraction of patches to mask during pretraining.
        """
        super().__init__()

        self.encoder = MAEEncoder(
            image_size=image_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            mask_ratio=mask_ratio,
        )

        self.decoder = MAEDecoder(
            image_size=image_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
            decoder_dim=emb_dim * 2,  # default expansion for reconstruction
            num_layers=decoder_layers,
            num_heads=decoder_heads,
        )

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the full MAE-ViT model.

        Args:
            imgs: Input tensor of shape (B, 3, H, W).

        Returns:
            pred: Reconstructed image patches tensor of shape
                  (B, num_patches, 3, patch_size, patch_size).
            mask: Binary mask tensor indicating which patches were reconstructed.
        """
        encoded, backward_indexes = self.encoder(imgs)
        predicted_img, mask = self.decoder(encoded, backward_indexes)
        return predicted_img, mask