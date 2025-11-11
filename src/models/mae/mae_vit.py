"""
Masked Autoencoder Vision Transformer (MAE-ViT).

Combines an encoder and decoder for self-supervised pretraining following
the MAE architecture.
"""

from __future__ import annotations

from typing import Tuple

import torch

from src.models.mae.decoder import MAEDecoder
from src.models.mae.encoder import MAEEncoder


class MAEViT(torch.nn.Module):
    """High-level MAE model that composes MAEEncoder and MAEDecoder."""

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 6,
        emb_dim: int = 384,
        encoder_layer: int = 12,
        encoder_head: int = 6,
        decoder_layer: int = 4,
        decoder_head: int = 6,
        mask_ratio: float = 0.75,
    ) -> None:
        """
        Args:
            image_size: input image size (assumes square images).
            patch_size: spatial size of each patch.
            emb_dim: embedding dimension for both encoder and decoder.
            encoder_layer: number of transformer blocks in encoder.
            encoder_head: number of attention heads in encoder.
            decoder_layer: number of transformer blocks in decoder.
            decoder_head: number of attention heads in decoder.
            mask_ratio: fraction of patches to mask in encoder.
        """
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.encoder: MAEEncoder = MAEEncoder(
            image_size=image_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
            num_layer=encoder_layer,
            num_head=encoder_head,
            mask_ratio=mask_ratio,
        )
        self.decoder: MAEDecoder = MAEDecoder(
            image_size=image_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
            num_layer=decoder_layer,
            num_head=decoder_head,
        )

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run a forward pass through encoder and decoder.

        Args:
            img: input image tensor of shape (B, 3, H, W).

        Returns:
            predicted_img: reconstructed image tensor of shape (B, 3, H, W).
            mask: binary mask tensor of same spatial dims as predicted_img.
        """
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask
