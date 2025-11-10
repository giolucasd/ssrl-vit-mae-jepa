"""
Masked Autoencoder Vision Transformer (MAE-ViT).

Combines an encoder and decoder for self-supervised pretraining following
the MAE architecture.
"""

from __future__ import annotations

import torch

from src.models.mae.decoder import MAEDecoder
from src.models.mae.encoder import MAEEncoder


class MAEViT(torch.nn.Module):
    def __init__(
        self,
        image_size=96,
        patch_size=8,
        emb_dim=192,
        encoder_layer=12,
        encoder_head=3,
        decoder_layer=4,
        decoder_head=3,
        mask_ratio=0.75,
    ) -> None:
        super().__init__()

        self.encoder = MAEEncoder(
            image_size,
            patch_size,
            emb_dim,
            encoder_layer,
            encoder_head,
            mask_ratio,
        )
        self.decoder = MAEDecoder(
            image_size,
            patch_size,
            emb_dim,
            decoder_layer,
            decoder_head,
        )

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask
