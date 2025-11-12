# src/models/mae.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from timm.models.vision_transformer import VisionTransformer
from torch import nn

from src.models.heads import MLPHead


class MaskedAutoencoder(nn.Module):
    """Masked Autoencoder (MAE) — pretraining with ViT backbone."""

    def __init__(
        self,
        general_cfg: Dict[str, Any],
        encoder_cfg: Dict[str, Any],
        decoder_cfg: Dict[str, Any],
    ):
        super().__init__()

        self.mask_ratio = general_cfg.get("mask_ratio", 0.75)
        self.image_size = general_cfg.get("image_size", 96)
        self.patch_size = general_cfg.get("patch_size", 6)
        self.in_chans = general_cfg.get("in_chans", 3)

        vit = VisionTransformer(
            img_size=self.image_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=encoder_cfg.get("embed_dim", 384),
            depth=encoder_cfg.get("depth", 12),
            num_heads=encoder_cfg.get("num_heads", 6),
            num_classes=0,
        )

        self.encoder = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = getattr(
            self.encoder,
            "sequence_length",
            vit.patch_embed.num_patches + 1,
        )

        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=encoder_cfg.get("embed_dim", 384),
            decoder_embed_dim=decoder_cfg.get("decoder_embed_dim", 512),
            decoder_depth=decoder_cfg.get("decoder_depth", 4),
            decoder_num_heads=decoder_cfg.get("decoder_num_heads", 6),
        )

    def forward_encoder(self, images: torch.Tensor, idx_keep=None):
        return self.encoder.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)

        x_masked = utils.repeat_token(
            token=self.decoder.mask_token,
            size=(batch_size, self.sequence_length),
        )
        x_masked = utils.set_at_index(
            tokens=x_masked,
            index=idx_keep,
            value=x_decode.type_as(x_masked),
        )

        x_decoded = self.decoder.decode(x_masked)
        x_pred = utils.get_at_index(tokens=x_decoded, index=idx_mask)
        x_pred = self.decoder.predict(x_pred)

        return x_pred

    def forward(self, images: torch.Tensor):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )

        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        patches = utils.patchify(images=images, patch_size=self.patch_size)
        idx_mask_adj = torch.clamp(idx_mask - 1, min=0)
        target = utils.get_at_index(tokens=patches, index=idx_mask_adj)

        return x_pred, target


class MAEClassifier(nn.Module):
    """Classifier built on top of a pretrained MAE encoder."""

    def __init__(
        self,
        pretrained_encoder: VisionTransformer,
        num_classes: int = 10,
        head_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.encoder = pretrained_encoder
        head_cfg = head_cfg or {}

        embed_dim = head_cfg.get("emb_dim", pretrained_encoder.embed_dim)
        dropout = head_cfg.get("dropout", 0.2)
        pool_type = head_cfg.get("pool", "mean")  # or "cls"

        self.pool_type = pool_type
        self.head = MLPHead(
            input_dim=embed_dim,
            output_dim=num_classes,
            dropout=dropout,
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
