"""
Masked Autoencoder (MAE) Encoder module.

Implements the Vision Transformer encoder with patch shuffling for self-supervised
representation learning, following the MAE architecture.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate random forward and backward indexes for patch shuffling."""
    forward = np.arange(size)
    np.random.shuffle(forward)
    backward = np.argsort(forward)
    return forward, backward


def take_indexes(sequences: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
    """Gather indexed patches across the batch dimension."""
    return torch.gather(
        sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1])
    )


class PatchShuffle(nn.Module):
    """Randomly shuffle and mask a ratio of patches."""

    def __init__(self, ratio: float) -> None:
        """
        Args:
            ratio: Fraction of patches to mask (e.g., 0.75 for 75% masking).
        """
        super().__init__()
        self.ratio = ratio

    def forward(
        self, patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: Tensor of shape (T, B, C), where
                     T = number of patches, B = batch size, C = embedding dim.

        Returns:
            patches: Tensor of visible patches after masking.
            forward_indexes: Index mapping for shuffling.
            backward_indexes: Inverse mapping for reconstruction.
        """
        T, B, _ = patches.shape
        keep_T = int(T * (1 - self.ratio))

        # Compute random permutations per batch
        perms = [random_indexes(T) for _ in range(B)]
        forward = torch.as_tensor(
            np.stack([p[0] for p in perms], axis=-1),
            dtype=torch.long,
            device=patches.device,
        )
        backward = torch.as_tensor(
            np.stack([p[1] for p in perms], axis=-1),
            dtype=torch.long,
            device=patches.device,
        )

        patches = take_indexes(patches, forward)
        patches = patches[:keep_T]

        return patches, forward, backward


class MAEEncoder(nn.Module):
    """Masked Autoencoder Vision Transformer Encoder."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 2,
        emb_dim: int = 192,
        num_layers: int = 12,
        num_heads: int = 3,
        mask_ratio: float = 0.75,
    ) -> None:
        """
        Args:
            image_size: Input image size (assumed square).
            patch_size: Size of each patch.
            emb_dim: Embedding dimension.
            num_layers: Number of Transformer blocks.
            num_heads: Number of attention heads.
            mask_ratio: Fraction of patches to mask.
        """
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros((image_size // patch_size) ** 2, 1, emb_dim)
        )
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.transformer = nn.Sequential(
            *[Block(emb_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(emb_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize trainable parameters."""
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MAE encoder.

        Args:
            imgs: Input tensor of shape (B, 3, H, W).

        Returns:
            encoded: Encoded patch features of shape (T_vis+1, B, C).
            backward_indexes: Indexes for reconstructing original patch order.
        """
        patches = self.patchify(imgs)  # (B, C, H', W')
        patches = rearrange(patches, "b c h w -> (h w) b c")
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        # Add CLS token
        cls = self.cls_token.expand(-1, patches.shape[1], -1)
        patches = torch.cat([cls, patches], dim=0)

        patches = rearrange(patches, "t b c -> b t c")
        encoded = self.transformer(patches)
        encoded = self.norm(encoded)
        encoded = rearrange(encoded, "b t c -> t b c")

        return encoded, backward_indexes
