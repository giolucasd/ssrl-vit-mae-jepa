"""
Masked Autoencoder (MAE) Encoder module.

Implements the Vision Transformer encoder with patch shuffling for self-supervised
representation learning, following the MAE architecture.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from einops import rearrange, repeat
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random permutation of range(size) and its inverse mapping.

    Args:
        size: number of indices to permute.

    Returns:
        forward_indexes: a 1D numpy array of length `size` representing a random
            permutation of [0, 1, ..., size-1].
        backward_indexes: a 1D numpy array of length `size` such that
            forward_indexes[backward_indexes] == np.arange(size).
    """
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(patches: torch.Tensor, indexes: torch.LongTensor) -> torch.Tensor:
    """
    Reorder `patches` along the first dimension according to `indexes`.

    Args:
        patches: Tensor of shape (T, B, C) where T is number of patches,
                B is batch size, C is embedding dimension.
        indexes: LongTensor with shape (T, B) containing indices in [0, T).

    Returns:
        gathered: Tensor with shape (T, B, C) where gathered[t, b, :] =
            patches[indexes[t, b], b, :].
    """
    # repeat indexes along channel dimension so torch.gather can be used.
    return torch.gather(
        patches, 0, repeat(indexes, "t b -> t b c", c=patches.shape[-1])
    )


class PatchShuffle(torch.nn.Module):
    """
    Module that randomly shuffles patches independently per batch item and
    removes a fraction of the patches (masking).

    The forward pass returns the remaining patches (first dimension is time/patch),
    as well as the forward and backward index mappings so the original order can
    be restored by the decoder.
    """

    def __init__(self, ratio: float) -> None:
        """
        Args:
            ratio: fraction of patches to mask/remove (0.0 <= ratio < 1.0).
        """
        super().__init__()
        if not (0.0 <= ratio < 1.0):
            raise ValueError("ratio must satisfy 0.0 <= ratio < 1.0")
        self.ratio = float(ratio)

    def forward(
        self, patches: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
        """
        Args:
            patches: Tensor of shape (T, B, C) where T is number of patches,
                B is batch size, C is embedding dimension.

        Returns:
            remaining_patches: Tensor of shape (T_remain, B, C) with patches
                after shuffling and masking.
            forward_indexes: LongTensor of shape (T, B) with forward permutation
                indices applied to the input.
            backward_indexes: LongTensor of shape (T, B) with inverse mapping
                such that original = gathered(remaining, backward_indexes).
        """
        T, B, C = patches.shape
        remain_T = int(T * (1.0 - self.ratio))

        # generate per-batch random permutations (numpy -> stack -> torch)
        indexes_list: List[Tuple[np.ndarray, np.ndarray]] = [
            random_indexes(T) for _ in range(B)
        ]
        forward_indexes_np = np.stack(
            [pair[0] for pair in indexes_list], axis=-1
        )  # shape (T, B)
        backward_indexes_np = np.stack(
            [pair[1] for pair in indexes_list], axis=-1
        )  # shape (T, B)

        device = patches.device
        forward_indexes = torch.as_tensor(
            forward_indexes_np, dtype=torch.long, device=device
        )
        backward_indexes = torch.as_tensor(
            backward_indexes_np, dtype=torch.long, device=device
        )

        # reorder patches and keep only the first `remain_T` entries
        patches = take_indexes(patches=patches, indexes=forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes


class MAEEncoder(torch.nn.Module):
    """
    Masked Autoencoder (MAE) encoder using a Vision Transformer backbone.

    The encoder:
      - Converts images to patch embeddings via a conv2d "patchify".
      - Adds positional embeddings.
      - Applies PatchShuffle to mask a fraction of patches.
      - Prepends a learnable class token.
      - Runs a stack of Transformer Blocks and a final LayerNorm.

    Notes on shapes:
      - Input images: (B, 3, H, W)
      - After patchify and rearrange: (T, B, C) where T = (H/patch_size)*(W/patch_size)
      - Returned features: (T+1, B, C) where first token is cls_token.
    """

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 6,
        emb_dim: int = 256,
        num_layer: int = 12,
        num_head: int = 4,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim

        # learnable class token and positional embeddings
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(num_patches, 1, emb_dim))

        # patch shuffling / masking
        self.shuffle = PatchShuffle(ratio=mask_ratio)

        # patchify: conv2d that extracts non-overlapping patches
        self.patchify = torch.nn.Conv2d(
            in_channels=3,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # transformer backbone
        self.transformer = torch.nn.Sequential(
            *[Block(dim=emb_dim, num_heads=num_head) for _ in range(num_layer)]
        )

        self.layer_norm = torch.nn.LayerNorm(normalized_shape=emb_dim)

        self.init_weight()

    def init_weight(self) -> None:
        """Initialize class token and positional embeddings with truncated normal."""
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Forward pass of the encoder.

        Args:
            img: input image tensor of shape (B, 3, H, W).

        Returns:
            features: Tensor of shape (T_remain+1, B, C) after transformer and LayerNorm.
            backward_indexes: LongTensor of shape (T, B) representing the inverse mappings
                to restore original patch ordering (useful for the decoder).
        """
        # patchify -> (B, C, H_p, W_p) -> (T, B, C)
        patches = self.patchify(img)
        patches = rearrange(patches, "b c h w -> (h w) b c")

        # add positional embedding (pos_embedding shape: (T, 1, C))
        patches = patches + self.pos_embedding

        # shuffle and mask patches
        patches, forward_indexes, backward_indexes = self.shuffle(patches=patches)

        # prepend class token along the patch/time dimension and run transformer
        patches = torch.cat(
            [self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0
        )
        patches = rearrange(patches, "t b c -> b t c")

        features = self.layer_norm(self.transformer(patches))

        # convert back to (T, B, C) layout
        features = rearrange(features, "b t c -> t b c")

        return features, backward_indexes
