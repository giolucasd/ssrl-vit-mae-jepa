"""
Masked Autoencoder (MAE) Decoder module.

Reconstructs masked image patches from the encoder's visible patch embeddings.
Follows the original MAE architecture with a lightweight Transformer decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block

from src.models.mae.encoder import take_indexes


class MAEDecoder(nn.Module):
    """Masked Autoencoder Vision Transformer Decoder."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 2,
        emb_dim: int = 192,
        decoder_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 16,
    ) -> None:
        """
        Args:
            image_size: Input image size (assumed square).
            patch_size: Patch size used by the encoder.
            emb_dim: Encoder embedding dimension.
            decoder_dim: Decoder embedding dimension.
            num_layers: Number of Transformer blocks in the decoder.
            num_heads: Number of attention heads in each Transformer block.
        """
        super().__init__()

        num_patches = (image_size // patch_size) ** 2

        # Embedding projection from encoder dim to decoder dim
        self.proj = nn.Linear(emb_dim, decoder_dim, bias=True)

        # Learnable mask token (represents missing patches)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # Positional embeddings for all patches (shared with masked ones)
        self.pos_embedding = nn.Parameter(torch.zeros(num_patches, 1, decoder_dim))

        # Transformer decoder
        self.transformer = nn.Sequential(
            *[Block(decoder_dim, num_heads) for _ in range(num_layers)]
        )

        # Final normalization and reconstruction head
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_size**2 * 3, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following MAE convention."""
        trunc_normal_(self.pos_embedding, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        encoded: torch.Tensor,
        backward_indexes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MAE decoder.

        Args:
            encoded: Encoded features of shape (T_vis+1, B, emb_dim).
            backward_indexes: Index tensor to reconstruct the original patch order.

        Returns:
            pred: Reconstructed image patches of shape (B, num_patches, 3, patch_size, patch_size).
            mask: Binary mask tensor of shape (B, num_patches, 1, 1, 1),
                where 1 indicates a masked patch.
        """
        # Remove CLS token
        encoded = encoded[1:]
        T_vis, B, _ = encoded.shape

        # Project to decoder dimension
        decoded = self.proj(encoded)

        # Prepare mask tokens
        T_total = backward_indexes.shape[0]
        mask_tokens = self.mask_token.expand(T_total - T_vis, B, -1)

        # Concatenate visible and mask tokens
        full_sequence = torch.cat([decoded, mask_tokens], dim=0)

        # Reorder to original patch positions
        full_sequence = take_indexes(full_sequence, backward_indexes)

        # Add positional embeddings
        full_sequence = full_sequence + self.pos_embedding

        # Transformer decoding
        full_sequence = rearrange(full_sequence, "t b c -> b t c")
        decoded = self.transformer(full_sequence)
        decoded = self.norm(decoded)
        decoded = rearrange(decoded, "b t c -> t b c")

        # Predict pixel values (flattened patches)
        pred = self.head(decoded)  # (T, B, patch_size^2 * 3)

        # Reshape to image patches
        patch_size = int((pred.shape[-1] // 3) ** 0.5)
        pred = rearrange(pred, "t b (c p1 p2) -> b t c p1 p2", p1=patch_size, c=3)

        # ===== NEW: reconstruct mask =====
        mask = torch.zeros(T_total, B, 1, device=encoded.device)
        mask[T_vis:] = 1.0
        mask = take_indexes(mask, backward_indexes)
        mask = rearrange(mask, "t b c -> b t c 1 1")

        return pred, mask
