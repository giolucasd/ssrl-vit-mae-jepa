"""
Masked Autoencoder (MAE) Decoder module.

Reconstructs masked image patches from the encoder's visible patch embeddings.
Follows the original MAE architecture with a lightweight Transformer decoder.
"""

from __future__ import annotations

from typing import Tuple

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block

from src.models.mae.encoder import take_indexes


class MAEDecoder(torch.nn.Module):
    """
    MAE decoder that reconstructs image patches from encoder features.

    The decoder:
      - Inserts mask tokens for the removed patches according to `backward_indexes`.
      - Restores original patch ordering.
      - Adds positional embeddings and passes tokens through a small Transformer.
      - Predicts RGB patch pixels via a linear head and reassembles image patches.

    Notes:
      - `features` expected shape: (T_visible+1, B, C) where the first token is the
        global/class token. This matches the MAEEncoder output.
      - `backward_indexes` expected shape: (T_all, B) (LongTensor) mapping original
        positions to positions in the shuffled/masked sequence produced by the encoder.
    """

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 6,
        emb_dim: int = 384,
        num_layer: int = 4,
        num_head: int = 6,
    ) -> None:
        """
        Args:
            image_size: input image size (assumes square image).
            patch_size: spatial size of each patch.
            emb_dim: embedding dimension for tokens.
            num_layer: number of transformer blocks in decoder.
            num_head: number of attention heads in decoder blocks.
        """
        super().__init__()

        # mask token and positional embeddings (includes cls token position)
        self.mask_token: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros(1, 1, emb_dim)
        )
        self.pos_embedding: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros(((image_size // patch_size) ** 2) + 1, 1, emb_dim)
        )

        # lightweight transformer decoder
        self.transformer: torch.nn.Sequential = torch.nn.Sequential(
            *[Block(dim=emb_dim, num_heads=num_head) for _ in range(num_layer)]
        )

        # head that predicts flattened RGB patch pixels
        self.head: torch.nn.Linear = torch.nn.Linear(
            in_features=emb_dim, out_features=(3 * patch_size**2)
        )

        # rearrange predicted patches back to image
        self.patch2img: Rearrange = Rearrange(
            pattern="(h w) b (c p1 p2) -> b c (h p1) (w p2)",
            p1=patch_size,
            p2=patch_size,
            h=(image_size // patch_size),
        )

        self.init_weight()

    def init_weight(self) -> None:
        """Initialize mask token and positional embeddings with truncated normal."""
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(
        self, features: torch.Tensor, backward_indexes: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct image and mask from encoder outputs.

        Args:
            features: token features from encoder, shape (T_visible+1, B, C).
            backward_indexes: LongTensor of shape (T_all, B) that maps original
                patch positions to encoder positions (inverse permutation per-batch).

        Returns:
            img: reconstructed images of shape (B, 3, H, W).
            mask: binary mask of same spatial size as img (1 for reconstructed/masked patches).
        """
        # Basic input validation to catch misuse early
        if features.ndim != 3:
            raise ValueError(
                f"features must be a 3D tensor (t, b, c), got shape {features.shape}"
            )
        if backward_indexes.ndim != 2:
            raise ValueError(
                f"backward_indexes must be a 2D LongTensor (t, b), got shape {backward_indexes.shape}"
            )
        if not backward_indexes.dtype == torch.long:
            backward_indexes = backward_indexes.long()

        T_visible_plus_cls: int = int(features.shape[0])  # includes cls token
        batch_size: int = int(features.shape[1])
        device: torch.device = features.device
        dtype: torch.dtype = features.dtype

        # Create an indices tensor that includes the cls token position (0) and
        # shifts original backward_indexes by +1 because cls token is prepended.
        cls_row = torch.zeros(
            (1, backward_indexes.shape[1]), dtype=backward_indexes.dtype, device=device
        )
        shifted_backward = torch.cat(
            tensors=[cls_row, (backward_indexes + 1)], dim=0
        )  # shape (T_all+1, B)

        # Append mask tokens for the missing patches so that total token count matches shifted_backward
        total_tokens_needed = shifted_backward.shape[0]
        missing_tokens = total_tokens_needed - features.shape[0]
        if missing_tokens < 0:
            raise ValueError(
                "Number of features provided is larger than expected total tokens based on backward_indexes."
            )
        if missing_tokens > 0:
            mask_tokens_expanded = self.mask_token.expand(
                missing_tokens, batch_size, -1
            ).to(dtype=dtype, device=device)
            features = torch.cat(tensors=[features, mask_tokens_expanded], dim=0)

        # Rearrange tokens back to original ordering using backward indexes
        features = take_indexes(patches=features, indexes=shifted_backward)

        # Add positional embeddings (pos_embedding shape: (T_all+1, 1, C))
        features = features + self.pos_embedding

        # Transformer expects (B, T, C)
        features = rearrange(features, "t b c -> b t c")
        features = self.transformer(features)
        features = rearrange(features, "b t c -> t b c")

        # remove global token (first token)
        features = features[1:]

        # Predict patch pixels
        patches = self.head(features)  # shape ((T_all), B, patch_dim)

        # Create mask indicating which patches were originally masked.
        mask = torch.zeros_like(patches, dtype=dtype, device=device)
        # The encoder removed (masked) patches; those correspond to positions from T_visible to end in encoder output
        # Original code set mask[T - 1 :] = 1 where T was features.shape[0] before adding masks.
        # Reproduce that behavior: set ones for indices from (T_visible_plus_cls - 1) onward in the shifted indexing.
        start_mask_idx = T_visible_plus_cls - 1  # matches original logic
        if start_mask_idx < mask.shape[0]:
            mask[start_mask_idx:] = 1.0

        # Reorder mask to original patch order (exclude cls token in reordering)
        mask = take_indexes(patches=mask, indexes=shifted_backward[1:] - 1)

        # Reconstruct image and mask spatial layout
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask
