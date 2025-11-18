"""
Reconstruction validation module for Masked Autoencoder (MAE) models.

This module provides functionality to validate MAE models by reconstructing
masked images and visualizing the results.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from lightly.models import utils
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.data import get_train_dataloaders
from src.models.mae import MaskedAutoencoder

from ..utils import setup_reproducibility, shut_down_warnings

shut_down_warnings()
setup_reproducibility(seed=73)


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Reconstruct image")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mae.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/pretrain/mae_100/checkpoints/best.ckpt",
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--output_path_suffix",
        type=str,
        default="reconstruction_validation.png",
        help="Suffix to append to the output path.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    """Loads a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


class MAEReconstructor:
    """Handles MAE model reconstruction and validation."""

    def __init__(
        self, model_path: str, device: Optional[str] = None, mask_ratio: float = 0.75
    ):
        """
        Initialize the MAE reconstructor.

        Args:
            model_path: Path to the saved model checkpoint
            device: Device to run inference on. If None, auto-detects GPU/CPU
            mask_ratio: Ratio of patches to mask (0.0 to 1.0)
        """
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_ratio = mask_ratio
        self.model = None

    def load_model(
        self,
        general_cfg: Dict[str, Any],
        encoder_cfg: Dict[str, Any],
        decoder_cfg: Dict[str, Any],
    ) -> None:
        """
        Load the MAE model from checkpoint.

        Args:
            general_cfg: General model configuration
            encoder_cfg: Encoder configuration
            decoder_cfg: Decoder configuration
        """

        # Initialize model
        self.model = MaskedAutoencoder(
            general_cfg=general_cfg, encoder_cfg=encoder_cfg, decoder_cfg=decoder_cfg
        )

        # Load checkpoint
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Handle different checkpoint formats
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Clean the state dict by removing the "model." prefix
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    new_key = key[6:]  # Remove "model." prefix
                    cleaned_state_dict[new_key] = value
                else:
                    cleaned_state_dict[key] = value

            self.model.load_state_dict(cleaned_state_dict, strict=False)
            print(f"Model loaded successfully from {self.model_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {self.model_path}")

        self.model.to(self.device)
        self.model.eval()

    def reconstruct_batch(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct a batch of masked images.

        Args:
            images: Input images tensor [B, C, H, W]

        Returns:
            Tuple of (original_images, masked_images, reconstructed_images)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        images = images.to(self.device)

        with torch.no_grad():
            # Get random mask
            batch_size = images.shape[0]
            sequence_length = self.model.sequence_length

            # Create fixed mask indices (same for all batches)
            torch.manual_seed(42)  # Set seed for reproducible mask
            idx_keep, idx_mask = utils.random_token_mask(
                size=(batch_size, sequence_length),
                mask_ratio=self.mask_ratio,
                device=self.device,
            )

            # Forward pass through encoder and decoder
            x_encoded = self.model.forward_encoder(images=images, idx_keep=idx_keep)
            x_pred = self.model.forward_decoder(x_encoded, idx_keep, idx_mask)

            # Create masked images for visualization
            masked_images = self._create_masked_images(images, idx_keep, idx_mask)
            # Reconstruct full images
            reconstructed = self._reconstruct_full_images(
                images, x_pred, idx_keep, idx_mask
            )

        return images, masked_images, reconstructed

    def _create_masked_images(
        self, images: torch.Tensor, idx_keep: torch.Tensor, idx_mask: torch.Tensor
    ) -> torch.Tensor:
        """Create masked versions of images for visualization."""
        patches = utils.patchify(images=images, patch_size=self.model.patch_size)
        # Set masked patches to zero (or gray)
        masked_patches = patches.clone()
        idx_mask_adj = torch.clamp(idx_mask - 1, min=0)

        # Create mask tensor
        mask = torch.zeros_like(patches)
        for i in range(patches.shape[0]):
            mask[i, idx_mask_adj[i]] = 1

        # Apply mask (set to gray value)
        masked_patches = masked_patches * (1 - mask) + mask * 0.5
        # Reconstruct images from patches
        return utils.unpatchify(
            patches=masked_patches,
            patch_size=self.model.patch_size,
        )

    def __remove_cls_token(self, idx: torch.Tensor, B: int) -> torch.Tensor:
        mask_not_cls_keep = idx != 0
        idx_no_cls = idx[mask_not_cls_keep].reshape(B, -1)
        idx_no_cls = idx_no_cls - 1  # Adjust indices
        return idx_no_cls

    def _reconstruct_full_images(
        self,
        original: torch.Tensor,
        predictions: torch.Tensor,
        idx_keep: torch.Tensor,
        idx_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct full images by combining original and predicted patches."""

        patches = utils.patchify(images=original, patch_size=self.model.patch_size)
        B, N, D = patches.shape
        full_tokens = torch.zeros_like(patches)
        # The following logic assumes that the CLS token (index 0) is always kept
        # and needs to be removed from the indices before reconstructing the image,
        # as the image does not have a CLS token.

        # Remove CLS token index (0) from idx_keep
        idx_keep_no_cls = self.__remove_cls_token(idx_keep, B)
        # Remove CLS token index (0) from idx_mask if it exists
        idx_mask_no_cls = self.__remove_cls_token(idx_mask, B)

        # Gather the patches that were kept (visible)
        kept_patches = patches.gather(
            1, idx_keep_no_cls.unsqueeze(-1).expand(-1, -1, D)
        )
        # Insert original visible tokens
        full_tokens.scatter_(
            1, idx_keep_no_cls.unsqueeze(-1).expand(-1, -1, D), kept_patches
        )

        # Insert predicted masked tokens
        full_tokens.scatter_(
            1, idx_mask_no_cls.unsqueeze(-1).expand(-1, -1, D), predictions
        )

        # Step 4: unpatchify back into the image
        return utils.unpatchify(full_tokens, self.model.patch_size)

    def validate_reconstruction(
        self,
        dataloader: DataLoader,
        num_samples: int = 8,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Validate model reconstruction on dataset samples.

        Args:
            dataloader: DataLoader for validation dataset
            num_samples: Number of samples to visualize
            save_path: Path to save visualization (optional)
        """
        # Get a batch of data
        data_iter = iter(dataloader)
        batch = next(data_iter)

        if isinstance(batch, (list, tuple)):
            images = batch[0]  # Assume first element is images
        else:
            images = batch

        # Limit to num_samples
        images = images[:num_samples]

        # Reconstruct
        original, masked, reconstructed = self.reconstruct_batch(images)

        # Visualize results
        self._visualize_reconstruction(
            original.cpu(), masked.cpu(), reconstructed.cpu(), save_path
        )

        # Print statistics
        self._print_reconstruction_stats(original.cpu(), reconstructed.cpu())

    def _visualize_reconstruction(
        self,
        original: torch.Tensor,
        masked: torch.Tensor,
        reconstructed: torch.Tensor,
        save_path: Optional[str] = None,
    ) -> None:
        """Visualize original, masked, and reconstructed images."""
        num_samples = original.shape[0]

        fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 6))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)

        for i in range(num_samples):
            # Original
            axes[0, i].imshow(self._tensor_to_image(original[i]))
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")

            # Masked
            axes[1, i].imshow(self._tensor_to_image(masked[i]))
            axes[1, i].set_title(f"Masked ({self.mask_ratio:.0%})")
            axes[1, i].axis("off")

            # Reconstructed
            axes[2, i].imshow(self._tensor_to_image(reconstructed[i]))
            axes[2, i].set_title("Reconstructed")
            axes[2, i].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved to {save_path}")

        plt.show()

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to displayable image."""
        # Denormalize based on the provided transformation
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])

        if tensor.shape[0] == 3:  # CHW format
            tensor = tensor * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
            tensor = tensor.permute(1, 2, 0)  # HWC

        tensor = torch.clamp(tensor, 0, 1)
        return tensor.numpy()

    def _print_reconstruction_stats(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> None:
        """Print reconstruction quality statistics."""
        mse = nn.MSELoss()(original, reconstructed)
        mae = nn.L1Loss()(original, reconstructed)

        print("\nReconstruction Statistics:")
        print(f"MSE Loss: {mse.item():.6f}")
        print(f"MAE Loss: {mae.item():.6f}")
        print(f"PSNR: {-10 * torch.log10(mse).item():.2f} dB")


def main():
    """Example usage of MAEReconstructor."""
    args = parse_args()
    config = load_config(args.config)

    # --- Data Loading ---
    _, val_loader = get_train_dataloaders(config)

    general_cfg = config["model"]["general"]

    encoder_cfg = config["model"]["encoder"]

    decoder_cfg = config["model"]["decoder"]

    # Initialize reconstructor
    reconstructor = MAEReconstructor(
        model_path=args.model_path,
        mask_ratio=config["pretrain"].get("mask_ratio_end", 0.75),
    )

    # Load model
    reconstructor.load_model(general_cfg, encoder_cfg, decoder_cfg)

    # Validate (assuming you have train_dataloader and val_dataloader)
    save_dir = Path("assets") / "visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)
    reconstructor.validate_reconstruction(
        dataloader=val_loader,
        num_samples=8,
        save_path=save_dir / args.output_path_suffix,
    )


if __name__ == "__main__":
    main()
