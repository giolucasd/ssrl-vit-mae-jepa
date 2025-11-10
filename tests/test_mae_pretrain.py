"""
Quick integration test for MAE pretraining using STL10.

Runs a very short training (few batches) to verify that:
- Encoder and decoder connect correctly
- Forward/backward pass works
- Loss decreases over time
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import STL10

from src.models.mae.mae_vit import MAEViT

torch.set_float32_matmul_precision("medium")

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


class MAEPretrainModule(pl.LightningModule):
    """Minimal PyTorch Lightning module for MAE pretraining."""

    def __init__(
        self,
        mask_ratio: float = 0.75,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MAEViT(mask_ratio=mask_ratio)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        predicted_img, mask = self(imgs)
        loss = torch.mean((predicted_img - imgs) ** 2 * mask) / self.hparams.mask_ratio
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )
        return optimizer


def get_dataloaders(batch_size: int = 8):
    """Returns tiny STL10 dataloaders for quick sanity checks."""
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    # Use small subsets to keep it lightweight
    train_data = STL10(DATA_DIR, split="unlabeled", transform=transform)
    subset = Subset(train_data, range(16))  # just 16 samples for sanity check
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader


def main():
    train_loader = get_dataloaders()

    model = MAEPretrainModule()

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=10,
        limit_train_batches=2,  # just a couple of batches
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
