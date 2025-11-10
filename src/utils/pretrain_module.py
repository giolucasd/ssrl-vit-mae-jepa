from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.models.mae.mae_vit import MAEViT


class MAEPretrainModule(pl.LightningModule):
    """Self-supervised MAE pretraining with Lightning."""

    def __init__(
        self,
        mask_ratio: float = 0.75,
        base_learning_rate: float = 1.5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 200,
        total_epochs: int = 2000,
        batch_size: int = 4096,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MAEViT(mask_ratio=mask_ratio)

    # ------------------------------
    # Forward
    # ------------------------------
    def forward(self, x: torch.Tensor):
        return self.model(x)

    # ------------------------------
    # Training step
    # ------------------------------
    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        predicted_img, mask = self(imgs)

        # Reconstruction loss on masked patches only
        loss = ((predicted_img - imgs) ** 2 * mask).sum() / mask.sum()

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    # ------------------------------
    # Optimizer + Scheduler
    # ------------------------------
    def configure_optimizers(self):
        effective_lr = self.hparams.base_learning_rate * self.hparams.batch_size / 256
        optimizer = AdamW(
            self.model.parameters(),
            lr=effective_lr,
            betas=(0.9, 0.95),
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(epoch):
            warmup = (epoch + 1) / max(1, self.hparams.warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * epoch / self.hparams.total_epochs))
            return min(warmup, 1.0) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "lr",
            },
        }
