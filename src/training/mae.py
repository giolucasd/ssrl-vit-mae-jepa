from __future__ import annotations

import math
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.models.mae import MaskedAutoencoder


class MAEPretrainModule(pl.LightningModule):
    """Self-supervised pretraining for Masked Autoencoder."""

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        training_cfg: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MaskedAutoencoder(
            general_cfg=model_cfg["general"],
            encoder_cfg=model_cfg["encoder"],
            decoder_cfg=model_cfg["decoder"],
        )

        self.mask_start = training_cfg.get("mask_ratio_start", 0.5)
        self.mask_end = training_cfg.get("mask_ratio_end", 0.85)
        self.ramp_epochs = training_cfg.get("mask_ramp_epochs", 200)

        self.lr = float(training_cfg.get("base_learning_rate", 1.5e-4))
        self.weight_decay = float(training_cfg.get("weight_decay", 0.05))
        self.warmup_epochs = int(training_cfg.get("warmup_epochs", 20))
        self.total_epochs = int(training_cfg.get("total_epochs", 200))
        self.batch_size = int(training_cfg.get("batch_size", 512))
        self.criterion = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        preds, targets = self(imgs)
        loss = self.criterion(preds, targets)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        preds, targets = self(imgs)
        loss = self.criterion(preds, targets)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        effective_lr = self.lr * self.batch_size / 256
        optimizer = AdamW(
            self.parameters(),
            lr=effective_lr,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(epoch):
            warmup = (epoch + 1) / max(1, self.warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * epoch / self.total_epochs))
            return min(warmup, 1.0) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "name": "lr"},
        }

    def on_train_epoch_start(self):
        """Update mask ratio linearly over epochs."""
        progress = min(self.current_epoch / max(1, self.ramp_epochs - 1), 1.0)
        new_mask = self.mask_start + progress * (self.mask_end - self.mask_start)
        self.model.mask_ratio = new_mask
        self.log("mask_ratio", new_mask, prog_bar=True)
