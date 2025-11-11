from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.models.mae.classifier import ViTClassifier
from src.models.mae.encoder import MAEEncoder


class MAETrainModule(pl.LightningModule):
    """Lightning module for fine-tuning a pretrained MAE encoder on classification."""

    def __init__(
        self,
        pretrained_encoder: MAEEncoder,
        num_classes: int = 10,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_encoder"])

        # Build classifier
        self.model = ViTClassifier(encoder=pretrained_encoder, num_classes=num_classes)

        # Optionally freeze encoder parameters
        if freeze_encoder:
            for name, param in self.model.named_parameters():
                if "head" not in name:  # freeze everything except classification head
                    param.requires_grad = False

    # ------------------------------
    # Forward
    # ------------------------------
    def forward(self, x: torch.Tensor):
        return self.model(x)

    # ------------------------------
    # Training step
    # ------------------------------
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, labels)

        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    # ------------------------------
    # Validation step
    # ------------------------------
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, labels)

        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    # ------------------------------
    # Optimizer + Scheduler
    # ------------------------------
    def configure_optimizers(self):
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(epoch):
            warmup = (epoch + 1) / max(1, self.hparams.warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * epoch / self.hparams.total_epochs))
            return min(warmup, 1.0) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }
