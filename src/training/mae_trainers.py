from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.models.mae.classifier import ViTClassifier
from src.models.mae.encoder import MAEEncoder
from src.models.mae.mae_vit import MAEViT


class MAEPretrainModule(pl.LightningModule):
    """Self-supervised MAE pretraining with configuration-driven model creation."""

    def __init__(
        self,
        model_cfg: dict,
        training_cfg: dict,
    ):
        """
        Args:
            model_cfg: dict of model hyperparameters (image_size, patch_size, emb_dim, etc.)
            training_cfg: dict of training hyperparameters (lr, epochs, batch_size, etc.)
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model_cfg", "training_cfg"])
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg

        # Instantiate MAE model directly from config
        self.model = MAEViT(**model_cfg)

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

        # Reconstruction loss only on masked patches
        loss = ((predicted_img - imgs) ** 2 * mask).sum() / mask.sum()
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    # ------------------------------
    # Optimizer + Scheduler
    # ------------------------------
    def configure_optimizers(self):
        batch_size = self.training_cfg["batch_size"]
        base_lr = self.training_cfg["base_learning_rate"]
        total_epochs = self.training_cfg["total_epochs"]
        warmup_epochs = self.training_cfg["warmup_epochs"]
        weight_decay = self.training_cfg["weight_decay"]

        effective_lr = base_lr * batch_size / 256

        optimizer = AdamW(
            self.model.parameters(),
            lr=effective_lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )

        def lr_lambda(epoch):
            warmup = (epoch + 1) / max(1, warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
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


class MAETrainModule(pl.LightningModule):
    """Lightning module for fine-tuning or training a pretrained MAE encoder."""

    def __init__(
        self,
        pretrained_encoder: MAEEncoder | None = None,
        model_cfg: dict | None = None,
        training_cfg: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_encoder"])

        # ------------------------------
        # Configs
        # ------------------------------
        model_cfg = model_cfg or {}
        training_cfg = training_cfg or {}

        self.learning_rate = training_cfg.get("learning_rate", 3e-4)
        self.weight_decay = training_cfg.get("weight_decay", 0.05)
        self.warmup_epochs = training_cfg.get("warmup_epochs", 5)
        self.total_epochs = training_cfg.get("total_epochs", 100)
        self.num_classes = model_cfg.get("num_classes", 10)
        self.freeze_encoder_flag = training_cfg.get("freeze_encoder", True)

        # ------------------------------
        # Model setup
        # ------------------------------
        if pretrained_encoder is not None:
            self.model = ViTClassifier(
                encoder=pretrained_encoder,
                num_classes=self.num_classes,
            )
        else:
            print(
                "⚙️ No pretrained encoder provided — model will be reinitialized on load."
            )
            self.model = None

        # Freeze or unfreeze as requested
        if self.model is not None:
            if self.freeze_encoder_flag:
                self.freeze_encoder()
            else:
                self.unfreeze_encoder()

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
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(epoch):
            warmup = (epoch + 1) / max(1, self.warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * epoch / self.total_epochs))
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

    # ------------------------------
    # Encoder freezing utilities
    # ------------------------------
    def freeze_encoder(self):
        if self.model is None:
            return
        for name, param in self.model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        print("🧊 Encoder frozen (only classifier head is trainable).")

    def unfreeze_encoder(self):
        if self.model is None:
            return
        for param in self.model.parameters():
            param.requires_grad = True
        print("🔥 Encoder unfrozen (all parameters trainable).")

    # ------------------------------
    # Checkpoint loading hook
    # ------------------------------
    def on_load_checkpoint(self, checkpoint):
        """Rebuild model automatically if missing (for load_from_checkpoint)."""
        if self.model is None:
            print("🔁 Reinitializing ViTClassifier from checkpoint metadata...")
            encoder = MAEEncoder()
            self.model = ViTClassifier(
                encoder=encoder,
                num_classes=self.hparams.get("num_classes", 10),
            )

        if self.freeze_encoder_flag:
            self.freeze_encoder()
        else:
            self.unfreeze_encoder()
