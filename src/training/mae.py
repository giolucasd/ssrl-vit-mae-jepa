from __future__ import annotations

import math
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.models.mae import MAEClassifier, MaskedAutoencoder


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


class MAETrainModule(pl.LightningModule):
    """Supervised fine-tuning or linear probing using a pretrained MAE encoder."""

    def __init__(
        self,
        pretrained_encoder: Optional[torch.nn.Module] = None,
        model_cfg: Optional[Dict[str, Any]] = None,
        training_cfg: Optional[Dict[str, Any]] = None,
        num_classes: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_encoder"])

        self.model_cfg = model_cfg or {}
        self.training_cfg = training_cfg or {}

        self.learning_rate = float(training_cfg.get("learning_rate", 3e-4))
        self.weight_decay = float(training_cfg.get("weight_decay", 0.05))
        self.warmup_epochs = int(training_cfg.get("warmup_epochs", 5))
        self.total_epochs = int(training_cfg.get("total_epochs", 100))
        self.freeze_encoder_flag = self.training_cfg.get("freeze_encoder", True)
        self.num_classes = num_classes

        # Build model
        if pretrained_encoder is not None:
            self.model = MAEClassifier(
                pretrained_encoder=pretrained_encoder,
                num_classes=self.num_classes,
                head_cfg=self.model_cfg.get("head", {}),
            )
        else:
            print(
                "⚙️ No pretrained encoder provided; model can be reloaded from checkpoint."
            )
            self.model = None

        # Freeze or unfreeze encoder as requested
        if self.model is not None:
            if self.freeze_encoder_flag:
                self.freeze_encoder()
            else:
                self.unfreeze_encoder()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

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

    def unfreeze_last_layers(self, n_layers: int):
        """
        Unfreezes only the last `n_layers` Transformer blocks of the ViT encoder.
        All earlier layers remain frozen.
        """

        if self.model is None:
            raise RuntimeError("Model not built yet; cannot unfreeze layers.")

        encoder = self.model.encoder  # timm VisionTransformer
        blocks = encoder.blocks  # list of Transformer blocks
        total = len(blocks)

        if n_layers < 0 or n_layers > total:
            raise ValueError(f"n_layers must be between 0 and {total}, got {n_layers}")

        print(f"🔓 Unfreezing last {n_layers} of {total} encoder layers...")

        # 1) Freeze ALL parameters first
        for param in encoder.parameters():
            param.requires_grad = False

        # 2) Unfreeze the last N Transformer blocks
        for block in blocks[total - n_layers :]:
            for param in block.parameters():
                param.requires_grad = True

        # 3) Also unfreeze the final LN (norm) layer
        if hasattr(encoder, "norm"):
            for param in encoder.norm.parameters():
                param.requires_grad = True

        # 4) Head (classifier) is always trainable
        for param in self.model.head.parameters():
            param.requires_grad = True

        print("🔥 Selective unfreezing complete.")

    def on_load_checkpoint(self, checkpoint):
        """Rebuild classifier if loading from checkpoint without explicit model."""
        if self.model is None:
            print("🔁 Reinitializing MAEClassifier from checkpoint metadata...")
            from timm.models.vision_transformer import VisionTransformer

            encoder_cfg = self.model_cfg.get("encoder", {})
            encoder = VisionTransformer(
                img_size=self.model_cfg["general"]["image_size"],
                patch_size=self.model_cfg["general"]["patch_size"],
                in_chans=self.model_cfg["general"]["in_chans"],
                embed_dim=encoder_cfg.get("embed_dim", 384),
                depth=encoder_cfg.get("depth", 12),
                num_heads=encoder_cfg.get("num_heads", 6),
                num_classes=0,
            )

            self.model = MAEClassifier(
                pretrained_encoder=encoder,
                num_classes=self.num_classes,
                head_cfg=self.model_cfg.get("head", {}),
            )

        if self.freeze_encoder_flag:
            self.freeze_encoder()
        else:
            self.unfreeze_encoder()
