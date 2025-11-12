from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import get_pretrain_dataloaders
from src.training.mae import MAEPretrainModule


def parse_args():
    parser = argparse.ArgumentParser(description="Self-supervised MAE pretraining")
    parser.add_argument("--config", type=str, default="configs/mae.yaml")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------
    # Load configuration
    # ------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 73)
    pl.seed_everything(seed, workers=True)

    pre_cfg = cfg["pretrain"]
    model_cfg = cfg["model"]
    log_cfg = cfg["logging"]

    # ------------------------------
    # Output dirs
    # ------------------------------
    output_dir = (
        Path(log_cfg["output_dir_base"])
        / "pretrain"
        / pre_cfg.get("output_dir_suffix", "default")
    )
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Data
    # ------------------------------
    train_loader, val_loader = get_pretrain_dataloaders(cfg)

    # ------------------------------
    # Model setup
    # ------------------------------
    module = MAEPretrainModule(
        model_cfg=model_cfg,
        training_cfg=pre_cfg,
    )

    # ------------------------------
    # Logging + Checkpoints
    # ------------------------------
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tb")

    ckpt_best = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=False,
        verbose=True,
    )
    ckpt_last = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="last",
        save_top_k=1,
        every_n_epochs=1,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ------------------------------
    # Trainer
    # ------------------------------
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=pre_cfg["total_epochs"],
        logger=tb_logger,
        callbacks=[ckpt_best, ckpt_last, lr_monitor],
        log_every_n_steps=10,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
    )

    trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume_from)

    # ------------------------------
    # Save model + metadata
    # ------------------------------
    model_path = output_dir / log_cfg["model_path"]
    torch.save(module.model.state_dict(), model_path)

    print("\n✅ Pretraining complete")
    print(f"📦 Model weights saved to: {model_path}")
    print(f"🏁 Best checkpoint: {ckpt_best.best_model_path}")
    print(f"📈 Logs available at: {tb_logger.log_dir}")


if __name__ == "__main__":
    main()
