# scripts/pretrain_mae.py
from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import get_pretrain_dataloader
from src.training.mae_trainers import MAEPretrainModule

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(description="MAE pretraining (config-driven)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mae.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint to resume from.",
    )
    parser.add_argument(
        "--override_output_dir",
        type=str,
        default=None,
        help="Optionally override output dir defined in config.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    pl.seed_everything(cfg["training"]["seed"])
    output_dir = Path(args.override_output_dir or cfg["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Logger + Checkpoint setup
    # ------------------------------
    tb_logger = TensorBoardLogger(save_dir=str(output_dir), name="logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="mae-{epoch:03d}-{train_loss:.3f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        save_last=True,
    )

    # ------------------------------
    # Data + Model setup
    # ------------------------------
    loader = get_pretrain_dataloader(
        batch_size=cfg["training"]["batch_size"],
        data_fraction=cfg["training"]["data_fraction"],
    )

    model = MAEPretrainModule(
        model_cfg=cfg["model"],
        training_cfg=cfg["training"],
    )

    # ------------------------------
    # Trainer
    # ------------------------------
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=cfg["training"]["total_epochs"],
        default_root_dir=str(output_dir),
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
    )

    trainer.fit(model, loader, ckpt_path=args.resume_from)

    # ------------------------------
    # Save final weights
    # ------------------------------
    save_path = output_dir / cfg["logging"]["model_path"]
    torch.save(model.model.state_dict(), save_path)
    print(f"\n✅ Pretraining finished. Model saved to: {save_path}")
    print(f"📈 TensorBoard logs at: {tb_logger.log_dir}")
    print(f"💾 Checkpoints saved in: {checkpoint_callback.dirpath}")


if __name__ == "__main__":
    main()
