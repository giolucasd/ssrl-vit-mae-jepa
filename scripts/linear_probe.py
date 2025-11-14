from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from timm.models.vision_transformer import VisionTransformer

from src.data import get_train_dataloaders
from src.models.linear_probe import LinearProbe
from src.training.mae import MAETrainModule

warnings.filterwarnings(
    "ignore",
    "Precision 16-mixed is not supported",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    "Please use the new API settings to control TF32 behavior",
    category=UserWarning,
)

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear probe evaluation of MAE encoder"
    )
    parser.add_argument("--config", type=str, default="configs/mae.yaml")
    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        required=True,
        help="Path to pretrained MAE encoder checkpoint (.pt or .ckpt)",
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

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    log_cfg = cfg["logging"]

    # ------------------------------
    # Output dirs
    # ------------------------------
    output_dir = (
        Path(log_cfg["output_dir_base"])
        / "linear_probe"
        / train_cfg.get("output_dir_suffix", "default")
    )
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Save a copy of the config
    # ------------------------------
    config_copy_path = output_dir / "config.yaml"
    with open(config_copy_path, "w") as f_out:
        yaml.safe_dump(cfg, f_out)
    print(f"📝 Saved config snapshot to: {config_copy_path}")

    # ------------------------------
    # Data
    # ------------------------------
    train_loader, val_loader = get_train_dataloaders(cfg)

    # ------------------------------
    # Load pretrained encoder
    # ------------------------------
    print(f"🧩 Loading pretrained encoder from: {args.encoder_ckpt}")
    encoder = VisionTransformer(
        img_size=model_cfg["general"]["image_size"],
        patch_size=model_cfg["general"]["patch_size"],
        in_chans=model_cfg["general"]["in_chans"],
        embed_dim=model_cfg["encoder"]["embed_dim"],
        depth=model_cfg["encoder"]["depth"],
        num_heads=model_cfg["encoder"]["num_heads"],
        num_classes=0,  # no head
    )

    # Load pretrained weights
    state = torch.load(args.encoder_ckpt, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    encoder.load_state_dict(state, strict=False)

    # ------------------------------
    # Build linear probe
    # ------------------------------
    model = LinearProbe(
        encoder=encoder,
        num_classes=10,
        head_cfg=model_cfg.get("head", {}),
    )

    module = MAETrainModule(
        pretrained_encoder=encoder,
        model_cfg=model_cfg,
        training_cfg=train_cfg,
    )
    module.model = model

    # ------------------------------
    # Logging + Checkpoints
    # ------------------------------
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tb")

    ckpt_best = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        save_top_k=1,
        monitor="val_acc",
        mode="max",
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
        accelerator="auto",
        devices=1,
        max_epochs=train_cfg["total_epochs"],
        logger=tb_logger,
        callbacks=[ckpt_best, ckpt_last, lr_monitor],
        log_every_n_steps=10,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(module, train_loader, val_loader)

    # ------------------------------
    # Save model + report
    # ------------------------------
    model_path = output_dir / log_cfg["model_path"]
    torch.save(model.state_dict(), model_path)

    print("\n✅ Linear probe training complete")
    print(f"📦 Model weights saved to: {model_path}")
    print(f"🏁 Best checkpoint: {ckpt_best.best_model_path}")
    print(f"📈 Logs available at: {tb_logger.log_dir}")


if __name__ == "__main__":
    main()
