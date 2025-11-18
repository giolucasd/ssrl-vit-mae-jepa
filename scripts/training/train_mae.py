from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from timm.models.vision_transformer import VisionTransformer

from src.data import get_train_dataloaders
from src.models.mae import MaskedAutoencoder
from src.training.mae import MAETrainModule

from ..utils import setup_reproducibility, shut_down_warnings

shut_down_warnings()
setup_reproducibility(seed=73)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune or train MAE encoder on classification task"
    )
    parser.add_argument("--config", type=str, default="configs/mae.yaml")
    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        default=None,
        help="Path to pretrained MAE encoder weights (.pt or .ckpt)",
    )
    parser.add_argument(
        "--classifier_ckpt",
        type=str,
        default=None,
        help="Path to full classifier checkpoint (for fine-tuning continuation)",
    )
    parser.add_argument(
        "--output_dir_suffix",
        type=str,
        default="mae_finetune",
        help="Suffix for the output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------
    # Load configuration
    # ------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    log_cfg = cfg["logging"]

    # ------------------------------
    # Output dirs
    # ------------------------------
    output_dir = Path(log_cfg["output_dir_base"]) / "train" / args.output_dir_suffix
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
    # Model setup
    # ------------------------------
    if args.classifier_ckpt:
        print(f"🔁 Loading full classifier checkpoint: {args.classifier_ckpt}")
        module = MAETrainModule.load_from_checkpoint(
            args.classifier_ckpt,
            map_location="cpu",
            strict=False,
        )
    elif args.encoder_ckpt:
        # Build model from encoder weights or from scratch
        print(f"🧩 Loading pretrained encoder: {args.encoder_ckpt}")
        mae = MaskedAutoencoder(
            general_cfg=model_cfg["general"],
            encoder_cfg=model_cfg["encoder"],
            decoder_cfg=model_cfg["decoder"],
        )

        if args.encoder_ckpt:
            ckpt = torch.load(args.encoder_ckpt, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)

            # Possible prefixes where encoder weights may live
            possible_prefixes = [
                "model.encoder.",  # LightningModule → self.model.encoder
                "encoder.",  # plain state_dict from MAE
                "module.encoder.",  # DDP/DataParallel
            ]

            # Detect which prefix exists in this checkpoint
            prefix = None
            for p in possible_prefixes:
                if any(k.startswith(p) for k in state_dict.keys()):
                    prefix = p
                    break

            if prefix is None:
                raise ValueError(
                    "❌ Could not find encoder weights in checkpoint. "
                    "Expected keys starting with one of: "
                    + ", ".join(possible_prefixes)
                )

            print(f"🔎 Detected encoder prefix in checkpoint: '{prefix}'")

            # Extract only encoder weights and strip the prefix
            encoder_state = {
                k[len(prefix) :]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }

            # Now load them safely into the MAE encoder
            missing, unexpected = mae.encoder.load_state_dict(
                encoder_state,
                strict=False,  # allow decoder/head keys to be ignored
            )

            print(
                f"✅ Loaded encoder weights: {len(encoder_state)} tensors "
                f"({len(missing)} missing, {len(unexpected)} unexpected)"
            )
        else:
            print("⚠️ No encoder checkpoint provided — training from scratch!")

        module = MAETrainModule(
            pretrained_encoder=mae.encoder.vit,
            model_cfg=model_cfg,
            training_cfg=train_cfg,
        )
    else:
        print("🧪 Baseline: random-initialized VisionTransformer (no MAE)")

        encoder = VisionTransformer(
            img_size=model_cfg["general"]["image_size"],
            patch_size=model_cfg["general"]["patch_size"],
            in_chans=model_cfg["general"]["in_chans"],
            embed_dim=model_cfg["encoder"]["embed_dim"],
            depth=model_cfg["encoder"]["depth"],
            num_heads=model_cfg["encoder"]["num_heads"],
            num_classes=0,  # no cls head
        )

        module = MAETrainModule(
            pretrained_encoder=encoder,
            model_cfg=model_cfg,
            training_cfg=train_cfg,
        )

    # Unfreeze encoder if configured
    if train_cfg.get("unfreeze_last_layers", None) is not None:
        n_layers = int(train_cfg["unfreeze_last_layers"])
        print(f"🧠 Unfreezing {n_layers} encoder layers...")
        module.unfreeze_last_layers(n_layers)
    elif train_cfg.get("freeze_encoder", True):
        print("🧊 Freezing encoder weights...")
        module.freeze_encoder()
    else:
        print("🧠 Unfreezing encoder weights...")
        module.unfreeze_encoder()

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
        log_every_n_steps=2,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(module, train_loader, val_loader)

    # ------------------------------
    # Save model + summary
    # ------------------------------
    model_path = output_dir / log_cfg["model_path"]
    torch.save(module.model.state_dict(), model_path)

    print("\n✅ Training complete")
    print(f"📦 Model weights saved to: {model_path}")
    print(f"🏁 Best checkpoint: {ckpt_best.best_model_path}")
    print(f"📈 Logs available at: {tb_logger.log_dir}")


if __name__ == "__main__":
    main()
