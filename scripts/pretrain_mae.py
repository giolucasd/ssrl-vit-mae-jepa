from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import STL10

from src.utils.pretrain_module import MAEPretrainModule

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(description="MAE pretraining on STL-10")

    # Core hyperparameters
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_device_batch_size", type=int, default=512)
    parser.add_argument("--base_learning_rate", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--total_epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=10)

    # Data fraction (percentage of unlabeled samples)
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        choices=[0.05, 0.25, 0.5, 0.75, 1.0],
        help="Fraction of unlabeled STL-10 to use.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pretrain",
        help="Where to save checkpoints and logs.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="vit-t-mae.pt",
        help="Final model save path.",
    )

    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (from zero if not provided).",
    )

    return parser.parse_args()


def get_dataloader(data_dir: Path, batch_size: int, data_fraction: float):
    """Build a DataLoader for the unlabeled STL-10 subset."""
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    dataset = STL10(
        str(data_dir),
        split="unlabeled",
        transform=transform,
        download=False,
    )

    if data_fraction < 1.0:
        n = int(len(dataset) * data_fraction)
        dataset = Subset(dataset, range(n))

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return loader


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    data_dir = Path("data")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Logging + Checkpoint setup
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

    resume_path = args.resume_from
    if resume_path is not None:
        print(f"🔁 Resuming from checkpoint: {resume_path}")

    # ------------------------------
    # Data and model setup
    # ------------------------------
    train_loader = get_dataloader(data_dir, args.batch_size, args.data_fraction)

    model = MAEPretrainModule(
        mask_ratio=args.mask_ratio,
        base_learning_rate=args.base_learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.total_epochs,
        batch_size=args.batch_size,
    )

    # ------------------------------
    # Trainer
    # ------------------------------
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.total_epochs,
        default_root_dir=str(output_dir),
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
    )

    trainer.fit(model, train_loader, ckpt_path=resume_path)

    # ------------------------------
    # Save final weights
    # ------------------------------
    save_path = output_dir / args.model_path
    torch.save(model.model.state_dict(), save_path)
    print(f"\n✅ Pretraining finished. Model saved to: {save_path}")
    print(f"📈 TensorBoard logs available at: {tb_logger.log_dir}")
    print(f"💾 Checkpoints saved in: {checkpoint_callback.dirpath}")


if __name__ == "__main__":
    main()
