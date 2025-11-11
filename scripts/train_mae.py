from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import STL10

from src.models.mae.encoder import MAEEncoder
from src.utils.train_module import MAETrainModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MAE encoder on STL-10 classification"
    )
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        required=True,
        help="Path to pretrained MAE encoder weights",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument(
        "--freeze_encoder",
        type=bool,
        default=True,
        help="Freeze encoder weights during training",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=400,
        choices=[10, 25, 50, 100, 200, 300, 400],
        help="Number of labeled samples per class to use for training. Remaining samples are used for validation.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="outputs/train")
    return parser.parse_args()


def get_dataloaders(
    data_dir: Path,
    output_dir: Path,
    batch_size: int,
    samples_per_class: int | None = None,
):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize(96),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    full_train_ds = STL10(
        str(data_dir),
        split="train",
        transform=transform_train,
        download=False,
    )

    # ===============================
    # Subsample labeled examples per class
    # ===============================
    if samples_per_class is not None:
        labels = np.array(full_train_ds.labels)
        indices_per_class = {
            cls: np.where(labels == cls)[0] for cls in np.unique(labels)
        }

        selected_indices = []
        val_indices = []

        for cls, idxs in indices_per_class.items():
            np.random.shuffle(idxs)
            n = min(samples_per_class, len(idxs))
            selected_indices.extend(idxs[:n])
            val_indices.extend(idxs[n:])  # leftover for validation

        train_ds = Subset(full_train_ds, selected_indices)
        val_ds = Subset(full_train_ds, val_indices)

        print(
            f"⚙️ Using {samples_per_class} samples per class "
            f"→ {len(selected_indices)} total training samples, "
            f"{len(val_indices)} for validation."
        )

        with open(output_dir / "train_indices.json", "w") as f:
            json.dump([int(i) for i in selected_indices], f)

    else:
        # Default: use full train for training, test for validation
        train_ds = full_train_ds
        val_ds = STL10(
            str(data_dir),
            split="test",
            transform=transform_val,
            download=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained encoder
    encoder = MAEEncoder()
    ckpt = torch.load(args.encoder_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"]

    encoder_state_dict = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }

    missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=False)
    print(
        f"Loaded encoder weights: {len(encoder_state_dict)} params "
        f"({len(missing)} missing, {len(unexpected)} unexpected)"
    )

    # Init Lightning module
    module = MAETrainModule(
        pretrained_encoder=encoder,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        freeze_encoder=args.freeze_encoder,
    )

    # Dataloaders
    data_dir = Path("data")
    train_loader, val_loader = get_dataloaders(
        data_dir,
        output_dir,
        args.batch_size,
        samples_per_class=args.samples_per_class,
    )

    # Callbacks (logging + checkpoint)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=str(output_dir), name="logs")
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        save_top_k=1,
        monitor="val_acc",
        mode="max",
        filename="best-valacc-{epoch:03d}-{val_acc:.4f}",
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        logger=tb_logger,
        callbacks=[ckpt_callback],
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        log_every_n_steps=10,
    )

    trainer.fit(module, train_loader, val_loader)

    print(
        f"\n✅ Training finished. Best model saved to: {ckpt_callback.best_model_path}"
    )


if __name__ == "__main__":
    main()
