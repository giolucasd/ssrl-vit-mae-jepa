# src/data.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import STL10


# ============================================================
# 🔧 Helpers
# ============================================================
def _build_transform(train: bool = True) -> transforms.Compose:
    """Standard STL-10 augmentations."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(96),
                transforms.CenterCrop(96),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )


def _subset_dataset(dataset, fraction: float) -> Subset | STL10:
    """Optionally take a fraction of a dataset."""
    if fraction < 1.0:
        n = int(len(dataset) * fraction)
        return Subset(dataset, range(n))
    return dataset


# ============================================================
# 🧩 Dataloaders
# ============================================================
def get_stl10_dataloader(
    split: Literal["unlabeled", "train", "test"],
    data_dir: str | Path = "data",
    batch_size: int = 512,
    train: bool = True,
    data_fraction: float = 1.0,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """General STL-10 DataLoader factory."""
    transform = _build_transform(train=train)
    dataset = STL10(str(data_dir), split=split, transform=transform, download=True)
    dataset = _subset_dataset(dataset, data_fraction)

    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_pretrain_dataloader(
    batch_size: int,
    data_fraction: float = 1.0,
    data_dir: str | Path = "data",
) -> DataLoader:
    """Unlabeled STL-10 loader for MAE pretraining."""
    return get_stl10_dataloader(
        split="unlabeled",
        data_dir=data_dir,
        batch_size=batch_size,
        train=True,
        data_fraction=data_fraction,
    )


def get_train_val_dataloaders(
    batch_size: int,
    samples_per_class: Optional[int] = None,
    data_dir: str | Path = "data",
) -> tuple[DataLoader, DataLoader]:
    """Supervised STL-10 dataloaders for MAE fine-tuning."""
    train_dataset = STL10(
        str(data_dir), split="train", transform=_build_transform(True)
    )
    val_dataset = STL10(str(data_dir), split="test", transform=_build_transform(False))

    # Optionally limit samples per class (semi-supervised setting)
    if samples_per_class is not None:
        labels = np.array(train_dataset.labels)
        train_indices = []
        for c in np.unique(labels):
            cls_idx = np.where(labels == c)[0]
            np.random.shuffle(cls_idx)
            train_indices.extend(cls_idx[:samples_per_class])
        train_dataset = Subset(train_dataset, train_indices)
        print(
            f"⚙️ Using {samples_per_class} samples/class → {len(train_indices)} train samples"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader
