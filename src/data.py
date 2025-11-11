# src/data.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import STL10


def build_transform(train: bool = True) -> transforms.Compose:
    """Data augmentations for STL-10."""
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


def get_stl10_dataloader(
    split: Literal["unlabeled", "train", "test"] = "unlabeled",
    data_dir: str | Path = "data",
    batch_size: int = 512,
    data_fraction: float = 1.0,
    num_workers: int = 4,
    train: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader for STL-10 (supports unlabeled, train, test)."""
    transform = build_transform(train=train)

    dataset = STL10(str(data_dir), split=split, transform=transform, download=True)

    if data_fraction < 1.0:
        n = int(len(dataset) * data_fraction)
        dataset = Subset(dataset, range(n))

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def get_pretrain_dataloader(batch_size: int, data_fraction: float = 1.0) -> DataLoader:
    """Unlabeled STL-10 for MAE pretraining."""
    return get_stl10_dataloader(
        split="unlabeled",
        batch_size=batch_size,
        data_fraction=data_fraction,
        train=True,
    )


def get_finetune_dataloaders(batch_size: int, samples_per_class: Optional[int] = None):
    """Supervised dataloaders for fine-tuning (train/test)."""
    train_loader = get_stl10_dataloader(
        split="train",
        batch_size=batch_size,
        train=True,
        shuffle=True,
    )
    test_loader = get_stl10_dataloader(
        split="test",
        batch_size=batch_size,
        train=False,
        shuffle=False,
    )

    # Optionally limit number of labeled samples per class
    if samples_per_class is not None:
        # naive class balancing subset
        targets = torch.tensor(train_loader.dataset.labels)
        indices = torch.cat(
            [
                (targets == c).nonzero(as_tuple=True)[0][:samples_per_class]
                for c in range(10)
            ]
        )
        subset = Subset(train_loader.dataset, indices)
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
