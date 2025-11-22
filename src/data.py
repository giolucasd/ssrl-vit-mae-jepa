from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import STL10

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


def _build_transform(train: bool = True) -> transforms.Compose:
    """Standard STL-10 augmentations."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
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


def get_pretrain_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """
    Builds train/val dataloaders for self-supervised pretraining (unlabeled split).

    Args:
        cfg (dict): configuration dictionary with keys:
            - batch_size (int)
            - val_split (float)
            - data_fraction (float)
            - seed (int)
            - num_workers (int)
    """
    pre_cfg = cfg["pretrain"]
    seed = cfg.get("seed", 73)

    full_dataset = STL10(
        DATA_DIR,
        split="unlabeled",
        transform=_build_transform(train=True),
        download=True,
    )

    # Optionally subsample dataset
    full_dataset = _subset_dataset(full_dataset, pre_cfg.get("data_fraction", 1.0))

    # Split into train/val subsets
    n_total = len(full_dataset)
    val_split = pre_cfg.get("val_split", 0.1)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_subset, val_subset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    val_subset.dataset.transform = _build_transform(train=False)

    batch_size = pre_cfg.get("batch_size", 512)
    num_workers = pre_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"📦 Unlabeled pretrain split: {n_train} train, {n_val} val "
        f"({val_split * 100:.1f}% validation)"
    )

    return train_loader, val_loader


def get_train_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """
    Builds train/val dataloaders for fine-tuning on labeled STL-10 split.
    The training set is sampled with `samples_per_class`, and the remaining
    labeled data becomes the validation set.

    Args:
        cfg (dict): configuration dictionary with keys:
            - batch_size (int)
            - samples_per_class (int)
            - seed (int)
            - num_workers (int)
    """
    train_cfg = cfg["train"]
    seed = cfg.get("seed", 73)

    full_dataset = STL10(DATA_DIR, split="train", transform=_build_transform(True))
    labels = np.array(full_dataset.labels)

    samples_per_class = train_cfg.get("samples_per_class", 400)
    train_indices, val_indices = [], []

    for c in np.unique(labels):
        cls_idx = np.where(labels == c)[0]
        np.random.default_rng(seed).shuffle(cls_idx)
        train_indices.extend(cls_idx[:samples_per_class])
        val_indices.extend(cls_idx[samples_per_class:])

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    val_dataset.dataset.transform = _build_transform(train=False)

    print(
        f"⚙️ Using {samples_per_class} samples/class → {len(train_indices)} train, "
        f"{len(val_indices)} val"
    )

    batch_size = train_cfg.get("batch_size", 256)
    num_workers = train_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(cfg: dict) -> DataLoader:
    """Returns only the STL-10 test split for final evaluation."""
    test_cfg = cfg.get("test", {})
    batch_size = test_cfg.get("batch_size", 256)
    num_workers = test_cfg.get("num_workers", 4)

    test_dataset = STL10(DATA_DIR, split="test", transform=_build_transform(False))

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"🧪 Loaded STL-10 test split: {len(test_dataset)} samples")

    return test_loader
