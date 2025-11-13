from __future__ import print_function

import argparse
import yaml
from pathlib import Path
import numpy as np
import torch

from src.models.baseline.cnn import CNN
from src.data import get_train_dataloaders


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a simple CNN with PyTorch Lightning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    """Loads a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main training and evaluation script."""
    args = parse_args()
    config = load_config(args.config)

    # --- Data Loading ---
    train_loader, val_loader = get_train_dataloaders(config)
    if train_loader is None or val_loader is None:
        print("Error: Data loaders could not be created.")
        return

    use_cuda = config["model"]["use_cuda"] and torch.cuda.is_available()
    seed = config["model"]["seed"]

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Handel GPU stochasticity
    torch.backends.cudnn.enabled = use_cuda
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if use_cuda else "cpu")

    # --- Model Initialization ---
    model = CNN(
        input_dim=config["model"]["input_dim"],
        num_classes=config["model"]["num_classes"],
        lr=config["model"]["learning_rate"],
        optimizer_type=config["model"]["optimizer"],
        weight_decay=config["model"]["weight_decay"],
        device=device,
        momentum=config["model"]["momentum"],
    )
    model.to(device)
    train_losses, val_losses, train_acc, val_acc = model.fit(
        train_loader=train_loader,
        nr_epochs=config["train"]["total_epochs"],
        val_loader=val_loader,
        verbose=config["train"]["verbose"],
        print_interval=config["train"]["print_interval"],
    )
    model.plotTrace(
        train_losses,
        val_losses,
        train_acc,
        val_acc,
        config["model"]["name"],
        config["train"]["total_epochs"],
    )
    print(f"Validation Accuracy: {val_acc[-1]}")

    print("Training finished.")


if __name__ == "__main__":
    main()
