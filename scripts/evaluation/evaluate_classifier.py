from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data import get_test_dataloader

from ..utils import evaluate_checkpoint, setup_reproducibility, shut_down_warnings

shut_down_warnings()
setup_reproducibility(seed=73)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MAE classifier on STL-10 test set"
    )
    parser.add_argument("--config", type=str, default="configs/mae.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (.ckpt) — defaults to best.ckpt in training output dir",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------
    # Load configuration
    # ------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    log_cfg = cfg["logging"]
    train_cfg = cfg["train"]

    # ------------------------------
    # Locate checkpoint
    # ------------------------------
    if args.checkpoint is None:
        default_ckpt = (
            Path(log_cfg["output_dir_base"])
            / "train"
            / train_cfg.get("output_dir_suffix", "default")
            / "checkpoints"
            / "best.ckpt"
        )
        args.checkpoint = str(default_ckpt)
        print(f"🧩 Using default checkpoint: {args.checkpoint}")

    # ------------------------------
    # Data
    # ------------------------------
    test_loader = get_test_dataloader(cfg)

    # ------------------------------
    # Evaluate
    # ------------------------------
    acc = evaluate_checkpoint(cfg, args.checkpoint, test_loader)

    # ------------------------------
    # Output
    # ------------------------------
    print("\n✅ Evaluation complete")
    print(f"📈 Accuracy: {acc}")


if __name__ == "__main__":
    main()
