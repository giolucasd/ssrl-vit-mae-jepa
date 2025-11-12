# scripts/evaluate_classifier.py
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import get_test_dataloader
from src.training.mae import MAETrainModule

warnings.filterwarnings(
    "ignore",
    "Precision 16-mixed is not supported",
    category=UserWarning,
)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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

    seed = cfg.get("seed", 73)
    pl.seed_everything(seed, workers=True)

    test_cfg = cfg["test"]
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
    # Load model
    # ------------------------------
    print(f"🔁 Loading model from checkpoint: {args.checkpoint}")
    model = MAETrainModule.load_from_checkpoint(args.checkpoint, strict=False)

    # ------------------------------
    # Logging
    # ------------------------------
    output_dir = (
        Path(log_cfg["output_dir_base"])
        / "test"
        / test_cfg.get("output_dir_suffix", "default")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tb")

    # ------------------------------
    # Trainer
    # ------------------------------
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=tb_logger,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
    )

    # ------------------------------
    # Test
    # ------------------------------
    print("\n🚀 Starting evaluation...")
    results = trainer.test(model, test_loader, ckpt_path=None)

    # ------------------------------
    # Output
    # ------------------------------
    print("\n✅ Evaluation complete")
    print(f"📈 Results: {results[0]}")
    print(f"🧾 Logs saved to: {tb_logger.log_dir}")


if __name__ == "__main__":
    main()
