from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import get_test_dataloader
from src.training.classifier import ViTClassifierTrainModule


def setup_reproducibility(seed: int) -> None:
    """Sets up reproducibility by fixing random seeds and configuring deterministic behavior.

    Args:
        seed (int): The seed value to use for random number generators.
    """
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")


def shut_down_warnings() -> None:
    """Suppresses specific user warnings related to mixed precision and TF32 behavior."""
    import warnings

    warnings.filterwarnings(
        "ignore",
        "Precision bf16-mixed is not supported by the model summary",
    )

    warnings.filterwarnings(
        "ignore",
        "Please use the new API settings to control TF32 behavior",
    )


def evaluate_checkpoint(cfg: dict, checkpoint_path: str | Path):
    """
    Reusable evaluation helper.
    Loads model, loads test dataloader, runs evaluation, returns test accuracy.
    """
    test_cfg = cfg["test"]
    log_cfg = cfg["logging"]

    # ------------------------------
    # Data
    # ------------------------------
    test_loader = get_test_dataloader(cfg)

    # ------------------------------
    # Load model
    # ------------------------------
    print(f"🔁 Loading model from checkpoint: {checkpoint_path}")
    try:
        module = ViTClassifierTrainModule.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            strict=False,
        )
    except KeyError:
        print("📦 Detected pure PyTorch checkpoint (.pt)")

        module = ViTClassifierTrainModule(
            pretrained_encoder=None,
            model_cfg=cfg.get("model", {}),
            training_cfg=cfg.get("training", {}),
        )

        ckpt: dict = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        _, _ = module.model.load_state_dict(state_dict, strict=False)
    else:
        print("📦 Detected Lightning checkpoint (.ckpt)")

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
    results = trainer.test(module, test_loader, ckpt_path=None)

    acc = results[0].get("test_acc", None)
    print(f"🔎 Test Accuracy: {acc}")

    return acc
