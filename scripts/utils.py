from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from timm.models.vision_transformer import VisionTransformer

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


def load_vit_classifier_from_checkpoint(
    model_cfg: dict,
    training_cfg: dict,
    checkpoint_path: str | Path | None = None,
    encoder_only: bool = False,
) -> LightningModule:
    """
    Loads a ViTClassifierTrainModule from either:
        - Lightning checkpoint (.ckpt)
        - Pure PyTorch state_dict (.pt)
        - Encoder-only PyTorch checkpoint (.pt)
    Or returns a randomly initialized model when checkpoint_path=None.

    Args:
        checkpoint_path: path to .ckpt or .pt
        model_cfg, training_cfg: YAML configs
        encoder_only: if True, load only encoder weights

    Returns:
        ViTClassifierTrainModule
    """
    print(f"🔁 Loading ViTClassifierTrainModule from checkpoint: {checkpoint_path}")

    if checkpoint_path is None:
        encoder_cfg = model_cfg["encoder"]
        general = model_cfg["general"]

        encoder = VisionTransformer(
            img_size=general["image_size"],
            patch_size=general["patch_size"],
            in_chans=general["in_chans"],
            embed_dim=encoder_cfg["embed_dim"],
            depth=encoder_cfg["depth"],
            num_heads=encoder_cfg["num_heads"],
            num_classes=0,
        )

        print("🔧 Lightning module ahs been randomly initialized")
        return ViTClassifierTrainModule(
            pretrained_encoder=encoder,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
        )

    checkpoint_path = Path(checkpoint_path)
    is_ckpt = checkpoint_path.suffix == ".ckpt"

    if is_ckpt and not encoder_only:
        print("🔧 Loaded from lightning checkpoint")
        return ViTClassifierTrainModule.load_from_checkpoint(
            checkpoint_path,
            strict=False,
            map_location="cpu",
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    if encoder_only:
        general = model_cfg["general"]
        enc_cfg = model_cfg["encoder"]

        encoder = VisionTransformer(
            img_size=general["image_size"],
            patch_size=general["patch_size"],
            in_chans=general["in_chans"],
            embed_dim=enc_cfg["embed_dim"],
            depth=enc_cfg["depth"],
            num_heads=enc_cfg["num_heads"],
            num_classes=0,  # encoder only
        )

        possible_prefixes = ["model.encoder.", "encoder.", "module.encoder."]

        prefix = next(
            (p for p in possible_prefixes if any(k.startswith(p) for k in state_dict)),
            None,
        )
        if prefix is None:
            raise ValueError(
                "❌ Could not find encoder weights in PT checkpoint. "
                f"Tried prefixes: {possible_prefixes}"
            )

        encoder_state = {
            k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
        }

        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)

        print(
            f"🔧 Loaded encoder-only weights "
            f"({len(encoder_state)} tensors, {len(missing)} missing, {len(unexpected)} unexpected)"
        )
        return ViTClassifierTrainModule(
            pretrained_encoder=encoder,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
        )

    module = ViTClassifierTrainModule(
        pretrained_encoder=None,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
    )
    module.model.load_state_dict(state_dict, strict=False)

    print("🔧 Loaded full classifier weights")
    return module


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
    module = load_vit_classifier_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_cfg=cfg["model"],
        training_cfg=cfg["train"],
        encoder_only=False,
    )

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
