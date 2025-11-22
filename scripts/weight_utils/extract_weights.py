#!/usr/bin/env python3
from pathlib import Path

import torch

from src.training.classifier import ViTClassifierTrainModule

SOURCE_DIR = Path("assets/weights")
TARGET_DIR = SOURCE_DIR / "pt"


def extract_pytorch_weights(ckpt_path: Path, target_path: Path):
    print(f"🔍 Loading lightning module from checkpoint: {ckpt_path}")

    # Carrega corretamente o LightningModule
    module = ViTClassifierTrainModule.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        strict=False,
    )

    # Recupera somente os pesos do modelo (Vit + head)
    state = module.model.state_dict()

    print(f"   → Extracted {len(state)} tensors")

    # Salva como .pt puro
    torch.save(state, target_path)

    print(f"💾 Saved PyTorch weights → {target_path}\n")


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_files = sorted(SOURCE_DIR.glob("*.ckpt"))

    if not ckpt_files:
        print("❌ No .ckpt files found in assets/weights/")
        return

    print(f"🔎 Found {len(ckpt_files)} checkpoints")

    for ckpt in ckpt_files:
        new_name = ckpt.stem + ".pt"
        out_path = TARGET_DIR / new_name
        extract_pytorch_weights(ckpt, out_path)

    print("✅ All weights successfully converted!")


if __name__ == "__main__":
    main()
