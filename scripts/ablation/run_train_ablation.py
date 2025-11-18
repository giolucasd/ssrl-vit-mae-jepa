#!/usr/bin/env python3
"""Ablation study runner for MAE downstream fine-tuning experiments.

This script orchestrates a comprehensive ablation study by:
1. Testing multiple pretraining fractions (25%, 50%, 75%, 100%)
2. Varying labeled samples per class (10 to 400)
3. Comparing different fine-tuning strategies (frozen, partial unfreeze, full)

Each training stage builds upon the best checkpoint from the previous stage.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import yaml

BASE_CONFIG: Path = Path("configs/mae.yaml")
TRAIN_SCRIPT: str = "scripts/train_mae.py"

# Pretraining fractions with their checkpoint identifiers
PRETRAIN_FRACTIONS: dict[int, str] = {
    100: "mae_100",
    75: "mae_075",
    50: "mae_050",
    25: "mae_025",
}

# Number of labeled samples per class for downstream evaluation
LABELS_PER_CLASS: list[int] = [400, 300, 200, 100, 50, 25, 10]

# Fine-tuning strategies: (mode_name, freeze_encoder, unfreeze_last_layers, learning_rate)
TRAIN_MODES: list[tuple[str, bool, int | None, float]] = [
    ("frozen", True, None, 3e-4),
    ("unfreeze1", False, 1, 1e-4),
    ("unfreeze2", False, 2, 5e-5),
    ("full", False, None, 5e-5),
]


def run(cmd: list[str]) -> None:
    """Execute a shell command and handle errors.

    Args:
        cmd: Command and arguments as a list of strings.

    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    print("\n🚀 Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    """Execute the full ablation study pipeline."""
    print("===============================================")
    print("🔥 MAE Downstream Ablation Runner")
    print("===============================================")

    base_cfg: dict = yaml.safe_load(BASE_CONFIG.read_text())

    for frac, frac_name in PRETRAIN_FRACTIONS.items():
        print("\n\n=============================")
        print(f"📦 PRETRAIN FRACTION = {frac}%")
        print("=============================\n")

        for labels in LABELS_PER_CLASS:
            print(f"\n----- 🎯 LABELS PER CLASS = {labels} -----")

            for mode_name, freeze_flag, unfreeze_layers, learning_rate in TRAIN_MODES:
                # Determine encoder checkpoint based on fine-tuning stage
                if mode_name == "frozen":
                    encoder_ckpt = Path(
                        f"outputs/pretrain/{frac_name}/checkpoints/best.ckpt"
                    )
                elif mode_name == "unfreeze1":
                    encoder_ckpt = Path(
                        f"outputs/train/{frac_name}_{labels}_frozen/checkpoints/best.ckpt"
                    )
                elif mode_name == "unfreeze2":
                    encoder_ckpt = Path(
                        f"outputs/train/{frac_name}_{labels}_unfreeze1/checkpoints/best.ckpt"
                    )
                elif mode_name == "full":
                    encoder_ckpt = Path(
                        f"outputs/train/{frac_name}_{labels}_unfreeze2/checkpoints/best.ckpt"
                    )
                else:
                    raise ValueError(f"Unknown training mode: {mode_name}")

                if not encoder_ckpt.exists():
                    print(f"❌ ERROR: Missing checkpoint at: {encoder_ckpt}")
                    continue

                suffix: str = f"{frac_name}_{labels}_{mode_name}"
                outdir: Path = Path("outputs/train") / suffix
                ckpt_best: Path = outdir / "checkpoints" / "best.ckpt"

                # Skip experiments that have already been completed
                if ckpt_best.exists():
                    print(f"⏩ SKIPPING {suffix} (already done)")
                    continue

                # Create and configure experiment settings
                cfg: dict = base_cfg.copy()
                cfg["train"]["samples_per_class"] = labels
                cfg["train"]["freeze_encoder"] = freeze_flag
                cfg["train"]["learning_rate"] = learning_rate

                if (
                    not freeze_flag
                    and unfreeze_layers is not None
                    and mode_name != "full"
                ):
                    cfg["train"]["unfreeze_last_layers"] = unfreeze_layers
                else:
                    cfg["train"].pop("unfreeze_last_layers", None)

                # Write temporary configuration file
                tmp_cfg_path: Path = Path(f"configs/tmp_{suffix}.yaml")
                with open(tmp_cfg_path, "w") as f:
                    yaml.safe_dump(cfg, f)

                print(f"\n📝 Created config {tmp_cfg_path}")

                # Build training command
                ckpt_arg = (
                    "--encoder_ckpt" if mode_name == "frozen" else "--classifier_ckpt"
                )
                cmd: list[str] = [
                    "python",
                    "-m",
                    "scripts.training.train_mae",
                    "--config",
                    str(tmp_cfg_path),
                    ckpt_arg,
                    str(encoder_ckpt),
                    "--output_dir_suffix",
                    suffix,
                ]

                # Execute training
                try:
                    run(cmd)
                except subprocess.CalledProcessError:
                    print(f"❌ ERROR during training: {suffix}")
                    continue

                # Brief pause to avoid filesystem race conditions
                time.sleep(2)

                # Cleanup temporary configuration
                tmp_cfg_path.unlink(missing_ok=True)

    print("\n\n===============================================")
    print("🎉 ALL DOWNSTREAM ABLATIONS COMPLETED!")
    print("===============================================")


if __name__ == "__main__":
    main()
