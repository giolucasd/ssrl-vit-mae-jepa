#!/usr/bin/env python3
"""
Runs baseline ablation: trains a randomly initialized ViT
(no MAE, no encoder checkpoint) for multiple label budgets.

Uses scripts.training.train_mae and relies on the fact that
NOT passing --encoder_ckpt or --classifier_ckpt triggers
random initialization + full fine-tuning.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

BUDGETS = [10, 25, 50, 100, 200, 300, 400]


def run_cmd(cmd: list[str]):
    print("\n" + "=" * 80)
    print("📣 Running command:")
    print(" ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def main():
    config_path = Path("configs/mae.yaml")
    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    for k in BUDGETS:
        print(f"\n\n🚀 Starting BASELINE run with {k} samples per class\n")

        # Create a temporary config with modified train.samples_per_class
        cfg = base_cfg.copy()
        cfg["train"] = cfg["train"].copy()
        cfg["train"]["samples_per_class"] = k

        # Output directory
        suffix = f"mae_000_{k}"
        out_dir = Path("outputs") / "train" / suffix
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save modified config snapshot for this run
        cfg_path = out_dir / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        print(f"📝 Saved config for this run: {cfg_path}")

        # Call train_mae without encoder/classifier ckpt → baseline flow
        cmd = [
            "python",
            "-m",
            "scripts.training.train_mae",
            "--config",
            str(cfg_path),
            "--output_dir_suffix",
            suffix,
            # NOTE: no --encoder_ckpt or --classifier_ckpt -> random ViT
        ]

        run_cmd(cmd)

    print("\n🎉 Baseline ablation complete!")


if __name__ == "__main__":
    main()
