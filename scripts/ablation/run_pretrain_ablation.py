# scripts/run_pretrain_ablation.py
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import yaml

BASE_CONFIG = Path("configs/mae.yaml")

FRACTIONS = {
    "025": 0.25,
    "050": 0.50,
    "075": 0.75,
    "100": 1.00,
}


def main():
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"Base config not found: {BASE_CONFIG}")

    # Load base config
    with open(BASE_CONFIG, "r") as f:
        base_cfg = yaml.safe_load(f)

    for suffix, frac in FRACTIONS.items():
        print("\n" + "=" * 80)
        print(f"🚀 Starting pretrain run for {int(frac * 100)}% unlabeled data")
        print("=" * 80 + "\n")

        # Create modified config
        cfg = base_cfg.copy()

        # Modify fields
        cfg["pretrain"]["data_fraction"] = float(frac)

        # Output file
        output_dir_suffix = f"mae_{suffix}"
        cfg_path = Path(f"configs/mae_{suffix}.yaml")

        # Save new config
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        print(f"📝 Saved modified config: {cfg_path}")

        # Define output directory for this pretraining
        output_dir = (
            Path(base_cfg["logging"]["output_dir_base"])
            / "pretrain"
            / output_dir_suffix
        )

        # Skip if results already exist (avoid re-running)
        if (output_dir / "checkpoints" / "best.ckpt").exists():
            print(f"⏭️ Existing checkpoint found at {output_dir}, skipping...\n")
            continue

        # Run the training command
        cmd = [
            "python",
            "-m",
            "scripts.training.pretrain_mae",
            "--config",
            str(cfg_path),
            "--output_dir_suffix",
            output_dir_suffix,
        ]

        print(f"💻 Running command: {' '.join(cmd)}\n")

        # Call the subprocess (blocking)
        process = subprocess.Popen(cmd)
        process.wait()

        if process.returncode != 0:
            print(f"❌ Training for fraction {frac} failed. Stopping.")
            break

        print(f"✅ Finished pretraining for {int(frac * 100)}% unlabeled data\n")
        time.sleep(3)  # tiny cooldown between runs

    print("\n🎉 All requested pretraining experiments completed!")


if __name__ == "__main__":
    main()
