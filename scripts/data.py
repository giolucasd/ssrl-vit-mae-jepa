"""
Downloads all splits of the STL10 dataset into /data/stl10_binary.

Usage:
    uv run scripts/data.py
"""

from pathlib import Path

from torchvision import transforms
from torchvision.datasets import STL10

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

# Simple transform (not used for downloading, but required for instantiation)
transform = transforms.ToTensor()

# Available STL10 splits
SPLITS = ["train", "test", "unlabeled"]


def download_stl10_splits():
    """Download all STL10 dataset splits."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        print(f"\n📥 Downloading split: '{split}'...")
        dataset = STL10(
            root=str(DATA_DIR),
            split=split,
            download=True,
            transform=transform,
        )
        print(f"✅ Split '{split}' downloaded with {len(dataset)} samples.")

    print("\n🎉 All splits have been downloaded successfully!")

    tar_file = DATA_DIR / "stl10_binary.tar.gz"
    if tar_file.exists():
        tar_file.unlink()
        print("🧹 Removed 'stl10_binary.tar.gz' to save disk space.")

    print(f"Dataset is stored in: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    download_stl10_splits()
