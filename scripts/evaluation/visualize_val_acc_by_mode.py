import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..utils import setup_reproducibility, shut_down_warnings

shut_down_warnings()
setup_reproducibility(seed=73)


# -------------------------------------------------------------
# Extract accuracy from checkpoint
# -------------------------------------------------------------
def extract_accuracy_from_checkpoint(ckpt_path):
    """Extract validation accuracy from a Lightning checkpoint."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # 1. Lightning stores best score inside callbacks
        if "callbacks" in ckpt:
            for cb in ckpt["callbacks"].values():
                if "best_model_score" in cb:
                    return float(cb["best_model_score"])

        # 2. Sometimes stored directly
        for key in ["val_acc", "val_accuracy", "best_val_acc"]:
            if key in ckpt:
                return float(ckpt[key])

        print(f"⚠️  No accuracy found in: {ckpt_path}")
        return None

    except Exception as e:
        print(f"❌ Error loading {ckpt_path}: {e}")
        return None


# -------------------------------------------------------------
# Parse directory name: mae_<pretrain>_<labels>_<mode>
# -------------------------------------------------------------
def parse_dirname(dirname):
    """
    Expects something like:
    mae_100_400_frozen
    mae_075_25_unfreeze1
    mae_025_300_full
    """
    match = re.match(r"mae_(\d+)_(\d+)_(\w+)", dirname)
    if match:
        pretrain_pct = int(match.group(1))
        labels_per_class = int(match.group(2))
        mode = match.group(3)  # frozen / unfreeze1 / unfreeze2 / full
        return pretrain_pct, labels_per_class, mode
    return None, None, None


# -------------------------------------------------------------
# Scan all best.ckpt files
# -------------------------------------------------------------
def get_best_checkpoints(root):
    """Find all directories containing checkpoints/best.ckpt."""
    best_files = list(root.glob("**/checkpoints/best.ckpt"))
    if not best_files:
        print("❌ No best.ckpt files found.")
    return best_files


# -------------------------------------------------------------
# Main – create plot
# -------------------------------------------------------------
def create_accuracy_plot():
    weights_dir = Path("outputs/train")

    ckpt_files = get_best_checkpoints(weights_dir)

    # (labels_per_class, mode) -> list of accuracies
    results = {}

    for ckpt_path in ckpt_files:
        dirname = ckpt_path.parent.parent.name  # e.g. mae_100_400_frozen
        pretrain_pct, labels_per_class, mode = parse_dirname(dirname)

        if None in (pretrain_pct, labels_per_class, mode):
            print(f"⚠️  Skipping unrecognized directory: {dirname}")
            continue

        acc = extract_accuracy_from_checkpoint(ckpt_path)
        if acc is None:
            continue

        key = (labels_per_class, mode)
        results.setdefault(key, []).append(acc)

    if not results:
        print("❌ No valid accuracy data found.")
        return

    # Compute means
    mean_results = {key: float(np.mean(vals)) for key, vals in results.items()}

    # Organize by mode
    modes = ["frozen", "unfreeze1", "unfreeze2", "full"]
    mode_to_data = {m: {} for m in modes}

    for (labels, mode), acc in mean_results.items():
        if mode not in mode_to_data:
            mode_to_data[mode] = {}
        mode_to_data[mode][labels] = acc

    # Plot
    plt.figure(figsize=(12, 8))

    colors = ["blue", "red", "green", "orange"]
    markers = ["o", "s", "^", "D"]

    for i, mode in enumerate(modes):
        if not mode_to_data[mode]:
            continue

        labels_sorted = sorted(mode_to_data[mode].keys())
        accs = [mode_to_data[mode][l] for l in labels_sorted]

        plt.plot(
            labels_sorted,
            accs,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=8,
            label=mode,
        )

    plt.xlabel("Labels per Class", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.title(
        "Validation Accuracy vs Labels per Class\n(Averaged Across Pretraining Fractions)",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    out = Path("assets/visualizations/val_accuracy_by_mode.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    print(f"📈 Plot saved to: {out}")


if __name__ == "__main__":
    create_accuracy_plot()
