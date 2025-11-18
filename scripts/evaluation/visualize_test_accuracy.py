import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from ..utils import evaluate_checkpoint


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


def parse_filename(filename):
    """Parse filename to extract pretrain percentage and labels per class."""
    # Extract numbers from filename like mae_100_400.ckpt
    match = re.match(r"mae_(\d+)_(\d+)\.ckpt", filename)
    if match:
        pretrain_pct = int(match.group(1))
        labels_per_class = int(match.group(2))
        return pretrain_pct, labels_per_class
    return None, None


def create_accuracy_plot(cfg):
    """Create a plot showing accuracy vs labels per class for different pretrain percentages."""
    weights_dir = Path("assets/weights")

    if not weights_dir.exists():
        print(f"Error: {weights_dir} does not exist")
        return

    # Dictionary to store results: {pretrain_pct: {labels_per_class: accuracy}}
    results = {}

    # Scan all checkpoint files
    for ckpt_file in weights_dir.glob("mae_*.ckpt"):
        pretrain_pct, labels_per_class = parse_filename(ckpt_file.name)

        if pretrain_pct is not None and labels_per_class is not None:
            accuracy = evaluate_checkpoint(cfg, ckpt_file)

            if accuracy is not None:
                if pretrain_pct not in results:
                    results[pretrain_pct] = {}
                results[pretrain_pct][labels_per_class] = accuracy
                print(
                    f"Found: {pretrain_pct}% pretrain, {labels_per_class} labels/class -> {accuracy:.3f} accuracy"
                )

    if not results:
        print("No valid checkpoint files found or no accuracy data extracted")
        return

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Define colors and markers for different pretrain percentages
    colors = ["blue", "red", "green", "orange"]
    markers = ["o", "s", "^", "D"]

    pretrain_percentages = sorted(results.keys())

    for i, pretrain_pct in enumerate(pretrain_percentages):
        labels_per_class_list = sorted(results[pretrain_pct].keys())
        accuracies = [results[pretrain_pct][lpc] for lpc in labels_per_class_list]

        plt.plot(
            labels_per_class_list,
            accuracies,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=8,
            label=f"{pretrain_pct}% pretraining data",
        )

    plt.xlabel("Labels per Class", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(
        "Model Accuracy vs Labels per Class\nfor Different Pretraining Data Amounts",
        fontsize=14,
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Format y-axis as percentage if needed
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.1%}" if y <= 1 else f"{y:.1f}")
    )

    plt.tight_layout()

    # Save the plot
    output_path = "assets/visualizations/test_accuracy_comparison_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


def main():
    args = parse_args()

    # ------------------------------
    # Load configuration
    # ------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    create_accuracy_plot(cfg)


if __name__ == "__main__":
    main()
