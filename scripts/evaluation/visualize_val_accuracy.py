import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def extract_accuracy_from_checkpoint(ckpt_path):
    """Extract validation accuracy from a checkpoint file."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # Try different possible keys where accuracy might be stored
        possible_keys = ["val_acc", "val_accuracy", "best_val_acc"]

        for key in possible_keys:
            if key in ckpt:
                return float(ckpt[key])

        # If not found in direct keys, try callbacks
        if "callbacks" in ckpt:
            callbacks = ckpt["callbacks"]
            for callback_name, callback_data in callbacks.items():
                if "best_model_score" in callback_data:
                    return float(callback_data["best_model_score"])
                if "best_score" in callback_data:
                    return float(callback_data["best_score"])

        print(f"Warning: Could not find accuracy in {ckpt_path}")
        return None

    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")
        return None


def parse_filename(filename):
    """Parse filename to extract pretrain percentage, labels per class, and unfreeze layer."""
    # Extract numbers from filename like mae_100_400_8.ckpt
    match = re.match(r"mae_(\d+)_(\d+)_(\w+)\.ckpt", filename)

    if match:
        pretrain_pct = int(match.group(1))
        labels_per_class = int(match.group(2))
        unfreeze_layer = match.group(3)
        return pretrain_pct, labels_per_class, unfreeze_layer
    return None, None, None


def get_all_checkpoint_files(weights_dir):
    """Get all checkpoint files from weights_dir and its subdirectories."""
    ckpt_files = []
    for ckpt_file in weights_dir.rglob("*.ckpt"):
        ckpt_files.append(ckpt_file)
    return ckpt_files


def create_accuracy_plot():
    """Create a plot showing accuracy vs labels per class for different unfreeze layers, averaging across pretrain percentages."""
    weights_dir = Path("outputs/train/")

    if not weights_dir.exists():
        print(f"Error: {weights_dir} does not exist")
        return

    # Dictionary to store results: {(labels_per_class, unfreeze_layer): [accuracies]}
    results = {}

    # Scan all checkpoint files
    for ckpt_file in get_all_checkpoint_files(weights_dir):
        pretrain_pct, labels_per_class, unfreeze_layer = parse_filename(ckpt_file.name)

        if all(x is not None for x in [pretrain_pct, labels_per_class, unfreeze_layer]):
            accuracy = extract_accuracy_from_checkpoint(ckpt_file)

            if accuracy is not None:
                config_key = (labels_per_class, unfreeze_layer)
                if config_key not in results:
                    results[config_key] = []
                results[config_key].append(accuracy)

    if not results:
        print("No valid checkpoint files found or no accuracy data extracted")
        return

    # Calculate mean accuracies
    mean_acc_results = {}
    for config_key, accuracies in results.items():
        mean_accuracy = np.mean(accuracies)
        mean_acc_results[config_key] = mean_accuracy

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Define colors and markers for different unfreeze layers
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    # Group results by unfreeze layer
    unfreeze_results = {}
    for (labels_per_class, unfreeze_layer), mean_accuracy in mean_acc_results.items():
        if unfreeze_layer not in unfreeze_results:
            unfreeze_results[unfreeze_layer] = {}
        unfreeze_results[unfreeze_layer][labels_per_class] = mean_accuracy

    # Get sorted list of unfreeze layers
    unfreeze_layers = sorted(unfreeze_results.keys())

    for i, unfreeze_layer in enumerate(unfreeze_layers):
        labels_per_class_list = sorted(unfreeze_results[unfreeze_layer].keys())
        mean_accuracies = [
            unfreeze_results[unfreeze_layer][lpc] for lpc in labels_per_class_list
        ]

        plt.plot(
            labels_per_class_list,
            mean_accuracies,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=8,
            label=f"Unfreeze layer {unfreeze_layer}",
        )

    plt.xlabel("Labels per Class", fontsize=12)
    plt.ylabel("Mean Accuracy", fontsize=12)
    plt.title(
        "Mean Model Accuracy vs Labels per Class\nfor Different Unfreeze Layer Configurations\n(Averaged across Pretrain Percentages)",
        fontsize=14,
    )
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Format y-axis as percentage if needed
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.1%}" if y <= 1 else f"{y:.1f}")
    )

    plt.tight_layout()

    # Save the plot
    output_path = "assets/visualizations/val_accuracy_comparison_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    create_accuracy_plot()
