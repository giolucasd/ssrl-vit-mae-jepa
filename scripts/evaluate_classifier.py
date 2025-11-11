import argparse

import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils.classifier_module import MAEClassifierModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned MAE classifier on STL-10 test set."
    )
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the trained classifier checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--classifier_ckpt",
        type=str,
        default=None,
        help="Optional path to a full classifier checkpoint for fine-tuning.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def get_test_loader(batch_size: int, num_workers: int):
    transform = transforms.Compose(
        [
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    test_dataset = datasets.STL10(
        root="data", split="test", download=True, transform=transform
    )
    return DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )


def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro"
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    print(f"Loading checkpoint from {args.ckpt_path}")

    model = MAEClassifierModule.load_from_checkpoint(args.ckpt_path)
    dataloader = get_test_loader(args.batch_size, args.num_workers)

    metrics = evaluate(model, dataloader, args.device)

    print("\n=== Evaluation Results (STL-10 test set) ===")
    for k, v in metrics.items():
        print(f"{k.capitalize():10s}: {v:.4f}")


if __name__ == "__main__":
    main()
