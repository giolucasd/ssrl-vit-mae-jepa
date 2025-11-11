from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import STL10
from tqdm import tqdm

from src.models.mae.encoder import MAEEncoder, build_2d_sincos_position_embedding


def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear probe on pretrained MAE encoder"
    )
    parser.add_argument("--encoder_ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output_dir", type=str, default="outputs/linear_probe")
    return parser.parse_args()


def get_dataloaders(batch_size: int):
    """Simple resize + normalize — same normalization as MAE pretraining."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_set = STL10("data", split="train", transform=transform, download=False)
    test_set = STL10("data", split="test", transform=transform, download=False)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader


def extract_features(encoder: MAEEncoder, dataloader, device):
    """Extract mean features from encoder (no gradient)."""
    encoder.eval()
    feats, labels = [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(device)
            # === forward pass até o último token CLS ===
            patch_embeddings = encoder.patchify(imgs)
            B, C, H, W = patch_embeddings.shape
            tokens = patch_embeddings.flatten(2).permute(2, 0, 1)

            # === Fixed 2D sin-cos positional encoding ===
            num_patches_per_dim = int(
                (imgs.shape[2] // encoder.patchify.kernel_size[0])
            )
            emb_dim = encoder.patchify.out_channels
            pos_embed = build_2d_sincos_position_embedding(
                num_patches_per_dim, emb_dim
            ).to(device)
            tokens = tokens + pos_embed

            cls_token = encoder.cls_token.expand(-1, B, -1)
            tokens = torch.cat([cls_token, tokens], dim=0)
            tokens = tokens.permute(1, 0, 2)  # (B, T, C)

            out = encoder.layer_norm(encoder.transformer(tokens))
            features = out[:, 0, :]  # CLS token feature (B, C)
            feats.append(features.cpu())
            labels.append(lbls)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


def linear_probe(train_feats, train_labels, test_feats, test_labels, args):
    """Train a simple linear classifier on frozen features."""
    num_classes = len(torch.unique(train_labels))
    in_dim = train_feats.shape[1]

    model = nn.Linear(in_dim, num_classes).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    for epoch in range(args.epochs):
        model.train()
        logits = model(train_feats.to(args.device))
        loss = criterion(logits, train_labels.to(args.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            preds = model(test_feats.to(args.device)).argmax(dim=1)
            acc = (preds.cpu() == test_labels).float().mean().item()
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | Loss: {loss.item():.4f} | Acc: {acc:.4f}"
        )

    return acc


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"⚙️ Using device: {device}")

    # === Load pretrained encoder ===
    encoder = MAEEncoder()
    ckpt = torch.load(args.encoder_ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    encoder_state = {
        k.replace("encoder.", ""): v for k, v in state_dict.items() if "encoder." in k
    }
    encoder.load_state_dict(encoder_state, strict=False)
    encoder = encoder.to(device)
    print("✅ Loaded encoder checkpoint.")

    # === Data ===
    train_loader, test_loader = get_dataloaders(args.batch_size)

    # === Extract features ===
    train_feats, train_labels = extract_features(encoder, train_loader, device)
    test_feats, test_labels = extract_features(encoder, test_loader, device)
    print(f"Features: {train_feats.shape}, Labels: {train_labels.shape}")

    train_feats = nn.functional.normalize(train_feats, dim=1)
    test_feats = nn.functional.normalize(test_feats, dim=1)

    # === Train linear probe ===
    acc = linear_probe(train_feats, train_labels, test_feats, test_labels, args)
    print(f"\n🏁 Final Linear Probe Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
