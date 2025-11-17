#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.manifold import TSNE
from timm.models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️ UMAP not installed. Install with: pip install umap-learn")


# --------------------------------------------------
# Load checkpoint and extract encoder weights
# --------------------------------------------------
def load_encoder_from_ckpt(ckpt_path, model_cfg):
    print(f"🔍 Loading encoder from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # instantiate empty ViT encoder
    encoder_cfg = model_cfg["encoder"]
    general_cfg = model_cfg["general"]

    encoder = VisionTransformer(
        img_size=general_cfg["image_size"],
        patch_size=general_cfg["patch_size"],
        in_chans=general_cfg["in_chans"],
        embed_dim=encoder_cfg["embed_dim"],
        depth=encoder_cfg["depth"],
        num_heads=encoder_cfg["num_heads"],
        num_classes=0,
    )

    # Possible prefixes where encoder weights may live
    possible_prefixes = [
        "model.encoder.",  # LightningModule → self.model.encoder
        "encoder.",  # plain state_dict from MAE
        "module.encoder.",  # DDP/DataParallel
    ]

    # Detect which prefix exists in this checkpoint
    prefix = None
    for p in possible_prefixes:
        if any(k.startswith(p) for k in state_dict.keys()):
            prefix = p
            break

    if prefix is None:
        raise ValueError(
            "❌ Could not find encoder weights in checkpoint. "
            "Expected keys starting with one of: " + ", ".join(possible_prefixes)
        )

    print(f"🔎 Detected encoder prefix in checkpoint: '{prefix}'")

    # Extract only encoder weights and strip the prefix
    encoder_state = {
        k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
    }

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    print(f"   → Loaded {len(encoder_state)} params")
    print(f"   → Missing: {len(missing)} | Unexpected: {len(unexpected)}")

    return encoder


# --------------------------------------------------
# Extract features
# --------------------------------------------------
@torch.no_grad()
def extract_features(encoder, dataloader, device, max_samples=1000):
    encoder.eval()
    feats = []
    labels = []

    for imgs, lbls in dataloader:
        imgs = imgs.to(device)

        # (batch, seq_len, dim)
        out = encoder.forward_features(imgs)

        # CLS token
        cls_feats = out[:, 0, :]  # (b, dim)

        feats.append(cls_feats.cpu())
        labels.append(lbls)

        if sum(len(x) for x in labels) >= max_samples:
            break

    feats = torch.cat(feats, dim=0)[:max_samples]
    labels = torch.cat(labels, dim=0)[:max_samples]
    return feats.numpy(), labels.numpy()


# --------------------------------------------------
# Dimensionality reduction
# --------------------------------------------------
def project(features, method="umap"):
    if method == "tsne":
        print("🔎 Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto")
        return tsne.fit_transform(features)

    elif method == "umap":
        if not HAS_UMAP:
            raise RuntimeError("UMAP requested but not installed.")
        print("🔎 Running UMAP...")
        reducer = umap.UMAP(n_components=2)
        return reducer.fit_transform(features)

    else:
        raise ValueError(f"Unknown method: {method}")


# --------------------------------------------------
# Plot
# --------------------------------------------------
def plot_embedding(Z, y, out_path):
    plt.figure(figsize=(8, 8))
    num_classes = len(np.unique(y))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y, s=12, cmap="tab10")
    plt.title("2D feature projection")
    plt.colorbar(scatter, ticks=list(range(num_classes)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📁 Saved figure to {out_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mae.yaml")
    parser.add_argument("--encoder_ckpt", type=str, required=True)
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "tsne"])
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load encoder
    encoder = load_encoder_from_ckpt(args.encoder_ckpt, model_cfg)
    encoder.to(device)

    # Data
    transform = transforms.Compose(
        [
            transforms.Resize(96),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.STL10(
        root="data", split="train", download=True, transform=transform
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # Extract features
    print("📥 Extracting features…")
    feats, labels = extract_features(encoder, dataloader, device, args.max_samples)

    # Project
    Z = project(feats, method=args.method)

    # Save
    Path("assets").mkdir(exist_ok=True)
    ckpt_name = Path(args.encoder_ckpt).stem
    out_path = f"assets/representation_{args.method}_{ckpt_name}.png"
    plot_embedding(Z, labels, out_path)


if __name__ == "__main__":
    main()
