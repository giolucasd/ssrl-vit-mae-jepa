#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.manifold import TSNE
from timm.models.vision_transformer import VisionTransformer

from src.data import get_test_dataloader

from ..utils import setup_reproducibility, shut_down_warnings

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️ UMAP not installed. Install with: pip install umap-learn")


shut_down_warnings()
setup_reproducibility(seed=73)


# --------------------------------------------------
# Load checkpoint and extract encoder weights
# --------------------------------------------------
def load_encoder_from_ckpt(ckpt_path, model_cfg):
    print(f"🔍 Loading encoder from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

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
# Feature extraction with cls/mean pooling
# --------------------------------------------------
def pool_features(out, method):
    """
    out shape: (B, N, D)
    """
    if method == "cls":
        return out[:, 0, :]  # CLS token
    elif method == "mean":
        return out[:, 1:, :].mean(dim=1)  # average patch tokens
    else:
        raise ValueError(f"Unknown pool method: {method}")


def apply_normalization(features, mode):
    """
    features: numpy array (N, D)
    """
    if mode == "none":
        return features

    if mode == "l2":
        norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        return features / norm

    if mode == "channel":
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        return (features - mean) / std

    raise ValueError(f"Unknown normalization mode: {mode}")


@torch.no_grad()
def extract_features(
    encoder,
    dataloader,
    device,
    max_samples,
    pool_method,
    norm_method,
):
    encoder.eval()
    feats = []
    labels = []

    for imgs, lbls in dataloader:
        imgs = imgs.to(device)

        out = encoder.forward_features(imgs)
        pooled = pool_features(out, pool_method)

        feats.append(pooled.cpu())
        labels.append(lbls)

        if sum(len(x) for x in labels) >= max_samples:
            break

    feats = torch.cat(feats, dim=0)[:max_samples]
    labels = torch.cat(labels, dim=0)[:max_samples]

    # normalization step
    feats = feats.numpy()
    feats = apply_normalization(feats, norm_method)

    return feats, labels.numpy()


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

    raise ValueError(f"Unknown method: {method}")


# --------------------------------------------------
# Plot
# --------------------------------------------------
def plot_embedding(Z, y, out_path, title_extra, class_names):
    plt.figure(figsize=(8, 8))

    # unique sorted classes
    classes = np.unique(y)

    for cls in classes:
        mask = y == cls
        plt.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=12,
            label=class_names[cls],
        )

    plt.title(f"2D feature projection ({title_extra})")
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📁 Saved figure to {out_path}")


def plot_class_vs_all(Z, y, class_id, class_name, out_path):
    plt.figure(figsize=(6, 6))

    # Background classes (gray)
    mask_bg = y != class_id
    plt.scatter(Z[mask_bg, 0], Z[mask_bg, 1], c="lightgray", s=10, label="other")

    # Highlight class_id
    mask_cls = y == class_id
    plt.scatter(
        Z[mask_cls, 0],
        Z[mask_cls, 1],
        c="tab:red",
        s=12,
        label=class_name,
    )

    plt.title(f"Class {class_name} vs all")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mae.yaml")
    parser.add_argument("--encoder_ckpt", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["umap", "tsne"], default="umap")
    parser.add_argument("--pool", type=str, choices=["cls", "mean"], default="cls")
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["none", "l2", "channel"],
        default="none",
    )
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = load_encoder_from_ckpt(args.encoder_ckpt, model_cfg).to(device)

    dataloader = get_test_dataloader(cfg)
    class_names = dataloader.dataset.classes

    print("📥 Extracting features…")
    feats, labels = extract_features(
        encoder,
        dataloader,
        device,
        args.max_samples,
        args.pool,
        args.normalize,
    )

    print("📉 Projecting to 2D…")
    Z = project(feats, method=args.method)

    save_dir = Path("assets") / "visualizations"
    save_dir.mkdir(exist_ok=True)
    ckpt_name = Path(args.encoder_ckpt).parent.parent.stem

    out_path = (
        save_dir
        / f"representation_{ckpt_name}_{args.method}_{args.pool}_{args.normalize}.png"
    )
    title_info = f"{args.method}, pool={args.pool}, norm={args.normalize}"
    plot_embedding(Z, labels, str(out_path), title_info, class_names)
    # Generate 10 class-vs-all plots
    for cls_id in np.unique(labels):
        class_name = class_names[cls_id]
        out_path = (
            save_dir
            / f"representation_{ckpt_name}_{args.method}_{args.pool}_{args.normalize}.png"
        )
        out_cls = (
            save_dir
            / f"representation_{ckpt_name}_{args.method}_{args.pool}_{args.normalize}_class{cls_id}.png"
        )
        plot_class_vs_all(Z, labels, cls_id, class_name, str(out_cls))


if __name__ == "__main__":
    main()
