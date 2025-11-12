from __future__ import print_function

import argparse
import yaml
from pathlib import Path
import numpy as np
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from src.models.baseline.cnn import CNN, plotTrace
from src.data import get_train_dataloaders
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a simple CNN with PyTorch Lightning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()

def load_config(path: str | Path) -> dict:
    """Loads a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

# def get_dataloaders(data_dir: Path, output_dir: Path, batch_size: int):

#     def compute_mean_std(loader):
#         mean = 0.0
#         var = 0.0
#         for images, _ in loader:
#             batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
#             images = images.view(batch_samples, images.size(1), -1)
#             mean += images.mean(2).sum(0)
#             var += images.var(2).sum(0)

#         mean /= len(loader.dataset)
#         var /= len(loader.dataset)
#         std = torch.sqrt(var)

#         return mean, std
#     # Load the dataset without any normalization transforms
#     train_dataset_example = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transforms.ToTensor())
#     loader = torch.utils.data.DataLoader(train_dataset_example, batch_size=128, num_workers=1, shuffle=True)

#     # Now compute the mean and standard deviation
#     mean, std = compute_mean_std(loader)
#     train_transform=transforms.Compose([
#         transforms.RandomCrop((64, 64)),
#         transforms.TrivialAugmentWide(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean,std)
#         ])
#     test_transform=transforms.Compose([
#             transforms.CenterCrop(size=(64,64)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean,std)
#             ])

#     train_dataset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=train_transform)
#     val_dataset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=test_transform)
        
#     total_train_samples = len(train_dataset)
#     val_size = 0.2
#     num_val_samples = int(np.floor(val_size * total_train_samples))
#     num_train_samples = total_train_samples - num_val_samples

#     # Generate shuffled indices for train/validation split
#     indices = np.arange(total_train_samples)
#     np.random.shuffle(indices)
#     train_indices, val_indices = indices[num_val_samples:], indices[:num_val_samples]


#     train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
#     val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

#     # Create DataLoader instances for each dataset
#     batch_size = 64
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=1)
#     return train_loader, val_loader

def main():
    """Main training and evaluation script."""
    args = parse_args()
    config = load_config(args.config)
    

    output_dir = Path(config["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data")
    # --- Data Loading ---
    train_loader, val_loader = get_train_dataloaders(
        config
    )
    if train_loader is None or val_loader is None:
        print("Error: Data loaders could not be created.")
        return

    epochs = 200
    cuda = True
    seed = 42

    use_cuda = cuda and torch.cuda.is_available()

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Handel GPU stochasticity
    torch.backends.cudnn.enabled = use_cuda
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if use_cuda else "cpu")
    # --- Model Initialization ---
    model=CNN(lr=0.0005,optimizer_type='Adam',weight_decay=0.0001,device=device)
    model.to(device)
    train_losses,val_losses,train_acc,val_acc=model.fit(train_loader=train_loader,nr_epochs=epochs,val_loader=val_loader,verbose=True,print_interval=10)
    plotTrace(train_losses,val_losses,train_acc,val_acc,0.0005,0.0001,'CNN',200)
    print(f'Validation Accuracy: {val_acc[-1]}')
    
    print("Training finished.")



if __name__ == '__main__':
    main()