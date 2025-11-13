import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.baseline.base_model import BaseModel


class CNN(BaseModel):
    """
    A Convolutional Neural Network model with enhanced regularization to combat overfitting.
    This version includes more convolutional layers, increased dropout, and padding.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        optimizer_type,
        weight_decay,
        lr,
        device,
        momentum,
    ):
        """
        Initializes the enhanced CNN model.

        Args:
            input_dim (int): The height and width of the input images.
            num_classes (int): The number of output classes.
            optimizer_type (str): The type of optimizer to use (e.g., 'Adam', 'SGD').
            weight_decay (float): L2 regularization strength for the optimizer.
            lr (float): Learning rate for the optimizer.
            device (str): The device to run the model on ('cpu' or 'cuda').
            momentum (float): Momentum for SGD optimizer.
        """
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            device=device,
            momentum=momentum,
        )

        # Convolutional layers with increased depth and regularization
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            # Block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.4),
            # Block 4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.4),
        )

        # Determine the size of the flattened features after the conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_dim, input_dim)
            dummy_output = self.features(dummy_input)
            self._flattened_size = dummy_output.view(1, -1).shape[1]

        # Fully connected layers (classifier) with stronger dropout
        self.classifier = nn.Sequential(
            nn.Linear(self._flattened_size, 128),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes),
        )

        self.set_optimizer()

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.classifier(x)
        return x

    def plotTrace(
        self,
        train_losses,
        val_losses,
        train_acc,
        val_acc,
        model_name,
        epochs,
    ):
        """
        Plots the training and validation loss and accuracy over epochs.
        """
        plt.style.use("seaborn-v0_8-whitegrid")
        steps = np.arange(epochs)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))

        # Define a color palette
        colors = {
            "train_loss": "tab:blue",
            "val_loss": "tab:orange",
            "train_acc": "tab:green",
            "val_acc": "tab:red",
        }

        # Main title
        title_text = f"{model_name} - LR: {self.lr}, Weight Decay: {self.weight_decay}, Optimizer: {self.optimizer_type}"
        fig.suptitle(title_text, fontsize=16, fontweight="bold")

        # Subplot for losses
        ax1.set_title("Loss Over Epochs", fontsize=14)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.plot(
            steps,
            train_losses,
            label="Train Loss",
            color=colors["train_loss"],
            linestyle="-",
            marker="o",
            markersize=5,
        )
        ax1.plot(
            steps,
            val_losses,
            label="Validation Loss",
            color=colors["val_loss"],
            linestyle="--",
            marker="x",
            markersize=5,
        )
        ax1.legend(loc="best", fontsize=10)
        ax1.set_ylim(bottom=0)  # Loss should not be negative

        # Subplot for accuracies
        ax2.set_title("Accuracy Over Epochs", fontsize=14)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_xlabel("Epochs", fontsize=12)
        ax2.plot(
            steps,
            train_acc,
            label="Train Accuracy",
            color=colors["train_acc"],
            linestyle="-",
            marker="o",
            markersize=5,
        )
        ax2.plot(
            steps,
            val_acc,
            label="Validation Accuracy",
            color=colors["val_acc"],
            linestyle="--",
            marker="x",
            markersize=5,
        )
        ax2.legend(loc="best", fontsize=10)
        ax2.set_ylim([0, 1])  # Accuracy is between 0 and 1

        # Improve layout
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Save the figure
        fig.savefig(f"{model_name}_training_trace.png", dpi=300, bbox_inches="tight")

        plt.show()
