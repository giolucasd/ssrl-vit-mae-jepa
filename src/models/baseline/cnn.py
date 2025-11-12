import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.baseline.base_model import BaseModel

def plotTrace(train_losses,val_losses,train_acc,val_acc,lr,weight_decay,model_name,epochs=100):
  steps = np.arange(epochs)

  fig, ax1 = plt.subplots()

  title_text = f"{model_name} - LR: {lr}, Weight Decay: {weight_decay}"
  fig.suptitle(title_text)

  ax1.set_xlabel('epochs')
  ax1.set_ylabel('loss')
  #ax1.set_title('test loss: %.3f, test accuracy: %.3f' % (test_loss, test_acc))
  ax1.plot(steps, train_losses, label="train loss", color='red')
  ax1.plot(steps, val_losses, label="val loss", color='green')

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
  ax2.set_ylabel('acccuracy')  # we already handled the x-label with ax1
  ax2.plot(steps, val_acc, label="val acc", color='blue')
  ax2.plot(steps, train_acc, label="train acc", color='yellow')

  fig.legend()
  fig.tight_layout()
  plt.show()

class CNN(BaseModel):
    """
    A simple Convolutional Neural Network model with added regularization to combat overfitting.
    """
    def __init__(self, input_dim=96, num_classes=10, optimizer_type='Adam', weight_decay=1e-4, lr=0.01, device='cpu', dropout=0.5):
        """
        Initializes the CNN model.

        Args:
            input_dim (int): The height and width of the input images.
            num_classes (int): The number of output classes.
            optimizer_type (str): The type of optimizer to use (e.g., 'Adam', 'SGD').
            weight_decay (float): Weight decay for the optimizer.
            lr (float): Learning rate for the optimizer.
            device (str): The device to run the model on ('cpu' or 'cuda').
        """
        super().__init__(lr=lr, weight_decay=weight_decay, optimizer_type=optimizer_type, device=device)

        # Convolutional layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            # Block 2
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            # Block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            
        )

        # Determine the size of the flattened features after the conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_dim, input_dim)
            dummy_output = self.features(dummy_input)
            self._flattened_size = dummy_output.view(1, -1).shape[1]

        # Fully connected layers (classifier)
        self.classifier = nn.Sequential(
            nn.Linear(self._flattened_size, 64),
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