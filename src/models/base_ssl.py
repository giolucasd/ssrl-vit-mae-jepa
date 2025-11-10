from abc import ABC, abstractmethod

import torch.nn as nn


class BaseSSLModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        """Return model outputs."""
        pass

    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        """Compute SSL-specific loss (e.g. pixel MSE, embedding similarity)."""
        pass
