import pytorch_lightning as pl

from src.models.mae.classifier import ViTClassifier
from src.models.mae.encoder import MAEEncoder


class MAEClassifierModule(pl.LightningModule):
    """LightningModule for evaluation only (loads full classifier checkpoint)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        encoder = MAEEncoder()
        self.model = ViTClassifier(encoder=encoder, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
