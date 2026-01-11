# src/model.py
import torch
import torch.nn as nn


class InstrumentCNN(nn.Module):
    """Detects in each window the presence of each audio source class.

    Input shape: (B, n_mels, n_frame)
    Output shape: (B, n_classes)
    """

    def __init__(self, n_classes: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward method."""
        x = self.features(x)
        x = self.classifier(x)
        return x
