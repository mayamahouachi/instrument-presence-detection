import torch
from torch import nn


class CNNFeatureNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        cnn1_filter_nb: int = 16,
        cnn2_filter_nb: int = 32,
        cnn3_filter_nb: int = 64,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param cnn1_filter_nb: The number of output features of the first convolutional layer.
        :param cnn1_filter_nb: The number of output features of the second convolutional layer.
        :param cnn1_filter_nb: The number of output features of the third convolutional layer.
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, cnn1_filter_nb, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(cnn1_filter_nb, cnn2_filter_nb, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(cnn2_filter_nb, cnn3_filter_nb, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor. Shape (B, n_mels, frame)
        :return: A tensor of features. Shape (B, cnn3_filter_nb, ?)
        """
        return self.model(x)


if __name__ == "__main__":
    _ = CNNFeatureNet()
