import torch
from torch import nn


class SimpleDenseClassifier(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        hidden_size: int = 128,
        dropout: float = 0.3,
        output_size: int = 3,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param hidden_size: The number of output features of the first linear layer.
        :param dropout: The dropout rate.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        return self.model(x)


if __name__ == "__main__":
    _ = SimpleDenseClassifier()
