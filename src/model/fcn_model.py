"""Fully connected neural network for salary regression."""

from typing import List

import torch
import torch.nn as nn


class FCNModel(nn.Module):
    """Fully connected feed-forward network for tabular regression.

    Each hidden layer is followed by BatchNorm, ReLU, and Dropout.
    The output layer produces a single scalar (log-salary).

    Attributes:
        input_dim: Number of input features.
        hidden_dims: Sizes of the hidden layers in order.
        dropout_rate: Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float,
    ) -> None:
        """Build the network from layer dimension specs.

        Args:
            input_dim: Number of input features.
            hidden_dims: List of hidden layer widths, e.g. [256, 128, 64, 32].
            dropout_rate: Probability of zeroing a neuron during training.
        """
        super().__init__()
        self._net = self._build_network(input_dim, hidden_dims, dropout_rate)

    def _build_network(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float,
    ) -> nn.Sequential:
        """Assemble the sequential layer stack.

        Args:
            input_dim: Width of the first layer input.
            hidden_dims: Hidden layer widths.
            dropout_rate: Dropout probability.

        Returns:
            Assembled Sequential module.
        """
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size,).
        """
        return self._net(x).squeeze(1)
