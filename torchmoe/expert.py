import torch
from torch import nn


class Expert(nn.Module):
    """
    A single MoE (Mixture of Experts) expert.

    Args:
        input_dim (int): The input dimension of the expert.
        hidden_dim (int): The hidden dimension of the expert.
        activation (type[nn.Module]): The activation function to use.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, activation: type[nn.Module] = nn.GELU
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Linear layer to transform input to hidden dimension
        self.l1 = nn.Linear(self.input_dim, self.hidden_dim)
        # Linear layer to transform hidden dimension back to input dimension
        self.l2 = nn.Linear(self.hidden_dim, self.input_dim)
        # Activation function applied after the first linear layer
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the expert
        return self.l2(self.activation(self.l1(x)))  # Shape: [batch_size, input_dim]
