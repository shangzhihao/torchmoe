import torch
from torch import nn


class Gate(nn.Module):
    """
    Assign weights to experts.

    Args:
        input_dim (int): The input dimension of the gate.
        num_experts (int): The number of experts (output dimension of the gate).
    """

    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = num_experts
        # Linear layer to transform input to expert weights
        self.linear = nn.Linear(self.input_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the gate to get expert weights
        return self.linear(x)  # Shape: [batch_size, num_experts]
