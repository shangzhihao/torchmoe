import torch
from torch import nn

from .expert import Expert
from .gate import Gate


class DenseMoE(nn.Module):
    """
    MoE (Mixture of Experts) layer with dense routing.

    Args:
        input_dim (int): The input dimension.
        hidden_dim (int): The hidden dimension in experts.
        num_experts (int): The number of experts.
        expert_act (nn.Module): The activation function in experts.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        expert_act: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}")

        self.input_dim = input_dim
        self.num_experts = num_experts
        # List of experts
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, expert_act) for _ in range(self.num_experts)]
        )
        # Gate to assign weights to experts
        self.gate = Gate(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get gate weights and apply softmax
        gate_weights = self.gate(x).softmax(
            dim=1, dtype=x.dtype
        )  # Shape: [batch_size, num_experts]

        # Apply each expert with the corresponding weight
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # Shape: [batch_size, num_experts, input_dim]
        # Weighted sum of expert outputs
        output = torch.sum(
            expert_outputs * gate_weights.unsqueeze(2), dim=1
        )  # Shape: [batch_size, input_dim]

        return output  # Shape: [batch_size, input_dim]
