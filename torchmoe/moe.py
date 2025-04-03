
import torch
from torch import nn
from torch.nn import functional as F


class Expert(nn.Module):
    """
    A single MoE expert.
    Attributes:
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
        self.l1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, self.input_dim)
        self.activation = activation()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.l2(self.activation(self.l1(x)))


class Gate(nn.Module):
    """
    Assign weights to experts.
    Attributes:
        input_dim (int): The input dimension of the gate.
        out_dim (int): The output dimension of the gate.
    """
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = num_experts
        self.linear = nn.Linear(self.input_dim, self.out_dim)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.linear(x)

class DenseMoE(nn.Module):
    """
    MoE layer
    Attributes:
        input_dim (int): the input dimension.
        hidden_dim (int): the hidden dimension in experts.
        num_experts (int): the number of experts.
        expert_act (nn.Module): the activation function in experts.
        expert_norm (nn.Module): the normalization layer in experts.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        expert_act: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, expert_act) for _ in range(self.num_experts)])
        self.gate = Gate(input_dim, num_experts)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        gate_weights = self.gate(x).softmax(dim=1, dtype=x.dtype)

        # Apply each expert with the corresponding weight
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(expert_outputs * gate_weights.unsqueeze(2), dim=1)

        return output


class SparseMoE(nn.Module):
    def __init__(self,
                 input_dim:int, 
                 hidden_dim:int, 
                 num_expert:int = 8, 
                 expert_act: type[nn.Module] = nn.GELU,
                 top_k:int = 2,
                 shared:bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_expert = num_expert
        self.expert_act = expert_act
        self.top_k = top_k
        self.shared = shared

        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, expert_act) for _ in range(num_expert)])
        self.gate = Gate(input_dim, num_expert)
        if self.shared:
            self.shared_expert = Expert(input_dim, hidden_dim, expert_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)
        weights, selected_experts = torch.topk(gate_logits, self.top_k)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(x.dtype)
        results = torch.zeros_like(x, dtype=x.dtype)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(x[batch_idx])
        if self.shared:
            results = results + self.shared_expert(x)
        return results