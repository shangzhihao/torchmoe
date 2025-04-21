import torch
from torch import nn
from torch.nn import functional as F


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
        self.input_dim = input_dim
        self.num_experts = num_experts
        # List of experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, expert_act) for _ in range(self.num_experts)])
        # Gate to assign weights to experts
        self.gate = Gate(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get gate weights and apply softmax
        gate_weights = self.gate(x).softmax(dim=1, dtype=x.dtype)  # Shape: [batch_size, num_experts]

        # Apply each expert with the corresponding weight
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # Shape: [batch_size, num_experts, input_dim]
        # Weighted sum of expert outputs
        output = torch.sum(expert_outputs * gate_weights.unsqueeze(2), dim=1)  # Shape: [batch_size, input_dim]

        return output  # Shape: [batch_size, input_dim]


class SparseMoE(nn.Module):
    """
    Sparse MoE (Mixture of Experts) layer.
    
    Args:
        input_dim (int): The input dimension.
        hidden_dim (int): The hidden dimension in experts.
        num_expert (int): The number of experts.
        expert_act (nn.Module): The activation function in experts.
        top_k (int): The number of experts to select.
        shared (bool): Whether to share the weights of experts.
        aux_loss (bool): Whether to use auxiliary loss.
    """
    def __init__(self,
                 input_dim: int, 
                 hidden_dim: int, 
                 num_expert: int = 8, 
                 expert_act: type[nn.Module] = nn.GELU,
                 top_k: int = 2,
                 shared: bool = True,
                 aux_loss_flag: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_expert = num_expert
        self.expert_act = expert_act
        self.top_k = top_k
        self.shared = shared
        self.aux_loss_flag = aux_loss_flag

        # List of experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, expert_act) for _ in range(num_expert)])
        # Gate to assign weights to experts
        self.gate = Gate(input_dim, num_expert)
        # Shared expert if sharing is enabled
        if self.shared:
            self.shared_expert = Expert(input_dim, hidden_dim, expert_act)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor|None]:
        # Get gate logits and apply softmax
        gate_logits = self.gate(x)  # Shape: [batch_size, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)  # Shape: [batch_size, num_experts]

        # Select top k experts and their corresponding weights
        weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=-1)  # weights: [batch_size, top_k], selected_experts: [batch_size, top_k]
        weights = F.softmax(weights, dim=-1).to(x.dtype)  # Shape: [batch_size, top_k]

        # Initialize results tensor
        results = torch.zeros_like(x, dtype=x.dtype)  # Shape: [batch_size, input_dim]

        # Apply selected experts to the input
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)  # batch_idx: [num_selected], nth_expert: [num_selected]
            if len(batch_idx) > 0:
                results[batch_idx] += weights[batch_idx, nth_expert][:, None] * expert(x[batch_idx])  # Shape: [num_selected, input_dim]

        # Add shared expert output if sharing is enabled
        if self.shared:
            results += self.shared_expert(x)  # Shape: [batch_size, input_dim]

        # Calculate auxiliary loss if required
        if self.aux_loss_flag:
            # Importance: sum over gate probabilities
            importance = gate_probs.sum(dim=0)  # Shape: [num_experts]

            # Load: count how many times each expert was selected
            load = torch.zeros(self.num_expert, device=x.device)  # Shape: [num_experts]
            for i in range(self.num_expert):
                load[i] = (selected_experts == i).sum()  # Shape: []

            # Normalize importance and load
            importance = importance / (importance.sum() + 1e-8)  # Shape: [num_experts]
            load = load / (load.sum() + 1e-8)  # Shape: [num_experts]

            # Auxiliary loss: squared L2 distance between load and importance
            aux_loss = torch.sum((importance - load) ** 2)  # Shape: []
            return results, aux_loss  # Shape: ([batch_size, input_dim], [])

        return results, None  # Shape: ([batch_size, input_dim], None)