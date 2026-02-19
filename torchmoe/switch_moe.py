import math

import torch
from torch import nn
from torch.nn import functional as F

from .expert import Expert
from .gate import Gate


class SwitchMoE(nn.Module):
    """
    Switch-style MoE layer with top-1 routing.

    Args:
        input_dim (int): The input dimension.
        hidden_dim (int): The hidden dimension in experts.
        num_expert (int): The number of experts.
        expert_act (type[nn.Module]): The activation function in experts.
        capacity_factor (float): Per-expert capacity scaling factor.
        drop_tokens (bool): Whether to enforce capacity and drop overflow tokens.
        aux_loss_flag (bool): Whether to return a load-balancing auxiliary loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_expert: int = 8,
        expert_act: type[nn.Module] = nn.GELU,
        capacity_factor: float = 1.0,
        drop_tokens: bool = True,
        aux_loss_flag: bool = False,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if num_expert <= 0:
            raise ValueError(f"num_expert must be > 0, got {num_expert}")
        if capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be > 0, got {capacity_factor}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_expert = num_expert
        self.expert_act = expert_act
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.aux_loss_flag = aux_loss_flag

        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, expert_act) for _ in range(num_expert)]
        )
        self.gate = Gate(input_dim, num_expert)

    def _capacity(self, batch_size: int) -> int:
        base_capacity = self.capacity_factor * batch_size / self.num_expert
        return max(1, math.ceil(base_capacity))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        gate_logits = self.gate(x)  # Shape: [batch_size, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)  # Shape: [batch_size, num_experts]

        # Top-1 routing.
        route_probs, selected_experts = torch.max(
            gate_probs, dim=-1
        )  # route_probs: [batch_size], selected_experts: [batch_size]

        results = torch.zeros_like(x, dtype=x.dtype)  # Shape: [batch_size, input_dim]
        dispatched_mask = torch.zeros(
            x.size(0), device=x.device, dtype=torch.bool
        )  # Shape: [batch_size]
        capacity = self._capacity(x.size(0))

        for i, expert in enumerate(self.experts):
            token_idx = torch.where(selected_experts == i)[0]  # Shape: [num_selected]
            if token_idx.numel() == 0:
                continue
            if self.drop_tokens and token_idx.numel() > capacity:
                token_idx = token_idx[:capacity]

            expert_output = expert(
                x[token_idx]
            )  # Shape: [num_selected_kept, input_dim]
            token_weights = route_probs[token_idx].to(x.dtype).unsqueeze(1)
            results[token_idx] = expert_output * token_weights
            dispatched_mask[token_idx] = True

        if self.drop_tokens:
            # Preserve signal for dropped tokens by passing them through unchanged.
            dropped_mask = ~dispatched_mask
            if dropped_mask.any():
                results[dropped_mask] = x[dropped_mask]

        if self.aux_loss_flag:
            dispatched_experts = selected_experts[dispatched_mask]
            load = torch.bincount(dispatched_experts, minlength=self.num_expert).to(
                gate_probs.dtype
            )
            load = load / (dispatched_experts.numel() + 1e-8)
            importance = gate_probs.mean(dim=0)

            # Switch load-balancing objective (scaled by number of experts).
            aux_loss = self.num_expert * torch.sum(load * importance)
            return results, aux_loss

        return results, None
