import pytest
import torch

from torchmoe.moe import DenseMoE, Expert, Gate, SparseMoE


def test_expert_forward():
    input_dim = 8
    hidden_dim = 16
    expert = Expert(input_dim, hidden_dim)

    x = torch.randn(32, input_dim)
    output = expert(x)
    assert output.shape == (
        32,
        input_dim,
    ), f"Expected output shape (32, {input_dim}), got {output.shape}"


def test_gate_forward():
    input_dim = 8
    num_experts = 4
    gate = Gate(input_dim, num_experts)
    x = torch.randn(32, input_dim)
    output = gate(x)
    assert output.shape == (
        32,
        num_experts,
    ), f"Expected output shape (32, {num_experts}), got {output.shape}"


def test_dense_moe_forward():
    input_dim = 8
    hidden_dim = 16
    num_experts = 4
    moe = DenseMoE(input_dim, hidden_dim, num_experts)
    x = torch.randn(32, input_dim)
    output = moe(x)
    assert output.shape == (
        32,
        input_dim,
    ), f"Expected output shape (32, {input_dim}), got {output.shape}"

@pytest.mark.parametrize("aux_loss_flag, use_shared", [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
])
def test_sparse_moe_forward(aux_loss_flag, use_shared):
    torch.manual_seed(42)
    batch_size = 16
    input_dim = 32
    hidden_dim = 64
    num_expert = 4
    top_k = 2

    model = SparseMoE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_expert=num_expert,
        top_k=top_k,
        shared=use_shared,
        aux_loss_flag=aux_loss_flag
    )

    x = torch.randn(batch_size, input_dim)
    output, aux = model(x)

    # Check output shape
    assert output.shape == (batch_size, input_dim), "Output shape mismatch"

    # Check aux loss
    if aux_loss_flag:
        assert isinstance(aux, torch.Tensor), "Aux loss should be a tensor when enabled"
        assert aux.dim() == 0, "Aux loss should be a scalar"
        assert aux >= 0, "Aux loss should be non-negative"
    else:
        assert aux is None, "Aux loss should be None when not enabled"

def test_sparse_moe_deterministic_expert_selection():
    torch.manual_seed(123)
    model = SparseMoE(16, 32, num_expert=4, top_k=1, shared=False, aux_loss_flag=False)
    x = torch.randn(8, 16)

    with torch.no_grad():
        gate_logits = model.gate(x)
        _, selected = torch.topk(gate_logits, model.top_k, dim=-1)

    # Forward call
    output, _ = model(x)

    # Check expert selection count
    selected_flat = selected.view(-1).tolist()
    unique_experts = set(selected_flat)
    assert len(unique_experts) <= model.num_expert, "Selected experts exceed number of experts"

def test_sparse_moe_backward():
    model = SparseMoE(input_dim=16, hidden_dim=32, num_expert=4, top_k=2, shared=True, aux_loss_flag=True)
    x = torch.randn(10, 16, requires_grad=True)
    y, aux = model(x)
    loss = y.mean()
    if aux is not None:
        loss += aux
    loss.backward()
    assert x.grad is not None, "No gradients computed"
    assert x.grad.shape == x.shape, "Gradient shape mismatch"

if __name__ == "__main__":
    pytest.main()
