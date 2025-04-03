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


def test_sparse_moe_forward():
    input_dim = 8
    hidden_dim = 16
    num_experts = 8
    top_k = 2
    sparse_moe = SparseMoE(input_dim, hidden_dim, num_experts, top_k=top_k)
    x = torch.randn(32, input_dim)
    output = sparse_moe(x)
    assert output.shape == (
        32,
        input_dim,
    ), f"Expected output shape (32, {input_dim}), got {output.shape}"


def test_sparse_moe_shared_expert():
    input_dim = 8
    hidden_dim = 16
    num_experts = 8
    top_k = 2
    sparse_moe = SparseMoE(input_dim, hidden_dim, num_experts, top_k=top_k, shared=True)
    x = torch.randn(32, input_dim)
    output = sparse_moe(x)
    assert output.shape == (
        32,
        input_dim,
    ), f"Expected output shape (32, {input_dim}), got {output.shape}"


def test_sparse_moe_top_k_1():
    input_dim = 8
    hidden_dim = 16
    num_experts = 8
    top_k = 1
    sparse_moe = SparseMoE(input_dim, hidden_dim, num_experts, top_k=top_k)
    x = torch.randn(32, input_dim)
    output = sparse_moe(x)
    assert output.shape == (
        32,
        input_dim,
    ), f"Expected output shape (32, {input_dim}), got {output.shape}"



def test_sparse_moe_small_batch():
    input_dim = 8
    hidden_dim = 16
    num_experts = 8
    top_k = 2
    sparse_moe = SparseMoE(input_dim, hidden_dim, num_experts, top_k=top_k)
    x = torch.randn(1, input_dim)
    output = sparse_moe(x)
    assert output.shape == (
        1,
        input_dim,
    ), f"Expected output shape (1, {input_dim}), got {output.shape}"


if __name__ == "__main__":
    pytest.main()
