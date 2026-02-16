# Mixture of Experts (MoE) in PyTorch

This repository provides a clean and modular implementation of Mixture of Experts (MoE) layers in PyTorch, including both **DenseMoE** and **SparseMoE** variants.

## Features

- `DenseMoE`: Uses all experts with soft attention routing.
- `SparseMoE`: Activates only a few selected experts (Top-k) for efficiency.

## Installation

Clone the repository and install dependencies with `uv`:

```bash
uv sync --group dev
```

Requirements:

- Python 3.10+
- PyTorch 2.5.1+

## Development Checks

Install git hooks:

```bash
uv run pre-commit install
```

Run all hooks manually:

```bash
uv run pre-commit run --all-files
```

Configured hooks:

- `ruff format --check` for formatting consistency
- `ruff` lint check
- `isort --check-only` for import ordering
- `mypy` type check on `torchmoe`
- File hygiene checks (`trailing-whitespace`, `end-of-file-fixer`, `check-merge-conflict`)
- Config validation (`check-yaml`, `check-toml`)
- Security/safety checks (`detect-private-key`, `check-added-large-files`)

## Usage

### 1. Import the module

```python
from torchmoe.moe import DenseMoE, SparseMoE
import torch
```

### 2. Create dummy input

```python
x = torch.randn(32, 128)  # (batch_size, input_dim)
```

### 3. DenseMoE example

```python
moe_layer = DenseMoE(input_dim=128, hidden_dim=256, num_experts=4)
output = moe_layer(x)  # output shape: [32, 128]
```

### 4. SparseMoE example

```python
moe_layer = SparseMoE(
    input_dim=128,
    hidden_dim=256,
    num_expert=8,
    top_k=2,
    shared=True,
    aux_loss_flag=True
)
output, aux_loss = moe_layer(x)  # output shape: [32, 128], aux_loss is scalar or None
```

## Notes

- `DenseMoE` uses all experts, so it's compute-intensive.
- `SparseMoE` uses Top-k routing for better efficiency.
- `SparseMoE` requires `1 <= top_k <= num_expert`.
- When `aux_loss_flag=True`, `SparseMoE` returns an auxiliary load-balancing loss to encourage expert diversity.

## Roadmap

- [ ] `SwitchMoE` (Top-1 routing) for lower routing overhead.
- [ ] `Top2MoE` variant with per-token dispatch to two experts.
- [ ] `ExpertChoiceMoE` to improve expert load balancing.
- [ ] `HashMoE` for deterministic/static token-to-expert routing.
- [ ] `HierarchicalMoE` for multi-level expert selection.
- [ ] `SharedRoutedMoE` with always-on shared experts plus routed experts.
- [ ] `SoftMoE` with differentiable routing.
- [ ] Capacity-aware (dropless) routing to reduce token drops.

## License

MIT License
