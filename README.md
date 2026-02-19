# Mixture of Experts (MoE) in PyTorch

This repository provides a clean and modular implementation of Mixture of Experts (MoE) layers in PyTorch, including **DenseMoE**, **SparseMoE**, and **SwitchMoE** variants.

## Features

- `DenseMoE`: Uses all experts with soft attention routing.
- `SparseMoE`: Activates only a few selected experts (Top-k) for efficiency.
- `SwitchMoE`: Uses Top-1 routing with optional per-expert capacity control.

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

## CI

GitHub Actions runs on:

- every branch push
- merged pull requests

CI checks:

- `uv run ruff check .`
- `uv run mypy torchmoe`
- `uv run pytest -q`

## Usage

### 1. Import the module

```python
from torchmoe.moe import DenseMoE, SparseMoE, SwitchMoE
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

### 5. SwitchMoE example

```python
moe_layer = SwitchMoE(
    input_dim=128,
    hidden_dim=256,
    num_expert=8,
    capacity_factor=1.25,
    drop_tokens=True,
    aux_loss_flag=True
)
output, aux_loss = moe_layer(x)  # output shape: [32, 128], aux_loss is scalar or None
```

## Notes

- `DenseMoE` uses all experts, so it's compute-intensive.
- `SparseMoE` uses Top-k routing for better efficiency.
- `SwitchMoE` uses Top-1 routing and can enforce expert capacity with token drop/passthrough behavior.
- `SparseMoE` requires `1 <= top_k <= num_expert`.
- When `aux_loss_flag=True`, `SparseMoE` returns an auxiliary load-balancing loss to encourage expert diversity.
- When `aux_loss_flag=True`, `SwitchMoE` returns an auxiliary load-balancing loss.

## License

MIT License
