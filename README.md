
# Mixture of Experts (MoE) in PyTorch

This repository provides a clean and modular implementation of Mixture of Experts (MoE) layers in PyTorch, including both **DenseMoE** and **SparseMoE** variants.

## Features

- `DenseMoE`: Uses all experts with soft attention routing.
- `SparseMoE`: Activates only a few selected experts (Top-k) for efficiency.

## Installation

Simply copy `moe.py` into your project directory. Requires:

- Python 3.10+
- PyTorch 2.5.1+


## Usage

### 1. Import the module

```python
from moe import DenseMoE, SparseMoE
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
- When `aux_loss_flag=True`, `SparseMoE` returns an auxiliary load-balancing loss to encourage expert diversity.


## License

MIT License
