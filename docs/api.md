# API Reference

This document provides detailed API documentation for the muon_fsdp package.

## Table of Contents

- [MuonOptimizer](#muonoptimizer)
- [FSDPMuonOptimizer](#fsdpmuonoptimizer)
- [Utility Functions](#utility-functions)
- [Distributed Utilities](#distributed-utilities)

---

## MuonOptimizer

```python
from muon_fsdp import MuonOptimizer
```

Implements the Muon optimization algorithm for single-GPU training.

### Constructor

```python
MuonOptimizer(
    params: Iterable[Parameter],
    lr: float = 0.02,
    momentum: float = 0.95,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    ns_steps: int = 5,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | `Iterable[Parameter]` | - | Parameters to optimize |
| `lr` | `float` | 0.02 | Learning rate |
| `momentum` | `float` | 0.95 | Momentum coefficient (β) |
| `weight_decay` | `float` | 0.0 | Weight decay (L2 penalty) |
| `nesterov` | `bool` | False | Enable Nesterov momentum |
| `ns_steps` | `int` | 5 | Number of Newton-Schulz iterations |

#### Attributes

The optimizer maintains the following state:

```python
# For each parameter p:
state[p]["momentum_buffer"]  # Momentum buffer (same shape as p)
```

### Methods

#### step()

```python
@torch.no_grad()
def step(self, closure: Optional[Callable] = None) -> Optional[float]
```

Performs a single optimization step.

**Arguments:**
- `closure` (Optional[Callable]): A closure that reevaluates the model and returns the loss. If provided, the step is not differentiable.

**Returns:**
- `Optional[float]`: Loss value if `closure` is provided, `None` otherwise.

**Algorithm:**
```
1. Update momentum: buf = β * buf + (1 - β) * grad
2. For 2D matrices: orthogonalize using Newton-Schulz
3. Apply LR scaling: lr * max(1, m/n)^0.5
4. Apply weight decay: p *= (1 - lr * wd)
5. Update: p -= lr * update
```

#### state_dict()

```python
def state_dict(self) -> Dict[str, Any]
```

Returns the optimizer state as a dictionary.

**Returns:**
```python
{
    "state": {
        param_id: {"momentum_buffer": tensor, ...}
    },
    "param_groups": [...]
}
```

#### load_state_dict()

```python
def load_state_dict(self, state_dict: Dict[str, Any]) -> None
```

Loads optimizer state from a dictionary.

**Arguments:**
- `state_dict`: State dictionary from `state_dict()`.

### Usage Example

```python
import torch
import torch.nn as nn
from muon_fsdp import MuonOptimizer

model = nn.Linear(512, 2048)
optimizer = MuonOptimizer(
    model.parameters(),
    lr=0.02,
    momentum=0.95,
    weight_decay=0.01,
)

for input, target in dataloader:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

---

## FSDPMuonOptimizer

```python
from muon_fsdp import FSDPMuonOptimizer
```

Muon optimizer with FSDP2 integration for distributed training.

### Constructor

```python
FSDPMuonOptimizer(
    model: nn.Module,
    params: Optional[List[Parameter]] = None,
    lr: float = 0.02,
    weight_decay: float = 0.01,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    ns_stepsize: float = 1.0,
    beta2: float = 0.99,
    eps: float = 1e-8,
    gradient_accumulation_steps: int = 1,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | - | FSDP-wrapped model |
| `params` | `Optional[List[Parameter]]` | None | Parameters to optimize |
| `lr` | `float` | 0.02 | Learning rate |
| `weight_decay` | `float` | 0.01 | Weight decay coefficient |
| `momentum` | `float` | 0.95 | Momentum coefficient |
| `nesterov` | `bool` | True | Enable Nesterov momentum |
| `ns_steps` | `int` | 5 | Newton-Schulz iterations |
| `ns_stepsize` | `float` | 1.0 | Step size for NS updates |
| `beta2` | `float` | 0.99 | Second moment coefficient (Adam-style) |
| `eps` | `float` | 1e-8 | Numerical stability epsilon |
| `gradient_accumulation_steps` | `int` | 1 | Gradient accumulation steps |

#### State per Parameter

```python
state[p]["momentum_buffer"]    # First moment (same shape as p)
state[p]["second_moment"]      # Second moment (Adam-style)
state[p]["accum_count"]        # Gradient accumulation counter
```

### Methods

#### unshard_params()

```python
@contextmanager
def unshard_params(self) -> Generator[None, None, None]
```

Context manager to unshard FSDP parameters.

**Usage:**
```python
with optimizer.unshard_params():
    full_params = [p.full_tensor() for p in model.parameters()]
    # Parameters are unsharded here
# Parameters are automatically resharded on exit
```

#### step()

```python
def step(self, closure: Optional[Callable] = None) -> Optional[float]
```

Performs a single optimization step with FSDP2 handling.

**Algorithm:**
```
1. Apply weight decay and momentum
2. All-gather gradients from all processes
3. Compute Newton-Schulz orthogonalization on full gradients
4. Scatter updates back to sharded parameters
```

#### zero_grad()

```python
def zero_grad(self, set_to_none: bool = False) -> None
```

Zeros out gradients.

**Arguments:**
- `set_to_none` (bool): If True, set gradients to None instead of zeroing.

#### state_dict() / load_state_dict()

Same as MuonOptimizer, plus saves `_step_count`.

### Factory Function

```python
from muon_fsdp import create_fsdp_muon_optimizer

optimizer = create_fsdp_muon_optimizer(
    model=model,
    lr=0.02,
    weight_decay=0.01,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,
    ns_stepsize=1.0,
    beta2=0.99,
    eps=1e-8,
    gradient_accumulation_steps=1,
)
```

### FSDP2 Setup Example

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from muon_fsdp import FSDPMuonOptimizer

# Create model
model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8),
    num_layers=6,
)

# Wrap with FSDP2
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)

# Create optimizer
optimizer = FSDPMuonOptimizer(
    model=model,
    lr=0.02,
    weight_decay=0.01,
    momentum=0.95,
)

# Training loop
for input, target in dataloader:
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## Utility Functions

### zeropower_via_newtonschulz5()

```python
from muon_fsdp.utils import zeropower_via_newtonschulz5

orthogonalized = zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor
```

Computes the zeroth power of a matrix using Newton-Schulz iteration.

**Arguments:**
- `G` (Tensor): Input matrix (2D).
- `steps` (int): Number of iterations (default: 5).
- `dtype` (Optional[dtype]): Output dtype (default: G.dtype).

**Returns:**
- `torch.Tensor`: Orthogonalized matrix with same shape as input.

**Algorithm:**
Uses quintic polynomial iteration:
```
Y_0 = G
Z_0 = I
for i in range(steps):
    Y_{i+1} = Y_i * (a + b * Z_i * Y_i + c * (Z_i * Y_i)^2)
    Z_{i+1} = Z_i * (a + b * Y_i * Z_i + c * (Y_i * Z_i)^2)
```

**Note:** Coefficients (a, b, c) = (3.4445, -4.7750, 2.0315)

---

## Distributed Utilities

### is_available()

```python
from muon_fsdp.distributed import is_available

available: bool = is_available()
```

Checks if distributed training is available.

### get_rank()

```python
from muon_fsdp.distributed import get_rank

rank: int = get_rank()
```

Returns the current process rank.

### get_world_size()

```python
from muon_fsdp.distributed import get_world_size

world_size: int = get_world_size()
```

Returns the number of processes.

### all_gather_grads()

```python
from muon_fsdp.distributed import all_gather_grads

gathered = all_gather_grads(
    tensors: List[torch.Tensor],
) -> List[torch.Tensor]
```

All-gathers gradients from all processes.

**Arguments:**
- `tensors` (List[Tensor]): List of local gradient tensors.

**Returns:**
- `List[Tensor]`: List of gathered gradient tensors.

**Note:** Each tensor has shape `(world_size * dim0, dim1, ...)` where dim0 is the local dimension.

### get_dtensor_local_tensor()

```python
from muon_fsdp.fsdp import get_dtensor_local_tensor

local = get_dtensor_local_tensor(tensor: torch.Tensor) -> torch.Tensor
```

Extracts local tensor from DTensor or returns original tensor.

### get_dtensor_full_tensor()

```python
from muon_fsdp.fsdp import get_dtensor_full_tensor

full = get_dtensor_full_tensor(tensor: torch.Tensor) -> torch.Tensor
```

Gathers full tensor from DTensor by all-gathering shards.

---

## Exceptions

### OptimizerError

```python
from muon_fsdp.exceptions import OptimizerError
```

Base exception for optimizer errors.

**Subclasses:**
- `SparseGradientError`: Raised when sparse gradients are used (not supported).
- `InvalidParameterError`: Raised for invalid hyperparameters.

---

## Type Hints

```python
from muon_fsdp import MuonOptimizer, FSDPMuonOptimizer
from torch import Tensor, nn

# Parameter groups
ParamGroup = List[Dict[str, Any]]
ParamGroups = List[Dict[str, Any]]

# Optimizer state
OptimizerState = Dict[str, Any]
OptimizerStateDict = Dict[str, Any]

# FSDP types
DTensor = torch.distributed.tensor.DTensor
FSDPModule = torch.distributed.fsdp.FSDPModule
```
