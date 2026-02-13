# PROJECT KNOWLEDGE BASE

**Generated:** 2026-02-13
**Branch:** main

## OVERVIEW

Muon optimizer implementation with FSDP2 (Fully Sharded Data Parallel) support for PyTorch. Maintains orthogonal weight matrices via Newton-Schulz iteration for stable LLM training.

## STRUCTURE

```
./                    # Root
├── muon_fsdp/       # Core package (5 modules)
├── tests/           # Test suite (5 files + mocks)
├── examples/        # Usage examples (2 scripts)
├── docs/            # Documentation (2 guides)
├── setup.py         # Package configuration
└── .github/workflows/
    └── test-cpu.yml # CI/CD
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Core optimizer | `muon_fsdp/optimizer.py` | MuonOptimizer class |
| FSDP2 integration | `muon_fsdp/fsdp.py` | FSDPMuonOptimizer (590 lines) |
| Newton-Schulz | `muon_fsdp/utils.py` | `zeropower_via_newtonschulz5()` |
| Distributed utils | `muon_fsdp/distributed.py` | All-gather, scatter |
| Tests | `tests/*.py` | 64 tests passing |
| FSDP mocks | `tests/mocks/fsdp_mock.py` | MockDTensor, MockFSDPModule |
| Examples | `examples/*.py` | GPT training, hybrid optimizer |

## CODE MAP

| Symbol | Type | Location | Lines | Role |
|--------|------|----------|-------|------|
| MuonOptimizer | class | optimizer.py | 1-197 | Single-GPU optimizer |
| FSDPMuonOptimizer | class | fsdp.py | 115-545 | Distributed optimizer |
| zeropower_via_newtonschulz5 | func | utils.py | 26-82 | Orthogonalization |
| all_gather_grads | func | distributed.py | 58-95 | Gradient comm |
| MockDTensor | class | mocks/fsdp_mock.py | 36-88 | FSDP2 mock |
| MockFSDPModule | class | mocks/fsdp_mock.py | 91-168 | FSDP module mock |

## CONVENTIONS

**Import Order** (from actual code):
```python
# Standard library
from __future__ import annotations
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import logging
from contextlib import contextmanager

# Third party
import torch
import torch.nn as nn
from torch.optim import Optimizer

# Local
from .utils import zeropower_via_newtonschulz5
from .distributed import all_gather_grads
```

**Naming**:
- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

**Type Hints**: Required for public APIs (`from __future__ import annotations`)

**Docstrings**: Google-style for classes/functions

**Testing**:
- Class-based: `class Test<Component><Feature>:`
- Fixtures in `conftest.py`: `random_matrix_256()`, `random_matrix_512()`, etc.
- Custom markers: `@pytest.mark.slow`, `@pytest.mark.distributed`, `@pytest.mark.cuda`

## ANTI-PATTERNS (THIS PROJECT)

1. **Missing cli.py**: `setup.py` declares `muon-fsdp=muon_fsdp.cli:main` but `cli.py` does NOT exist. CLI will fail at runtime. Fix: create `muon_fsdp/cli.py` or remove entry point.

2. **No lint config files**: No `.ruff.toml`, `mypy.ini`, or `pyproject.toml`. Relies on tool defaults.

3. **No coverage config**: pytest-cov in dev deps but no `.coveragerc` or pyproject.toml coverage settings.

4. **Legacy setup.py**: Modern projects use `pyproject.toml`. Migration recommended.

5. **Single CI workflow**: Only `.github/workflows/test-cpu.yml` - no lint/check workflow.

## UNIQUE STYLES

- **Newton-Schulz coefficients**: `(3.4445, -4.7750, 2.0315)` quintic polynomial
- **Orthogonality tolerance**: ~2-15 error (by design, not exact)
- **Mock pattern**: `FSDPMockContext()` for atomic mock application
- **Matrix fixtures**: 6 standardized matrices (256x256, 512x512, tall, wide, small, large)
- **Auto-skip CUDA**: Tests marked `@cuda` auto-skip when GPU unavailable

## COMMANDS

```bash
# Install
pip install -e ".[dev]"

# Test (64 tests)
pytest tests/ -v
pytest tests/ -m "not slow"           # Skip slow tests
pytest tests/test_optimizer.py -v      # Specific file

# Coverage
pytest tests/ --cov=muon_fsdp

# Lint (requires setup)
ruff check .
mypy muon_fsdp/ --ignore-missing-imports

# Examples
python examples/train_gpt.py --model gpt2 --epochs 3
python examples/mixed_adamw.py --model gpt2
```

## NOTES

- **FSDP2 required**: `torch.distributed.fsdp` must be available
- **CPU offload unsupported**: Newton-Schulz requires GPU
- **No sparse gradients**: Sparse gradients raise `RuntimeError`
- **Memory**: All-gather gradients = `world_size * param_size`
- **State dict**: Includes `step_count` for FSDP optimizer
