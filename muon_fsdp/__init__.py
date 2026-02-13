"""Muon FSDP2 - Muon optimizer with Fully Sharded Data Parallel support.

A PyTorch implementation of the Muon optimizer, designed for efficient
distributed training with FSDP. This package provides core utilities
including Newton-Schulz iteration for orthogonal initialization and
distributed communication primitives.

Main Components:
- utils: Core numerical utilities (Newton-Schulz iteration)
- distributed: Distributed communication primitives for FSDP

Example:
    >>> import torch
    >>> from muon_fsdp import zeropower_via_newtonschulz5
    >>> G = torch.randn(512, 512)
    >>> W = zeropower_via_newtonschulz5(G)
"""

__version__ = "0.1.0"
__author__ = "Muon FSDP Contributors"

from muon_fsdp.utils import zeropower_via_newtonschulz5
from muon_fsdp.distributed import all_gather_grads, scatter_updates

__all__ = [
    "zeropower_via_newtonschulz5",
    "all_gather_grads",
    "scatter_updates",
    "__version__",
]
