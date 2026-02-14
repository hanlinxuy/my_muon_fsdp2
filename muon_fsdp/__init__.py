"""Muon FSDP2 - Muon optimizer with Fully Sharded Data Parallel support.

A PyTorch implementation of the Muon optimizer, designed for efficient
distributed training with FSDP. This package provides core utilities
including Newton-Schulz iteration for orthogonal initialization and
distributed communication primitives.

Main Components:
- optimizer: MuonOptimizer class for model optimization
- utils: Core numerical utilities (Newton-Schulz iteration)
- distributed: Distributed communication primitives for FSDP
- fsdp: FSDPMuonOptimizer for FSDP2 distributed training
- sso: SpectralSphereOptimizer for spectral constraint optimization
- spectral: Spectral operations (power iteration, bisection solver)
- hdsp: HDSPMuonOptimizer for HSDP (Hybrid Data Sharding Parallel)

Example:
    >>> import torch
    >>> from muon_fsdp import MuonOptimizer, zeropower_via_newtonschulz5
    >>> model = torch.nn.Linear(512, 512)
    >>> optimizer = MuonOptimizer(model.parameters(), lr=0.02, momentum=0.95)
    >>> G = torch.randn(512, 512)
    >>> W = zeropower_via_newtonschulz5(G)
"""

__version__ = "0.1.0"
__author__ = "Muon FSDP Contributors"

from muon_fsdp.distributed import all_gather_grads, scatter_updates
from muon_fsdp.fsdp import (
    FSDPMuonOptimizer,
    collect_fsdp_modules,
    create_fsdp_muon_optimizer,
    get_dtensor_full_tensor,
    get_dtensor_local_tensor,
    has_fsdp_modules,
    is_dtensor,
)
from muon_fsdp.hdsp import (
    HDSPMuonOptimizer,
    HSDPConfig,
    create_device_mesh_2d,
    gather_grads_group,
    get_hsdp_groups,
)
from muon_fsdp.optimizer import MuonOptimizer
from muon_fsdp.spectral import (
    apply_spectral_retraction,
    bisect_spectral_radius,
    compute_spectral_norm,
    compute_target_radius,
    power_iteration,
)
from muon_fsdp.sso import SpectralSphereOptimizer
from muon_fsdp.utils import zeropower_via_newtonschulz5

__all__ = [
    "MuonOptimizer",
    "zeropower_via_newtonschulz5",
    "all_gather_grads",
    "scatter_updates",
    "FSDPMuonOptimizer",
    "create_fsdp_muon_optimizer",
    "is_dtensor",
    "get_dtensor_local_tensor",
    "get_dtensor_full_tensor",
    "collect_fsdp_modules",
    "has_fsdp_modules",
    "SpectralSphereOptimizer",
    "compute_spectral_norm",
    "power_iteration",
    "compute_target_radius",
    "apply_spectral_retraction",
    "bisect_spectral_radius",
    "HDSPMuonOptimizer",
    "HSDPConfig",
    "create_device_mesh_2d",
    "gather_grads_group",
    "get_hsdp_groups",
    "__version__",
]
