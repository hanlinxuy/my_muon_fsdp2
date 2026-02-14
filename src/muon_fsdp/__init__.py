"""Muon FSDP2 - 支持完全分片数据并行的 Muon 优化器。

PyTorch 实现的 Muon 优化器，专为 FSDP 的高效分布式训练而设计。本包提供核心工具，
包括用于正交初始化的 Newton-Schulz 迭代和分布式通信原语。

主要组件:
- optimizer: 用于模型优化的 MuonOptimizer 类
- utils: 核心数值工具 (Newton-Schulz 迭代)
- distributed: FSDP 分布式通信原语
- fsdp: 用于 FSDP2 分布式训练的 FSDPMuonOptimizer
- sso: 用于谱约束优化的 SpectralSphereOptimizer
- spectral: 谱操作 (幂迭代, 二分搜索求解器)
- hdsp: 用于 HSDP (混合数据分片并行) 的 HDSPMuonOptimizer

示例:
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
