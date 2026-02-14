"""FSDP2 工具库 - 通用 PyTorch FSDP2 工具库。

本库提供 PyTorch FSDP2 下的通用操作和优化器抽象，包括：
- DTensor 操作工具
- 分布式通信原语
- FSDP 模块管理
- 优化器抽象基类
- Muon 等优化器实现

示例:
    >>> import torch
    >>> import torch.nn as nn
    >>> from fsdp2_utils import MuonOptimizer
    >>> model = nn.Linear(512, 512)
    >>> optimizer = MuonOptimizer(model, lr=0.02, momentum=0.95)
"""

__version__ = "0.2.0"
__author__ = "FSDP2 Utils Contributors"

from .comm import (
    all_gather,
    all_gather_grads,
    all_reduce,
    all_reduce_avg,
    all_reduce_sum,
    barrier,
    broadcast,
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    scatter,
    scatter_updates,
)
from .dtensor import (
    dtensor_meta_info,
    get_full_tensor,
    get_local_tensor,
    get_shard_dim,
    is_dtensor,
    HAS_DTENSOR,
)
from .fsdp_utils import (
    apply_local_updates,
    collect_fsdp_modules,
    get_local_grads,
    get_local_params,
    get_model_params_with_grad,
    has_fsdp_modules,
    is_fsdp_module,
    unshard_model,
    unshard_modules,
    HAS_FSDP2,
)
from .base import FSDPCompatibleOptimizer
from .muon import MuonOptimizer
from .numerical import (
    compute_spectral_norm,
    power_iteration,
    zeropower_via_newtonschulz5,
)

__all__ = [
    "FSDPCompatibleOptimizer",
    "MuonOptimizer",
    "zeropower_via_newtonschulz5",
    "power_iteration",
    "compute_spectral_norm",
    "is_dtensor",
    "get_local_tensor",
    "get_full_tensor",
    "get_shard_dim",
    "dtensor_meta_info",
    "HAS_DTENSOR",
    "is_distributed",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "all_gather",
    "all_gather_grads",
    "scatter",
    "scatter_updates",
    "all_reduce",
    "all_reduce_sum",
    "all_reduce_avg",
    "broadcast",
    "barrier",
    "is_fsdp_module",
    "collect_fsdp_modules",
    "has_fsdp_modules",
    "unshard_modules",
    "unshard_model",
    "get_model_params_with_grad",
    "get_local_params",
    "get_local_grads",
    "apply_local_updates",
    "HAS_FSDP2",
    "__version__",
]
