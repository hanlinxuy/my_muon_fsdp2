"""FSDP2 工具库 - FSDP 模块管理工具。"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, List, Optional

import torch
import torch.nn as nn

from .dtensor import get_local_tensor, is_dtensor
from .comm import is_distributed

logger = logging.getLogger(__name__)

# 尝试导入 FSDP2，优雅处理不可用情况
HAS_FSDP2 = False
try:
    from torch.distributed.fsdp import FSDPModule, fully_shard
    from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision

    HAS_FSDP2 = True
except ImportError:
    logger.warning("FSDP2 不可用，使用单进程模式")


def is_fsdp_module(module: nn.Module) -> bool:
    """检查模块是否为 FSDPModule。"""
    if not HAS_FSDP2:
        return False
    return isinstance(module, FSDPModule)


def collect_fsdp_modules(model: nn.Module) -> List[nn.Module]:
    """从模型中收集所有 FSDP 包装的模块。"""
    fsdp_modules = []
    if not HAS_FSDP2:
        return fsdp_modules

    for module in model.modules():
        if is_fsdp_module(module):
            fsdp_modules.append(module)
    return fsdp_modules


def has_fsdp_modules(model: nn.Module) -> bool:
    """检查模型是否具有任何 FSDP 包装的模块。"""
    return len(collect_fsdp_modules(model)) > 0


@contextmanager
def unshard_modules(modules: List[nn.Module]) -> Generator[None, None, None]:
    """取消分片 FSDP 模块的上下文管理器。"""
    if not HAS_FSDP2:
        yield
        return

    fsdp_modules = [m for m in modules if is_fsdp_module(m)]
    if not fsdp_modules:
        yield
        return

    handles = []
    try:
        for module in fsdp_modules:
            handle = module.unshard()
            handles.append(handle)
            handle.__enter__()
        yield
    finally:
        for handle in handles:
            try:
                handle.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"重分片时出错: {e}")


@contextmanager
def unshard_model(model: nn.Module) -> Generator[None, None, None]:
    """取消分片整个模型的上下文管理器。"""
    fsdp_modules = collect_fsdp_modules(model)
    with unshard_modules(fsdp_modules):
        yield


def get_model_params_with_grad(model: nn.Module) -> List[nn.Parameter]:
    """获取模型中所有需要梯度的参数。"""
    return [p for p in model.parameters() if p.requires_grad]


def get_local_params(params: List[nn.Parameter]) -> List[torch.Tensor]:
    """获取参数的局部数据列表。"""
    return [get_local_tensor(p) for p in params]


def get_local_grads(params: List[nn.Parameter]) -> List[Optional[torch.Tensor]]:
    """获取参数的局部梯度列表。"""
    grads: List[Optional[torch.Tensor]] = []
    for p in params:
        if p.grad is None:
            grads.append(None)
        else:
            grads.append(get_local_tensor(p.grad))
    return grads


def apply_local_updates(
    params: List[nn.Parameter],
    updates: List[torch.Tensor],
) -> None:
    """将更新应用到局部参数。"""
    for p, update in zip(params, updates):
        if update is None:
            continue
        local_p = get_local_tensor(p)
        with torch.no_grad():
            local_p.copy_(local_p + update)


__all__ = [
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
]
