"""FSDP2 工具库 - DTensor 操作模块。

本模块提供 PyTorch FSDP2 中 DTensor（分布式张量）的通用操作工具，
包括类型检查、局部/全局张量转换、分片维度获取等功能。
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# 尝试导入 DTensor，优雅处理不可用情况
try:
    from torch.distributed.tensor import DTensor, Shard, Replicate
    from torch.distributed.tensor import distribute_tensor
    from torch.distributed.device_mesh import DeviceMesh

    HAS_DTENSOR = True
except ImportError:
    HAS_DTENSOR = False
    logger.warning("DTensor 不可用，使用单进程模式")


def is_dtensor(tensor: Any) -> bool:
    """检查张量是否为 DTensor (FSDP2 分片参数)。

    参数:
        tensor: 要检查的对象。

    返回:
        如果是 DTensor 则返回 True，否则返回 False。
    """
    if not HAS_DTENSOR:
        return False
    return isinstance(tensor, DTensor)


def get_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """从张量获取局部数据。"""
    if is_dtensor(tensor) and HAS_DTENSOR:
        return tensor.to_local()
    return tensor


def get_full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """获取张量的全局完整数据。

    对于 DTensor，会收集所有分片；对于普通张量，直接返回。

    参数:
        tensor: DTensor 或常规张量。

    返回:
        完整张量数据。
    """
    if is_dtensor(tensor):
        return tensor.full_tensor()
    return tensor


def get_shard_dim(tensor: torch.Tensor) -> Optional[int]:
    """获取 DTensor 的分片维度。

    参数:
        tensor: DTensor 或常规张量。

    返回:
        分片维度索引，如果不是 DTensor 或没有分片则返回 None。
    """
    if not is_dtensor(tensor):
        return None

    placements = tensor.placements
    for idx, placement in enumerate(placements):
        if isinstance(placement, Shard):
            return placement.dim
    return None


def create_dtensor(
    tensor: torch.Tensor,
    device_mesh: Optional[DeviceMesh] = None,
    shard_dim: int = 0,
    *,
    replicate: bool = False,
) -> torch.Tensor:
    """创建 DTensor 分片张量。

    参数:
        tensor: 要分片的全局张量。
        device_mesh: 设备网格，如果为 None 则尝试使用默认网格。
        shard_dim: 分片维度（仅当 replicate=False 时使用）。
        replicate: 是否复制而不是分片。

    返回:
        DTensor 对象。

    异常:
        RuntimeError: 如果 DTensor 不可用或 device_mesh 无法创建。
    """
    if not HAS_DTENSOR:
        raise RuntimeError("DTensor 不可用")

    if device_mesh is None:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("需要初始化分布式环境或提供 device_mesh")
        device_mesh = DeviceMesh(
            device_type=tensor.device.type,
            mesh=list(range(dist.get_world_size())),
        )

    if replicate:
        placement = Replicate()
    else:
        placement = Shard(shard_dim)

    return distribute_tensor(tensor, device_mesh, [placement])


def redistribute_dtensor(
    tensor: torch.Tensor,
    device_mesh: Optional[DeviceMesh] = None,
    shard_dim: Optional[int] = None,
    *,
    replicate: bool = False,
) -> torch.Tensor:
    """重新分发 DTensor 到新的分片策略。

    参数:
        tensor: 要重新分发的 DTensor。
        device_mesh: 新的设备网格，None 表示保持不变。
        shard_dim: 新的分片维度，None 表示保持不变。
        replicate: 是否改为复制模式。

    返回:
        重新分发后的 DTensor。
    """
    if not is_dtensor(tensor):
        return tensor

    if not HAS_DTENSOR:
        return tensor

    if device_mesh is None:
        device_mesh = tensor.device_mesh

    if replicate:
        placements = [Replicate()]
    elif shard_dim is not None:
        placements = [Shard(shard_dim)]
    else:
        placements = tensor.placements

    return redistribute(tensor, device_mesh, placements)


def dtensor_meta_info(tensor: torch.Tensor) -> dict[str, Any]:
    """获取 DTensor 的元信息。

    参数:
        tensor: DTensor 或普通张量。

    返回:
        包含元信息的字典：
        - is_dtensor: 是否为 DTensor
        - local_shape: 局部张量形状
        - global_shape: 全局张量形状
        - shard_dim: 分片维度（如果有）
        - device_mesh_shape: 设备网格形状（如果有）
    """
    info = {
        "is_dtensor": False,
        "local_shape": tensor.shape,
        "global_shape": tensor.shape,
        "shard_dim": None,
        "device_mesh_shape": None,
    }

    if is_dtensor(tensor):
        info["is_dtensor"] = True
        info["global_shape"] = tensor.shape
        info["local_shape"] = tensor.to_local().shape
        info["shard_dim"] = get_shard_dim(tensor)
        if hasattr(tensor, "device_mesh"):
            info["device_mesh_shape"] = tensor.device_mesh.shape

    return info


__all__ = [
    "is_dtensor",
    "get_local_tensor",
    "get_full_tensor",
    "get_shard_dim",
    "create_dtensor",
    "redistribute_dtensor",
    "dtensor_meta_info",
    "HAS_DTENSOR",
]
