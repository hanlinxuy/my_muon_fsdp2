"""FSDP2 工具库 - 分布式通信模块。

本模块提供通用的 PyTorch 分布式通信原语，包括梯度收集、更新分片、
聚合操作等，优雅处理单设备和多设备场景。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def is_distributed() -> bool:
    """检查 PyTorch 分布式是否可用并已初始化。

    返回:
        如果分布式可用且已正确初始化则返回 True。
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """获取分布式进程数量。

    返回:
        分布式组中的进程数，如果未使用分布式则返回 1。
    """
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """获取当前进程的 rank。

    返回:
        当前进程的 rank，如果未使用分布式则返回 0。
    """
    if is_distributed():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """获取当前进程的本地 rank。

    返回:
        当前进程在节点内的本地 rank，如果未使用分布式则返回 0。
    """
    if is_distributed():
        try:
            return dist.get_local_rank()
        except (AttributeError, RuntimeError):
            return get_rank()
    return 0


def all_gather(
    tensors: List[torch.Tensor],
    group: Optional[dist.ProcessGroup] = None,
) -> List[torch.Tensor]:
    """从分布式组中的所有进程收集张量。

    此函数从所有进程收集张量，并将它们沿第一维连接。

    参数:
        tensors: 当前进程的张量列表。
        group: 可选的分布式进程组。

    返回:
        收集的张量列表。
    """
    if not is_distributed():
        return tensors

    world_size = get_world_size()
    gathered: List[torch.Tensor] = []

    for tensor in tensors:
        tensor = tensor.contiguous()
        output_shape = list(tensor.shape)
        output_shape[0] *= world_size
        output = tensor.new_zeros(output_shape)
        dist.all_gather_into_tensor(output, tensor, group=group)
        gathered.append(output)

    return gathered


def all_gather_grads(
    grads: List[torch.Tensor],
    group: Optional[dist.ProcessGroup] = None,
) -> List[torch.Tensor]:
    """从分布式组中的所有进程收集梯度。

    参数:
        grads: 当前进程的梯度张量列表。
        group: 可选的分布式进程组。

    返回:
        收集的梯度张量列表。
    """
    return all_gather(grads, group=group)


def scatter(
    tensors: List[torch.Tensor],
    src_rank: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> List[torch.Tensor]:
    """将张量从源 rank 分片到所有进程。

    参数:
        tensors: 要分片的张量列表（所有进程应相同）。
        src_rank: 拥有完整张量的源 rank。
        group: 可选的分布式进程组。

    返回:
        分片后的张量列表。
    """
    if not is_distributed():
        return tensors

    world_size = get_world_size()
    rank = get_rank()
    scattered: List[torch.Tensor] = []

    for tensor in tensors:
        tensor = tensor.contiguous()
        dim_size = tensor.shape[0]
        slice_size = dim_size // world_size

        if dim_size % world_size != 0:
            raise ValueError(f"维度 {dim_size} 不能被 world_size {world_size} 整除")

        output_shape = list(tensor.shape)
        output_shape[0] = slice_size
        output = tensor.new_zeros(output_shape)

        if rank == src_rank:
            input_tensor = tensor
        else:
            input_tensor = tensor.new_zeros(dim_size)

        dist.scatter(
            output,
            scatter_list=list(input_tensor.chunk(world_size)) if rank == src_rank else None,
            src=src_rank,
            group=group,
        )

        scattered.append(output)

    return scattered


def scatter_updates(
    updates: List[torch.Tensor],
    src_rank: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> List[torch.Tensor]:
    """将更新张量从源 rank 分片到所有进程。

    参数:
        updates: 要分片的更新张量列表。
        src_rank: 拥有完整更新的源 rank。
        group: 可选的分布式进程组。

    返回:
        分片后的更新张量列表。
    """
    return scatter(updates, src_rank=src_rank, group=group)


def all_reduce(
    tensors: List[torch.Tensor],
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
) -> List[torch.Tensor]:
    """对所有进程的张量进行归约操作。

    参数:
        tensors: 要归约的张量列表。
        op: 归约操作类型（SUM, AVG, MAX, MIN 等）。
        group: 可选的分布式进程组。

    返回:
        归约后的张量列表（原地修改）。
    """
    if not is_distributed():
        return tensors

    for tensor in tensors:
        dist.all_reduce(tensor, op=op, group=group)

    return tensors


def all_reduce_sum(
    tensors: List[torch.Tensor], group: Optional[dist.ProcessGroup] = None
) -> List[torch.Tensor]:
    """对所有进程的张量进行求和归约。"""
    return all_reduce(tensors, op=dist.ReduceOp.SUM, group=group)


def all_reduce_avg(
    tensors: List[torch.Tensor], group: Optional[dist.ProcessGroup] = None
) -> List[torch.Tensor]:
    """对所有进程的张量进行平均归约。"""
    all_reduce(tensors, op=dist.ReduceOp.SUM, group=group)
    world_size = get_world_size()
    if world_size > 1:
        for tensor in tensors:
            tensor.div_(world_size)
    return tensors


def broadcast(
    tensors: List[torch.Tensor],
    src_rank: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> List[torch.Tensor]:
    """将张量从源 rank 广播到所有进程。

    参数:
        tensors: 要广播的张量列表。
        src_rank: 源 rank。
        group: 可选的分布式进程组。

    返回:
        广播后的张量列表（原地修改）。
    """
    if not is_distributed():
        return tensors

    for tensor in tensors:
        dist.broadcast(tensor, src=src_rank, group=group)

    return tensors


def barrier(group: Optional[dist.ProcessGroup] = None) -> None:
    """同步所有进程。"""
    if is_distributed():
        dist.barrier(group=group)


__all__ = [
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
]
