"""Muon FSDP2 分布式通信工具。

本模块提供用于 FSDP 集成的分布式通信原语，包括梯度收集和更新分片操作。
这些工具抽象了 PyTorch 分布式 API 的复杂性，并优雅地处理单设备场景。
"""

from typing import List, Optional

import torch
import torch.distributed as dist


def is_available() -> bool:
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
    if is_available():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """获取当前进程的 rank。

    返回:
        当前进程的 rank，如果未使用分布式则返回 0。
    """
    if is_available():
        return dist.get_rank()
    return 0


def all_gather_grads(
    grads: List[torch.Tensor],
    group: Optional[dist.ProcessGroup] = None,
) -> List[torch.Tensor]:
    """从分布式组中的所有进程收集梯度。

    此函数从指定分布式组中的所有进程收集梯度张量，并将它们沿第一维连接。
    这对于在梯度跨进程分片的 FSDP 场景中计算完整梯度更新非常有用。

    参数:
        grads: 当前进程的梯度张量列表。
              每个张量将从所有进程收集并连接。
        group: 可选的分布式进程组。如果为 None，则使用默认 (world) 组。

    返回:
        收集的梯度张量列表。列表中每个张量的形状为
        (world_size * original_dim, ...)，其中第一维是所有进程梯度的连接。

    异常:
        RuntimeError: 如果在没有分布式初始化的情况下调用。

    示例:
        >>> # 在每个进程中
        >>> local_grad = [torch.randn(128, 512)]
        >>> gathered = all_gather_grads(local_grad)
        >>> # gathered[0] 的形状为 (world_size * 128, 512)
        >>> # 包含来自所有进程的梯度
    """
    if not is_available():
        # 单进程/单 GPU: 原样返回
        return grads

    gathered_grads: List[torch.Tensor] = []

    for grad in grads:
        # 确保梯度是连续的
        grad = grad.contiguous()

        # 获取用于收集的张量形状
        world_size = get_world_size()

        # 创建用于所有进程的空间的输出张量
        output_shape = list(grad.shape)
        output_shape[0] *= world_size
        output = grad.new_zeros(output_shape)

        # 从所有进程收集
        dist.all_gather_into_tensor(output, grad, group=group)

        gathered_grads.append(output)

    return gathered_grads


def scatter_updates(
    updates: List[torch.Tensor],
    src_rank: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> List[torch.Tensor]:
    """将更新张量从源 rank 分片到所有进程。

    此函数从源 rank 获取更新张量，并将它们均匀分片到分布式组中的所有进程。
    每个进程根据其 rank 接收更新的一片。这对于在 FSDP 场景中
    分发计算出的更新非常有用。

    参数:
        updates: 要分片的更新张量列表。这些在所有进程中应该相同
                (只使用 src_rank 的值)。
        src_rank: 拥有完整更新的源 rank。默认为 0。
        group: 可选的分布式进程组。如果为 None，则使用默认 (world) 组。

    返回:
        分片的更新张量列表。每个张量的形状为
        (original_dim / world_size, ...)，表示分配给此进程的一片。

    异常:
        RuntimeError: 如果在没有分布式初始化的情况下调用。

    示例:
        >>> # 在每个进程中，更新是相同的 (只使用 src_rank=0 的值)
        >>> full_updates = [torch.randn(512, 512)]  # 在所有进程中相同
        >>> scattered = scatter_updates(full_updates)
        >>> # scattered[0] 的形状为 (512/world_size, 512)
        >>> # 每个进程获得其分配的一片
    """
    if not is_available():
        # 单进程/单 GPU: 原样返回
        return updates

    world_size = get_world_size()
    rank = get_rank()
    scattered_updates: List[torch.Tensor] = []

    for update in updates:
        # 确保更新是连续的
        update = update.contiguous()

        # 计算每个进程的片大小
        dim_size = update.shape[0]
        slice_size = dim_size // world_size

        if dim_size % world_size != 0:
            raise ValueError(f"更新维度 ({dim_size}) 必须能被 world_size ({world_size}) 整除")

        # 为此进程创建输出张量
        output_shape = list(update.shape)
        output_shape[0] = slice_size
        output = update.new_zeros(output_shape)

        if rank == src_rank:
            # 源 rank: 分割并分片
            input_tensor = update
        else:
            # 其他 rank: 为分片创建输入张量
            input_tensor = update.new_zeros(dim_size)

        # 分片更新
        dist.scatter(
            output,
            src_tensor=input_tensor if rank == src_rank else None,
            src=src_rank,
            group=group,
        )

        scattered_updates.append(output)

    return scattered_updates
