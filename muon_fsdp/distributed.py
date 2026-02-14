"""Distributed communication utilities for Muon FSDP2.

This module provides distributed communication primitives for FSDP integration,
including gradient gathering and update scattering operations. These utilities
abstract away the complexity of PyTorch's distributed API and handle single-device
scenarios gracefully.
"""

from typing import List, Optional

import torch
import torch.distributed as dist


def is_available() -> bool:
    """Check if PyTorch distributed is available and initialized.

    Returns:
        True if distributed is available and properly initialized.
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get the number of distributed processes.

    Returns:
        Number of processes in the distributed group, or 1 if not distributed.
    """
    if is_available():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get the rank of the current process.

    Returns:
        Rank of current process, or 0 if not distributed.
    """
    if is_available():
        return dist.get_rank()
    return 0


def all_gather_grads(
    grads: List[torch.Tensor],
    group: Optional[dist.ProcessGroup] = None,
) -> List[torch.Tensor]:
    """Gather gradients from all processes in the distributed group.

    This function collects gradient tensors from all processes in the specified
    distributed group and concatenates them along the first dimension. This is
    useful for computing full-gradient updates in FSDP scenarios where gradients
    are sharded across processes.

    Args:
        grads: List of gradient tensors from the current process.
              Each tensor will be gathered from all processes and concatenated.
        group: Optional distributed process group. If None, uses the default
              (world) group.

    Returns:
        List of gathered gradient tensors. Each tensor in the list has shape
        (world_size * original_dim, ...) where the first dimension is the
        concatenation of gradients from all processes.

    Raises:
        RuntimeError: If called without distributed initialization.

    Example:
        >>> # In each process
        >>> local_grad = [torch.randn(128, 512)]
        >>> gathered = all_gather_grads(local_grad)
        >>> # gathered[0] has shape (world_size * 128, 512)
        >>> # containing gradients from all processes
    """
    if not is_available():
        # Single process / single GPU: return as-is
        return grads

    gathered_grads: List[torch.Tensor] = []

    for grad in grads:
        # Ensure gradient is contiguous
        grad = grad.contiguous()

        # Get tensor shape for gathering
        world_size = get_world_size()

        # Create output tensor with space for all processes
        output_shape = list(grad.shape)
        output_shape[0] *= world_size
        output = grad.new_zeros(output_shape)

        # Gather from all processes
        dist.all_gather_into_tensor(output, grad, group=group)

        gathered_grads.append(output)

    return gathered_grads


def scatter_updates(
    updates: List[torch.Tensor],
    src_rank: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> List[torch.Tensor]:
    """Scatter update tensors from source rank to all processes.

    This function takes update tensors from the source rank and scatters them
    evenly across all processes in the distributed group. Each process receives
    a slice of the updates based on its rank. This is useful for distributing
    computed updates in FSDP scenarios.

    Args:
        updates: List of update tensors to scatter. These should be the same
                across all processes (only src_rank's values are used).
        src_rank: Source rank that owns the full updates. Default is 0.
        group: Optional distributed process group. If None, uses the default
              (world) group.

    Returns:
        List of scattered update tensors. Each tensor has shape
        (original_dim / world_size, ...) for the first dimension, representing
        the slice assigned to this process.

    Raises:
        RuntimeError: If called without distributed initialization.

    Example:
        >>> # In each process, updates are the same (only src_rank=0's values matter)
        >>> full_updates = [torch.randn(512, 512)]  # Same on all processes
        >>> scattered = scatter_updates(full_updates)
        >>> # scattered[0] has shape (512/world_size, 512)
        >>> # Each process gets its assigned slice
    """
    if not is_available():
        # Single process / single GPU: return as-is
        return updates

    world_size = get_world_size()
    rank = get_rank()
    scattered_updates: List[torch.Tensor] = []

    for update in updates:
        # Ensure update is contiguous
        update = update.contiguous()

        # Calculate slice size per process
        dim_size = update.shape[0]
        slice_size = dim_size // world_size

        if dim_size % world_size != 0:
            raise ValueError(
                f"Update dimension ({dim_size}) must be divisible by world_size ({world_size})"
            )

        # Create output tensor for this process
        output_shape = list(update.shape)
        output_shape[0] = slice_size
        output = update.new_zeros(output_shape)

        if rank == src_rank:
            # Source rank: split and scatter
            input_tensor = update
        else:
            # Other ranks: create input tensor for scatter
            input_tensor = update.new_zeros(dim_size)

        # Scatter the update
        dist.scatter(
            output,
            src_tensor=input_tensor if rank == src_rank else None,
            src=src_rank,
            group=group,
        )

        scattered_updates.append(output)

    return scattered_updates
