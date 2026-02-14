"""Mock utilities for HDSP testing.

提供 HSDP (Hybrid Data Sharding Parallel) 测试所需的 mock 实现，
无需真实分布式环境或 GPU 即可进行测试。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


class MockDeviceMesh:
    """Mock 2D DeviceMesh for HSDP testing.

    模拟 PyTorch 的 DeviceMesh，用于测试 2D 并行配置。

    Attributes:
        device_type: 设备类型 ("cuda" 或 "cpu")
        mesh: 2D 设备网格
        mesh_dim_names: 维度名称 ("dp_replicate", "fsdp_shard")
    """

    def __init__(
        self,
        device_type: str = "cuda",
        mesh: Optional[List[List[int]]] = None,
        mesh_dim_names: Optional[Tuple[str, str]] = None,
    ):
        self.device_type = device_type
        self.mesh = mesh or [[0, 1], [2, 3]]
        self.mesh_dim_names = mesh_dim_names or ("dp_replicate", "fsdp_shard")
        self._groups: Dict[str, Any] = {}

    def get_group(self, dim_name: str) -> Optional[Any]:
        """获取指定维度的进程组"""
        return self._groups.get(dim_name)

    def __repr__(self) -> str:
        return f"MockDeviceMesh(mesh={self.mesh}, dims={self.mesh_dim_names})"


class MockHSDPTensor:
    """Mock HSDP Tensor for testing.

    模拟 HSDP 模式下的张量，支持节点内分片和节点间复制。
    """

    def __init__(
        self,
        local_tensor: torch.Tensor,
        full_shape: Optional[Tuple[int, ...]] = None,
        mesh: Optional[MockDeviceMesh] = None,
        is_replicated: bool = False,
    ):
        self._local_tensor = local_tensor
        self._full_shape = full_shape or local_tensor.shape
        self._mesh = mesh
        self._is_replicated = is_replicated

    def to_local(self) -> torch.Tensor:
        """返回本地分片"""
        return self._local_tensor

    def full_tensor(self) -> torch.Tensor:
        """返回完整张量"""
        return self._local_tensor

    @property
    def shape(self) -> torch.Size:
        return torch.Size(self._full_shape)

    @property
    def dtype(self) -> torch.dtype:
        return self._local_tensor.dtype

    @property
    def device(self) -> torch.device:
        return self._local_tensor.device

    def __repr__(self) -> str:
        return f"MockHSDPTensor(shape={self._full_shape}, replicated={self._is_replicated})"


class MockHSDPModule:
    """Mock HSDP Module for testing.

    模拟 HSDP 包装的模块，支持 2D DeviceMesh 配置。
    """

    def __init__(
        self,
        module: nn.Module,
        mesh: Optional[MockDeviceMesh] = None,
    ):
        self._module = module
        self._mesh = mesh
        self._sharded_params: Dict[str, torch.Tensor] = {}

    def unshard(self) -> MagicMock:
        """返回取消分片的 mock 上下文管理器"""
        mock_handle = MagicMock()
        mock_handle.__enter__ = MagicMock(return_value=None)
        mock_handle.__exit__ = MagicMock(return_value=None)
        return mock_handle

    def reshard(self) -> None:
        """Mock 重新分片操作"""
        pass

    def parameters(self, recurse: bool = True):
        return self._module.parameters(recurse)

    def named_parameters(self, recurse: bool = True):
        return self._module.named_parameters(recurse)

    def modules(self, recurse: bool = True):
        if recurse:
            yield from self._module.modules()
        else:
            yield self._module

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return getattr(self._module, name)


def mock_hsdp_fully_shard(
    module: nn.Module,
    mesh: Optional[MockDeviceMesh] = None,
    **kwargs: Any,
) -> MockHSDPModule:
    """Mock 实现 torch.distributed.fsdp.fully_shard for HSDP

    Args:
        module: 要包装的模块
        mesh: 2D DeviceMesh

    Returns:
        MockHSDPModule 包装输入模块
    """
    return MockHSDPModule(module, mesh)


def mock_create_device_mesh_2d(
    dp_size: int,
    fsdp_size: int,
) -> MockDeviceMesh:
    """创建 Mock 2D DeviceMesh

    Args:
        dp_size: 数据并行大小
        fsdp_size: FSDP 分片大小

    Returns:
        MockDeviceMesh 对象
    """
    import itertools

    mesh = [list(range(i, i + fsdp_size)) for i in range(0, dp_size * fsdp_size, fsdp_size)]
    return MockDeviceMesh(
        device_type="cuda",
        mesh=mesh,
        mesh_dim_names=("dp_replicate", "fsdp_shard"),
    )


def mock_get_hsdp_groups(
    mesh: MockDeviceMesh,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Mock 获取 HSDP 进程组

    Args:
        mesh: DeviceMesh

    Returns:
        (dp_group, fsdp_group) 元组
    """
    return MagicMock(), MagicMock()


def mock_gather_grads_group(
    grads: List[torch.Tensor],
    group: Any,
) -> List[torch.Tensor]:
    """Mock 组内梯度聚合

    在 mock 模式下直接返回输入梯度
    """
    return grads


class HSDPMockContext:
    """Context manager for applying HDSP mocks"""

    def __init__(self, world_size: int = 1, num_nodes: int = 1):
        self.world_size = world_size
        self.num_nodes = num_nodes
        self.patches: List[Any] = []

    def __enter__(self) -> "HSDPMockContext":
        self.patches.append(patch("torch.distributed.is_available", lambda: False))
        self.patches.append(patch("torch.distributed.get_world_size", lambda: self.world_size))
        self.patches.append(patch("torch.distributed.get_rank", lambda: 0))

        try:
            self.patches.append(patch("torch.distributed.fsdp.FSDPModule", MockHSDPModule))
        except ImportError:
            pass

        try:
            self.patches.append(patch("torch.distributed.fsdp.fully_shard", mock_hsdp_fully_shard))
        except ImportError:
            pass

        for p in self.patches:
            p.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for p in self.patches:
            p.stop()


def create_mock_hsdp_model(
    model: nn.Module,
    dp_size: int = 1,
    fsdp_size: int = 1,
) -> MockHSDPModule:
    """创建 Mock HSDP 包装的模型

    Args:
        model: 要包装的模型
        dp_size: 数据并行大小
        fsdp_size: FSDP 分片大小

    Returns:
        MockHSDPModule
    """
    mesh = mock_create_device_mesh_2d(dp_size, fsdp_size)
    return MockHSDPModule(model, mesh)


def create_mock_hsdp_config(
    dp_size: int = 2,
    fsdp_size: int = 2,
) -> Dict[str, Any]:
    """创建 Mock HSDP 配置

    Args:
        dp_size: 数据并行大小
        fsdp_size: FSDP 分片大小

    Returns:
        配置字典
    """
    mesh = mock_create_device_mesh_2d(dp_size, fsdp_size)
    dp_group, fsdp_group = mock_get_hsdp_groups(mesh)

    return {
        "dp_size": dp_size,
        "fsdp_size": fsdp_size,
        "device_mesh": mesh,
        "dp_group": dp_group,
        "fsdp_group": fsdp_group,
    }
