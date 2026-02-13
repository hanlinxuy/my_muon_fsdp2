"""Mock utilities for FSDP2 testing.

This module provides mock implementations of FSDP2 components for testing
without requiring a distributed environment or GPU.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


class MockDTensor:
    """Mock DTensor for testing FSDP2 integration."""

    def __init__(
        self,
        local_tensor: torch.Tensor,
        full_shape: Optional[Tuple[int, ...]] = None,
        device_mesh: Optional[Any] = None,
    ):
        """Initialize mock DTensor.

        Args:
            local_tensor: The local tensor shard.
            full_shape: The full shape of the tensor (before sharding).
            device_mesh: Mock device mesh.
        """
        self._local_tensor = local_tensor
        self._full_shape = full_shape or local_tensor.shape
        self._device_mesh = device_mesh or MagicMock()

    def to_local(self) -> torch.Tensor:
        """Return the local tensor shard."""
        return self._local_tensor

    def full_tensor(self) -> torch.Tensor:
        """Return the full tensor (simulated by repeating local tensor)."""
        # For mock purposes, just return the local tensor
        # In real FSDP2, this would gather from all processes
        return self._local_tensor

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the full tensor."""
        return torch.Size(self._full_shape)

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the local tensor."""
        return self._local_tensor.dtype

    @property
    def device(self) -> torch.device:
        """Return the device of the local tensor."""
        return self._local_tensor.device

    def __repr__(self) -> str:
        return f"MockDTensor(shape={self._full_shape}, local_shape={self._local_tensor.shape})"


class MockFSDPModule:
    """Mock FSDPModule for testing."""

    def __init__(self, module: nn.Module):
        """Initialize mock FSDP module.

        Args:
            module: The wrapped module.
        """
        self._module = module
        self._sharded_params: Dict[str, torch.Tensor] = {}

    def unshard(self) -> MagicMock:
        """Return a mock context manager for unsharding."""
        mock_handle = MagicMock()
        mock_handle.__enter__ = MagicMock(return_value=None)
        mock_handle.__exit__ = MagicMock(return_value=None)
        return mock_handle

    def reshard(self) -> None:
        """Mock reshard operation."""
        pass

    def parameters(self, recurse: bool = True):
        """Return parameters of the wrapped module."""
        return self._module.parameters(recurse)

    def named_parameters(self, recurse: bool = True):
        """Return named parameters of the wrapped module."""
        return self._module.named_parameters(recurse)

    def modules(self, recurse: bool = True):
        """Return modules."""
        if recurse:
            yield from self._module.modules()
        else:
            yield self._module

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped module."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return getattr(self._module, name)


def mock_fully_shard(
    module: nn.Module,
    mesh: Optional[Any] = None,
    reshard_after_forward: bool = True,
) -> MockFSDPModule:
    """Mock implementation of torch.distributed.fsdp.fully_shard.

    Args:
        module: The module to wrap.
        mesh: Mock device mesh.
        reshard_after_forward: Whether to reshard after forward.

    Returns:
        MockFSDPModule wrapping the input module.
    """
    return MockFSDPModule(module)


def mock_is_dtensor(tensor: torch.Tensor) -> bool:
    """Mock implementation of is_dtensor check.

    Args:
        tensor: The tensor to check.

    Returns:
        True if tensor is a MockDTensor, False otherwise.
    """
    return isinstance(tensor, MockDTensor)


def mock_get_world_size() -> int:
    """Mock implementation of get_world_size.

    Returns:
        Always returns 1 for single-process testing.
    """
    return 1


def mock_get_rank() -> int:
    """Mock implementation of get_rank.

    Returns:
        Always returns 0 for single-process testing.
    """
    return 0


def mock_is_available() -> bool:
    """Mock implementation of distributed.is_available.

    Returns:
        Returns False to simulate non-distributed environment.
    """
    return False


def mock_all_gather(grads: List[torch.Tensor]) -> List[torch.Tensor]:
    """Mock implementation of all_gather_grads.

    In a real distributed setting, this would gather gradients from all processes.
    In mock mode, just returns the input gradients unchanged.

    Args:
        grads: List of gradient tensors.

    Returns:
        The same list of gradients (no actual gathering in mock).
    """
    return grads


class FSDPMockContext:
    """Context manager for applying FSDP2 mocks."""

    def __init__(self):
        """Initialize the mock context."""
        self.patches: List[Any] = []

    def __enter__(self) -> "FSDPMockContext":
        """Apply all mocks."""
        # Mock torch.distributed functions
        self.patches.append(patch("torch.distributed.is_available", mock_is_available))
        self.patches.append(
            patch("torch.distributed.get_world_size", mock_get_world_size)
        )
        self.patches.append(patch("torch.distributed.get_rank", mock_get_rank))

        # Mock FSDP2 imports if they exist
        try:
            self.patches.append(
                patch(
                    "torch.distributed.fsdp.FSDPModule",
                    MockFSDPModule,
                )
            )
        except ImportError:
            pass

        try:
            self.patches.append(
                patch(
                    "torch.distributed.fsdp.fully_shard",
                    mock_fully_shard,
                )
            )
        except ImportError:
            pass

        try:
            self.patches.append(
                patch(
                    "torch.distributed.tensor.DTensor",
                    MockDTensor,
                )
            )
        except ImportError:
            pass

        # Apply all patches
        for p in self.patches:
            p.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Remove all mocks."""
        for p in self.patches:
            p.stop()


def create_mock_dtensor(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> MockDTensor:
    """Create a mock DTensor with the given shape.

    Args:
        shape: The full shape of the tensor.
        dtype: The data type.
        device: The device.

    Returns:
        A MockDTensor instance.
    """
    local_tensor = torch.randn(*shape, dtype=dtype, device=device)
    return MockDTensor(local_tensor, full_shape=shape)


def create_mock_fsdp_model(
    model: nn.Module,
    world_size: int = 1,
) -> MockFSDPModule:
    """Create a mock FSDP-wrapped model.

    Args:
        model: The model to wrap.
        world_size: Simulated world size.

    Returns:
        MockFSDPModule wrapping the model.
    """
    fsdp_module = MockFSDPModule(model)

    # Simulate parameter sharding by creating MockDTensors
    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() >= 2:
            # Create mock DTensor for 2D+ parameters
            mock_tensor = create_mock_dtensor(
                param.shape,
                dtype=param.dtype,
                device=param.device,
            )
            fsdp_module._sharded_params[name] = mock_tensor

    return fsdp_module
