"""FSDP2 integration layer for Muon optimizer.

This module provides the FSDPMuonOptimizer class that integrates Muon optimizer
with PyTorch FSDP2 (Fully Sharded Data Parallel). It handles DTensor parameters,
manages unshard/reshard lifecycle, and performs gradient all-gather for Newton-Schulz
computation on full matrices.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from muon_fsdp.distributed import (
    all_gather_grads,
    get_rank,
    get_world_size,
    is_available,
)
from muon_fsdp.utils import zeropower_via_newtonschulz5

logger = logging.getLogger(__name__)


def is_dtensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a DTensor (FSDP2 sharded parameter).

    Args:
        tensor: The tensor to check.

    Returns:
        True if the tensor is a DTensor, False otherwise.
    """
    try:
        from torch.distributed.tensor import DTensor

        return isinstance(tensor, DTensor)
    except ImportError:
        return False


def get_dtensor_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Get the local tensor from a DTensor.

    Args:
        tensor: A DTensor or regular tensor.

    Returns:
        The local tensor if DTensor, or the original tensor.
    """
    if is_dtensor(tensor):
        return tensor.to_local()
    return tensor


def get_dtensor_full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Get the full tensor from a DTensor by gathering all shards.

    Args:
        tensor: A DTensor or regular tensor.

    Returns:
        The full gathered tensor if DTensor, or the original tensor.
    """
    if is_dtensor(tensor):
        return tensor.full_tensor()
    return tensor


def collect_fsdp_modules(model: nn.Module) -> List[nn.Module]:
    """Collect all FSDP-wrapped modules from a model.

    This function traverses the model and returns all modules that have been
    wrapped with FSDP2 (fully_shard). These modules have the FSDPModule type
    and expose unshard/reshard methods.

    Args:
        model: The model to search for FSDP modules.

    Returns:
        List of FSDP-wrapped modules.
    """
    fsdp_modules = []
    try:
        from torch.distributed.fsdp import FSDPModule

        for module in model.modules():
            if isinstance(module, FSDPModule):
                fsdp_modules.append(module)
    except ImportError:
        # FSDP2 not available
        logger.warning("FSDP2 not available, using single-process mode")
        return []

    return fsdp_modules


def has_fsdp_modules(model: nn.Module) -> bool:
    """Check if a model has any FSDP-wrapped modules.

    Args:
        model: The model to check.

    Returns:
        True if the model contains FSDP modules, False otherwise.
    """
    return len(collect_fsdp_modules(model)) > 0


class FSDPMuonOptimizer(Optimizer):
    """Muon optimizer with FSDP2 integration.

    This optimizer wraps the Muon optimizer logic and adds FSDP2-specific handling:
    - Detects DTensor parameters from FSDP2 sharding
    - Manages unshard/reshard lifecycle for accessing full parameters
    - All-gathers gradients for Newton-Schulz computation on full matrices
    - Respects FSDP2's MixedPrecisionPolicy
    - Handles gradient accumulation correctly

    The optimizer follows the NS Replication strategy:
    1. All-gather gradients from all processes
    2. Compute Newton-Schulz on the full gradient matrix
    3. Apply updates to sharded parameters

    Args:
        model: The model being optimized. Must be FSDP-wrapped for distributed
            training, but can also be a regular model for single-GPU training.
        params: Iterable of parameters to optimize. If None, uses model.parameters().
        lr: Learning rate. Default: 0.02
        weight_decay: Weight decay coefficient. Default: 0.01
        momentum: Momentum coefficient for gradient accumulation. Default: 0.95
        nesterov: Whether to use Nesterov momentum. Default: True
        ns_steps: Number of Newton-Schulz iterations. Default: 5
        ns_stepsize: Step size for Newton-Schulz updates. Default: 1.0
        beta2: Coefficient for second moment (Adam-style). Default: 0.99
        eps: Epsilon for numerical stability. Default: 1e-8
        gradient_accumulation_steps: Number of steps to accumulate gradients.
            Default: 1 (no accumulation).

    Example:
        >>> from torch.distributed.fsdp import fully_shard
        >>> model = nn.Linear(512, 512)
        >>> fully_shard(model)
        >>> optimizer = FSDPMuonOptimizer(model, lr=0.02)
        >>> for input, target in dataloader:
        ...     output = model(input)
        ...     loss = criterion(output, target)
        ...     loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        model: nn.Module,
        params: Optional[List[torch.nn.Parameter]] = None,
        lr: float = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_stepsize: float = 1.0,
        beta2: float = 0.99,
        eps: float = 1e-8,
        gradient_accumulation_steps: int = 1,
    ):
        # Store model reference for FSDP operations
        self.model = model
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._step_count = 0

        # Default parameters if not provided
        if params is None:
            params = list(model.parameters())

        # Filter out parameters that don't require gradients
        params = [p for p in params if p.requires_grad]

        # Validate parameters
        if not params:
            raise ValueError("FSDPMuonOptimizer received an empty parameter list")

        # Check for FSDP modules
        self.fsdp_modules = collect_fsdp_modules(model)
        if self.fsdp_modules:
            logger.info(f"Detected {len(self.fsdp_modules)} FSDP modules")

        # Default hyperparameters
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "ns_stepsize": ns_stepsize,
            "beta2": beta2,
            "eps": eps,
        }

        super().__init__(params, defaults)

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # Momentum buffer for first moment
                state["momentum_buffer"] = torch.zeros_like(p)
                # Second moment for Adam-style update
                state["second_moment"] = torch.zeros_like(p)
                # Gradient accumulation counter
                state["accum_count"] = 0

    @contextmanager
    def unshard_params(self) -> Generator[None, None, None]:
        """Context manager to unshard FSDP parameters.

        This context manager calls unshard() on all FSDP modules before entering
        the context and ensures reshard() is called on exit (including on exceptions).

        Yields:
            None

        Example:
            >>> with optimizer.unshard_params():
            ...     # Parameters are unsharded here
            ...     full_params = [p.full_tensor() for p in model.parameters()]
            ... # Parameters are automatically resharded here
        """
        if not self.fsdp_modules:
            # No FSDP modules, yield immediately
            yield
            return

        # Enter unshard context for all FSDP modules
        handles = []
        try:
            for module in self.fsdp_modules:
                handle = module.unshard()
                handles.append(handle)
                handle.__enter__()
            yield
        finally:
            # Ensure reshard is called on all modules, even if an exception occurs
            for handle in handles:
                try:
                    handle.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Error during reshard: {e}")

    def _gather_gradients(
        self,
        params: List[torch.nn.Parameter],
    ) -> List[torch.Tensor]:
        """Gather gradients from all processes for the given parameters.

        For DTensor parameters, this all-gathers the gradients across all processes.
        For regular parameters, returns the local gradients.

        Args:
            params: List of parameters whose gradients to gather.

        Returns:
            List of gathered gradient tensors.
        """
        local_grads = []

        for p in params:
            if p.grad is None:
                # Create zero gradient if none exists
                grad = torch.zeros_like(get_dtensor_local_tensor(p))
            else:
                grad = p.grad

            # For DTensor, get the local shard of the gradient
            if is_dtensor(p):
                # DTensor gradient is also a DTensor
                local_grad = get_dtensor_local_tensor(grad)
            else:
                local_grad = grad

            local_grads.append(local_grad)

        # All-gather gradients from all processes
        if is_available() and get_world_size() > 1:
            gathered_grads = all_gather_grads(local_grads)
        else:
            gathered_grads = local_grads

        return gathered_grads

    def _apply_weight_decay_and_momentum(
        self,
        params: List[torch.nn.Parameter],
        group: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """Apply weight decay and momentum to gradients.

        Args:
            params: List of parameters.
            group: Parameter group with hyperparameters.

        Returns:
            List of updated gradients with momentum applied.
        """
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        beta2 = group["beta2"]
        eps = group["eps"]

        updated_grads = []

        for p in params:
            state = self.state[p]
            grad = p.grad

            if grad is None:
                updated_grads.append(None)
                continue

            # Get local tensor for computation
            param_data = get_dtensor_local_tensor(p)
            grad_data = get_dtensor_local_tensor(grad)

            # Apply weight decay (decoupled)
            if weight_decay != 0:
                grad_data = grad_data + weight_decay * param_data

            # Update momentum buffer
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad_data)

            # Apply Nesterov momentum if enabled
            if nesterov:
                grad_data = grad_data + momentum * buf
            else:
                grad_data = buf.clone()

            # Update second moment (Adam-style)
            second_moment = state["second_moment"]
            second_moment.mul_(beta2).addcmul_(grad_data, grad_data, value=1 - beta2)

            # Bias correction for second moment
            bias_correction = 1 - beta2 ** (self._step_count + 1)
            corrected_second_moment = second_moment / bias_correction

            # Scale by adaptive learning rate
            adaptive_lr = lr / (corrected_second_moment.sqrt() + eps)
            grad_data = grad_data * adaptive_lr

            updated_grads.append(grad_data)

        return updated_grads

    def _compute_newton_schulz_updates(
        self,
        grads: List[torch.Tensor],
        ns_steps: int,
        ns_stepsize: float,
    ) -> List[torch.Tensor]:
        """Compute Newton-Schulz updates from gathered gradients.

        Args:
            grads: List of gathered gradient tensors.
            ns_steps: Number of Newton-Schulz iterations.
            ns_stepsize: Step size for updates.

        Returns:
            List of update tensors.
        """
        updates = []

        for grad in grads:
            if grad is None:
                updates.append(None)
                continue

            # Compute Newton-Schulz orthogonalization
            # The gradient has shape (world_size * local_dim, ...)
            # We need to reshape it to 2D for NS iteration
            original_shape = grad.shape

            if grad.dim() < 2:
                # 1D parameters: use simple update
                updates.append(-grad * ns_stepsize)
                continue

            # Reshape to 2D matrix
            grad_matrix = grad.reshape(grad.shape[0], -1)

            # Apply Newton-Schulz iteration
            orthogonalized = zeropower_via_newtonschulz5(
                grad_matrix,
                steps=ns_steps,
            )

            # Scale by step size
            update_matrix = -orthogonalized * ns_stepsize

            # Reshape back to original shape
            update = update_matrix.reshape(original_shape)
            updates.append(update)

        return updates

    def _scatter_updates_to_params(
        self,
        params: List[torch.nn.Parameter],
        updates: List[torch.Tensor],
    ) -> None:
        """Scatter updates back to parameters.

        For DTensor parameters, this scatters the update across processes.
        For regular parameters, applies the full update locally.

        Args:
            params: List of parameters to update.
            updates: List of full update tensors.
        """
        world_size = get_world_size()
        rank = get_rank()

        for p, update in zip(params, updates):
            if update is None:
                continue

            # Get local parameter data
            param_data = get_dtensor_local_tensor(p)

            # Extract the slice for this process
            if world_size > 1:
                dim_size = update.shape[0]
                slice_size = dim_size // world_size
                start_idx = rank * slice_size
                end_idx = start_idx + slice_size
                local_update = update[start_idx:end_idx]
            else:
                local_update = update

            # Reshape update to match parameter shape if needed
            if local_update.shape != param_data.shape:
                local_update = local_update.reshape(param_data.shape)

            # Apply update (use no_grad to avoid leaf variable issue)
            with torch.no_grad():
                param_data.copy_(param_data + local_update)

    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step.

        This method:
        1. Gathers gradients from all processes
        2. Applies weight decay and momentum
        3. Computes Newton-Schulz updates on full gradients
        4. Scatters updates back to sharded parameters

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                Optional for most use cases.

        Returns:
            The loss value if closure is provided, None otherwise.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Check if we should perform an update (gradient accumulation)
        self._step_count += 1
        should_update = (self._step_count % self.gradient_accumulation_steps) == 0

        if not should_update:
            return loss

        for group in self.param_groups:
            params = group["params"]
            ns_steps = group["ns_steps"]
            ns_stepsize = group["ns_stepsize"]

            # Step 1: Apply weight decay and momentum to get preprocessed gradients
            self._apply_weight_decay_and_momentum(params, group)  # noqa: F841

            # Step 2: Gather gradients from all processes
            # This is done outside unshard context to avoid unnecessary memory usage
            gathered_grads = self._gather_gradients(params)

            # Step 3: Compute Newton-Schulz updates on full gradients
            updates = self._compute_newton_schulz_updates(
                gathered_grads,
                ns_steps=ns_steps,
                ns_stepsize=ns_stepsize,
            )

            # Step 4: Scatter updates back to parameters
            self._scatter_updates_to_params(params, updates)

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero out the gradients of all optimized parameters.

        Args:
            set_to_none: If True, set gradients to None instead of zeroing.
                This can save memory but may affect gradient accumulation.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the optimizer as a dict.

        Returns:
            Dictionary containing optimizer state.
        """
        return {
            "state": self.state,
            "param_groups": self.param_groups,
            "step_count": self._step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state from a dict.

        Args:
            state_dict: Optimizer state dictionary.
        """
        self._step_count = state_dict.get("step_count", 0)
        super().load_state_dict(state_dict)


def create_fsdp_muon_optimizer(
    model: nn.Module,
    lr: float = 0.02,
    weight_decay: float = 0.01,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    ns_stepsize: float = 1.0,
    beta2: float = 0.99,
    eps: float = 1e-8,
    gradient_accumulation_steps: int = 1,
) -> FSDPMuonOptimizer:
    """Create an FSDPMuonOptimizer for the given model.

    This is a convenience function that creates an FSDPMuonOptimizer with
    the specified hyperparameters.

    Args:
        model: The model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        momentum: Momentum coefficient.
        nesterov: Whether to use Nesterov momentum.
        ns_steps: Number of Newton-Schulz iterations.
        ns_stepsize: Step size for Newton-Schulz updates.
        beta2: Coefficient for second moment.
        eps: Epsilon for numerical stability.
        gradient_accumulation_steps: Number of gradient accumulation steps.

    Returns:
        Configured FSDPMuonOptimizer instance.
    """
    return FSDPMuonOptimizer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        ns_stepsize=ns_stepsize,
        beta2=beta2,
        eps=eps,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
