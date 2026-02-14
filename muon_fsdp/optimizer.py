"""Muon optimizer implementation.

This module provides the MuonOptimizer class, which implements the Muon optimization
algorithm with momentum, weight decay, and Newton-Schulz orthogonalization for 2D
weight matrices.
"""

from typing import Any, Optional

import torch
from torch.optim.optimizer import Optimizer

from .utils import zeropower_via_newtonschulz5


class MuonOptimizer(Optimizer):
    """Implements the Muon optimizer.

    Muon is an optimization algorithm that maintains orthogonal weight matrices
    through Newton-Schulz iteration. It is particularly effective for training
    deep neural networks with 2D weight matrices.

    The optimizer applies:
    - Momentum-based gradient accumulation
    - Newton-Schulz orthogonalization for 2D matrices
    - Learning rate scaling based on matrix dimensions
    - Weight decay
    - Optional Nesterov momentum

    Algorithm:
        1. Update momentum buffer: buf = beta * buf + (1 - beta) * grad
        2. Compute update direction (with optional Nesterov)
        3. For 2D matrices: apply Newton-Schulz orthogonalization
        4. Scale learning rate: lr * max(1, m/n)**0.5 where m = min(dim_in, dim_out)
        5. Apply weight decay and parameter update

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate (default: 0.02).
        momentum: Momentum coefficient (default: 0.95).
        weight_decay: Weight decay (L2 penalty) (default: 0).
        nesterov: Enable Nesterov momentum (default: False).
        ns_steps: Number of Newton-Schulz iterations (default: 5).

    Example:
        >>> from muon_fsdp import MuonOptimizer
        >>> model = torch.nn.Linear(512, 512)
        >>> optimizer = MuonOptimizer(model.parameters(), lr=0.02, momentum=0.95)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Note:
        - Only 2D matrices (weight matrices) undergo Newton-Schulz orthogonalization.
        - 1D vectors (biases) and scalars are updated without orthogonalization.
        - Learning rate scaling helps balance updates across different layer sizes.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        ns_steps: int = 5,
    ) -> None:
        """Initialize MuonOptimizer.

        Args:
            params: Parameters to optimize.
            lr: Learning rate.
            momentum: Momentum coefficient.
            weight_decay: Weight decay coefficient.
            nesterov: Whether to use Nesterov momentum.
            ns_steps: Number of Newton-Schulz iterations.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if ns_steps < 0:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires positive momentum")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore optimizer state from pickle."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if closure is provided, None otherwise.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MuonOptimizer does not support sparse gradients")

                # Initialize state if needed
                state = self.state[p]
                if len(state) == 0:
                    # Initialize momentum buffer
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]

                # Update momentum buffer: buf = beta * buf + (1 - beta) * grad
                # Using lerp for efficiency: buf.lerp_(grad, 1 - beta)
                buf.lerp_(grad, 1 - momentum)

                # Compute update direction
                if nesterov:
                    # Nesterov: update = grad + beta * buf
                    update = grad + momentum * buf
                else:
                    # Standard momentum: update = buf
                    update = buf.clone()

                # Apply Newton-Schulz orthogonalization for 2D matrices
                if p.dim() == 2:
                    # Compute orthogonalized update
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                    # Apply learning rate scaling: max(1, m/n)**0.5
                    # where m = min(dim_in, dim_out), n = max(dim_in, dim_out)
                    m, n = p.shape
                    min_dim = min(m, n)
                    max_dim = max(m, n)
                    lr_scale = max(1.0, min_dim / max_dim) ** 0.5
                    update.mul_(lr_scale)

                # Apply weight decay: p = p * (1 - lr * weight_decay)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Apply update: p = p - lr * update
                p.add_(update, alpha=-lr)

        return loss

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the optimizer as a dict.

        Returns:
            Dict containing:
            - state: Dict mapping parameter IDs to their state (momentum buffers)
            - param_groups: List of parameter groups with their hyperparameters
        """
        return super().state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state from a state dict.

        Args:
            state_dict: Optimizer state returned from state_dict().
        """
        super().load_state_dict(state_dict)
