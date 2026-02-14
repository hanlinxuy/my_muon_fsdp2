"""Spectral Sphere Optimizer (SSO) implementation.

This module provides the SpectralSphereOptimizer class, which extends the Muon
optimizer with spectral constraints. SSO enforces that weight matrices maintain
a specific spectral norm (radius), providing better control over feature learning
and improved training stability.

The key insight is to constrain weights to lie on a spectral sphere of fixed
radius R, where ||W||_2 = R. The optimization proceeds by:

1. Power iteration to compute spectral norm σ and top singular vectors (u, v)
2. Retraction to spectral sphere: W ← (R/σ) * W
3. Apply Newton-Schulz orthogonalization
4. Update: W ← W - lr * orthogonalized_update

References:
    - Controlled LLM Training on Spectral Sphere. arXiv:2601.08393 (2026).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.optim.optimizer import Optimizer

from .spectral import (
    apply_spectral_retraction,
    compute_spectral_norm,
    compute_target_radius,
    power_iteration,
)
from .utils import zeropower_via_newtonschulz5


class SpectralSphereOptimizer(Optimizer):
    """Implements the Spectral Sphere Optimizer (SSO).

    SSO extends the Muon optimizer by adding spectral constraints that enforce
    weight matrices to maintain a specific spectral norm. This provides better
    control over feature learning and improved training stability, especially
    for large language models.

    The optimizer applies:
    - Momentum-based gradient accumulation
    - Spectral norm constraint: ||W||_2 = R
    - Newton-Schulz orthogonalization for 2D matrices
    - Learning rate scaling based on matrix dimensions
    - Weight decay
    - Optional Nesterov momentum

    Algorithm:
        1. Compute spectral norm: σ = ||W||_2 via power iteration
        2. Retract to spectral sphere: W ← (R/σ) * W
        3. Update momentum buffer: buf = beta * buf + (1 - beta) * grad
        4. Compute update direction (with optional Nesterov)
        5. For 2D matrices: apply Newton-Schulz orthogonalization
        6. Scale learning rate: lr * max(1, m/n)**0.5
        7. Apply weight decay and parameter update

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate (default: 0.02).
        momentum: Momentum coefficient (default: 0.95).
        weight_decay: Weight decay (L2 penalty) (default: 0).
        nesterov: Enable Nesterov momentum (default: False).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
        power_iteration_steps: Number of power iteration steps for spectral norm
            computation (default: 10).
        radius_mode: Target radius computation mode. Options:
            - "spectral_mup": R = sqrt(n_out / n_in) (default)
            - "identity": R = 1.0
        radius_scaler: Scaling factor for target radius (default: 1.0).
        retract_mode: Retraction mode for spectral sphere. Options:
            - "hard": Direct scaling W ← (R/σ) * W (default)
            - "dynamic": Adaptive scaling based on distance from target

    Example:
        >>> from muon_fsdp import SpectralSphereOptimizer
        >>> model = torch.nn.Linear(512, 512)
        >>> optimizer = SpectralSphereOptimizer(
        ...     model.parameters(),
        ...     lr=0.02,
        ...     momentum=0.95,
        ...     radius_mode="spectral_mup",
        ... )
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Note:
        - Only 2D matrices (weight matrices) undergo spectral constraint and
          Newton-Schulz orthogonalization.
        - 1D vectors (biases) and scalars are updated without spectral constraints.
        - Learning rate scaling helps balance updates across different layer sizes.
        - Spectral constraints help control feature learning and improve stability.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        ns_steps: int = 5,
        power_iteration_steps: int = 10,
        radius_mode: str = "spectral_mup",
        radius_scaler: float = 1.0,
        retract_mode: str = "hard",
    ) -> None:
        """Initialize SpectralSphereOptimizer.

        Args:
            params: Parameters to optimize.
            lr: Learning rate.
            momentum: Momentum coefficient.
            weight_decay: Weight decay coefficient.
            nesterov: Whether to use Nesterov momentum.
            ns_steps: Number of Newton-Schulz iterations.
            power_iteration_steps: Number of power iteration steps.
            radius_mode: Target radius computation mode.
            radius_scaler: Scaling factor for target radius.
            retract_mode: Retraction mode for spectral sphere.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if ns_steps < 0:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
        if power_iteration_steps < 1:
            raise ValueError(f"Invalid power_iteration_steps value: {power_iteration_steps}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires positive momentum")
        if radius_mode not in ("spectral_mup", "identity"):
            raise ValueError(f"Invalid radius_mode: {radius_mode}")
        if retract_mode not in ("hard", "dynamic"):
            raise ValueError(f"Invalid retract_mode: {retract_mode}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "power_iteration_steps": power_iteration_steps,
            "radius_mode": radius_mode,
            "radius_scaler": radius_scaler,
            "retract_mode": retract_mode,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore optimizer state from pickle."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("power_iteration_steps", 10)
            group.setdefault("radius_mode", "spectral_mup")
            group.setdefault("radius_scaler", 1.0)
            group.setdefault("retract_mode", "hard")

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
            power_iteration_steps = group["power_iteration_steps"]
            radius_mode = group["radius_mode"]
            radius_scaler = group["radius_scaler"]
            retract_mode = group["retract_mode"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SpectralSphereOptimizer does not support sparse gradients")

                # Initialize state if needed
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    # Store initial spectral norm for logging
                    if p.dim() == 2:
                        state["initial_sigma"] = compute_spectral_norm(
                            p, num_iterations=power_iteration_steps
                        )

                buf = state["momentum_buffer"]

                # Apply spectral constraint and retraction for 2D matrices
                if p.dim() == 2:
                    # Compute current spectral norm
                    current_sigma = compute_spectral_norm(p, num_iterations=power_iteration_steps)

                    # Compute target radius
                    target_radius = compute_target_radius(
                        p.shape, radius_mode=radius_mode, radius_scaler=radius_scaler
                    )

                    # Apply retraction to spectral sphere
                    apply_spectral_retraction(
                        p,
                        current_sigma=current_sigma,
                        target_radius=target_radius,
                        mode=retract_mode,
                    )

                    # Store current spectral norm for monitoring
                    state["current_sigma"] = current_sigma
                    state["target_radius"] = target_radius

                # Update momentum buffer: buf = beta * buf + (1 - beta) * grad
                buf.lerp_(grad, 1 - momentum)

                # Compute update direction
                if nesterov:
                    update = grad + momentum * buf
                else:
                    update = buf.clone()

                # Apply Newton-Schulz orthogonalization for 2D matrices
                if p.dim() == 2:
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                    # Apply learning rate scaling: max(1, m/n)**0.5
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

    def get_spectral_norms(self) -> dict[int, dict[str, float]]:
        """Get spectral norm information for all 2D parameters.

        Returns:
            Dictionary mapping parameter IDs to their spectral norm info:
            {
                param_id: {
                    "current_sigma": float,
                    "target_radius": float,
                    "initial_sigma": float (if available),
                }
            }
        """
        spectral_norms = {}
        for group in self.param_groups:
            power_iteration_steps = group.get("power_iteration_steps", 10)
            for p in group["params"]:
                if p.dim() != 2:
                    continue
                param_id = id(p)
                state = self.state[p]
                info = {}
                if "current_sigma" in state:
                    info["current_sigma"] = state["current_sigma"]
                if "target_radius" in state:
                    info["target_radius"] = state["target_radius"]
                if "initial_sigma" in state:
                    info["initial_sigma"] = state["initial_sigma"]
                elif p.dim() == 2:
                    # Compute on-the-fly if not stored
                    info["current_sigma"] = compute_spectral_norm(
                        p, num_iterations=power_iteration_steps
                    )
                if info:
                    spectral_norms[param_id] = info
        return spectral_norms
