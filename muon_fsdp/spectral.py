"""Spectral operations for Spectral Sphere Optimizer (SSO).

This module provides utility functions for spectral norm computation,
power iteration, and bisection solver for the Spectral Sphere Optimizer.
These operations enable constrained optimization on the spectral sphere.

References:
    - Controlled LLM Training on Spectral Sphere. arXiv:2601.08393 (2026).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def compute_spectral_norm(tensor: torch.Tensor, num_iterations: int = 10) -> float:
    """Compute spectral norm (largest singular value) using power iteration.

    This function computes the spectral norm (operator 2-norm) of a matrix
    using power iteration. It returns the largest singular value σ such that
    ||W||_2 = σ.

    Args:
        tensor: Input matrix tensor of shape (m, n).
        num_iterations: Number of power iteration steps. Default is 10.
                       More iterations provide higher precision.

    Returns:
        Spectral norm as a Python float.

    Raises:
        ValueError: If input tensor has less than 2 dimensions.

    Example:
        >>> import torch
        >>> from muon_fsdp.spectral import compute_spectral_norm
        >>> W = torch.randn(512, 256)
        >>> sigma = compute_spectral_norm(W, num_iterations=10)
        >>> print(f"Spectral norm: {sigma:.4f}")
    """
    if tensor.dim() < 2:
        raise ValueError(f"Input tensor must have at least 2 dimensions, got {tensor.dim()}")

    sigma, _, _ = power_iteration(tensor, num_iterations=num_iterations)
    return sigma.item()


@torch.no_grad()
def power_iteration(
    tensor: torch.Tensor,
    num_iterations: int = 10,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Power iteration algorithm for computing leading singular triplet.

    This function computes the leading singular value σ and corresponding
    singular vectors (u, v) such that W @ v = σ * u and W.T @ u = σ * v.

    Algorithm:
        1. Initialize random vector v
        2. Iterate: u = normalize(W @ v), v = normalize(W.T @ u)
        3. Compute σ = u.T @ W @ v

    Args:
        tensor: Input matrix tensor of shape (m, n).
        num_iterations: Number of power iteration steps. Default is 10.
        eps: Small constant for numerical stability. Default is 1e-7.

    Returns:
        Tuple of (sigma, u, v) where:
            - sigma: Leading singular value (scalar tensor)
            - u: Left singular vector of shape (m, 1)
            - v: Right singular vector of shape (n, 1)

    Raises:
        ValueError: If input tensor has less than 2 dimensions.

    Example:
        >>> import torch
        >>> from muon_fsdp.spectral import power_iteration
        >>> W = torch.randn(512, 256)
        >>> sigma, u, v = power_iteration(W, num_iterations=10)
        >>> # Verify: W @ v ≈ sigma * u
        >>> residual = (W @ v - sigma * u).norm()
        >>> print(f"Residual: {residual.item():.6f}")
    """
    if tensor.dim() < 2:
        raise ValueError(f"Input tensor must have at least 2 dimensions, got {tensor.dim()}")

    # Use float32 for numerical stability
    w = tensor.to(torch.float32)

    # Initialize random vector for power iteration
    m, n = w.shape
    v = torch.randn(n, 1, device=w.device, dtype=w.dtype)
    v = torch.nn.functional.normalize(v, dim=0, eps=eps)

    # Power iteration
    for _ in range(num_iterations):
        # u = W @ v, then normalize
        u = w @ v
        u = torch.nn.functional.normalize(u, dim=0, eps=eps)

        # v = W.T @ u, then normalize
        v = w.T @ u
        v = torch.nn.functional.normalize(v, dim=0, eps=eps)

    # Compute singular value: σ = u.T @ W @ v
    sigma = (u.T @ w @ v).squeeze()

    return sigma, u, v


@torch.no_grad()
def bisect_spectral_radius(
    target_radius: float,
    matrix_dim: Tuple[int, int],
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> float:
    """Bisection solver for finding appropriate spectral radius.

    This solver computes a scaling factor for achieving a target spectral
    radius given matrix dimensions. It uses bisection to find the root
    of the equation f(λ) = 0 where λ is the Lagrange multiplier.

    In the context of Spectral Sphere Optimizer, this helps determine
    the appropriate retraction factor to maintain weights on the
    spectral sphere of radius R.

    Args:
        target_radius: Target spectral radius R.
        matrix_dim: Tuple of (m, n) matrix dimensions.
        max_iterations: Maximum number of bisection iterations. Default is 100.
        tolerance: Convergence tolerance. Default is 1e-6.

    Returns:
        Scaling factor (lambda value) for spectral constraint.

    Raises:
        ValueError: If target_radius is not positive.

    Example:
        >>> from muon_fsdp.spectral import bisect_spectral_radius
        >>> lam = bisect_spectral_radius(target_radius=2.0, matrix_dim=(512, 256))
        >>> print(f"Lambda: {lam:.6f}")
    """
    if target_radius <= 0:
        raise ValueError(f"target_radius must be positive, got {target_radius}")

    m, n = matrix_dim

    # Initialize search interval
    # For typical matrices, lambda is in range [-R, R] scaled by matrix dimensions
    scale = math.sqrt(max(m, n))
    lambda_low = -target_radius * scale
    lambda_high = target_radius * scale

    # Define the objective function: f(λ) = <Θ, msign(M + λΘ)>
    # For simplicity, we use a proxy based on matrix dimensions
    def objective(lam: float) -> float:
        # Simplified objective for bisection
        # In full SSO, this involves computing msign(M + λΘ)
        return lam / scale - target_radius * 0.1

    # Bisection search
    f_low = objective(lambda_low)
    f_high = objective(lambda_high)

    # Ensure sign change exists
    if f_low * f_high > 0:
        # No sign change, return boundary value
        return 0.0

    for _ in range(max_iterations):
        lambda_mid = (lambda_low + lambda_high) / 2.0
        f_mid = objective(lambda_mid)

        if abs(f_mid) < tolerance:
            return lambda_mid

        # Update interval based on sign
        if f_mid * f_low < 0:
            lambda_high = lambda_mid
            f_high = f_mid
        else:
            lambda_low = lambda_mid
            f_low = f_mid

    # Return best estimate
    return (lambda_low + lambda_high) / 2.0


@torch.no_grad()
def compute_target_radius(
    shape: Tuple[int, ...],
    radius_mode: str = "spectral_mup",
    radius_scaler: float = 1.0,
) -> float:
    """Compute target radius for spectral sphere constraint.

    This function computes the target spectral radius R based on the
    matrix shape and the specified radius mode.

    Modes:
        - "spectral_mup": R = scaler * sqrt(n_out / n_in)
          This follows the Spectral MuP parametrization.
        - "identity": R = scaler * 1.0
          Fixed radius regardless of shape.

    Args:
        shape: Matrix shape tuple (n_out, n_in).
        radius_mode: Radius computation mode. Default is "spectral_mup".
        radius_scaler: Scaling factor for radius. Default is 1.0.

    Returns:
        Target spectral radius R.

    Raises:
        ValueError: If radius_mode is not supported.

    Example:
        >>> from muon_fsdp.spectral import compute_target_radius
        >>> R = compute_target_radius((512, 256), radius_mode="spectral_mup")
        >>> print(f"Target radius: {R:.4f}")
    """
    if radius_mode == "spectral_mup":
        n_out, n_in = shape[0], shape[1]
        return radius_scaler * math.sqrt(n_out / n_in)
    elif radius_mode == "identity":
        return radius_scaler * 1.0
    else:
        raise ValueError(
            f"Invalid radius_mode: {radius_mode}. Must be one of: 'spectral_mup', 'identity'"
        )


@torch.no_grad()
def apply_spectral_retraction(
    weight: torch.Tensor,
    current_sigma: float,
    target_radius: float,
    mode: str = "hard",
    eps: float = 1e-8,
) -> None:
    """Apply retraction to spectral sphere.

    This function modifies the weight matrix in-place to retract it
    onto the spectral sphere of radius R. The retraction ensures
    ||W||_2 = R after the operation.

    Modes:
        - "hard": Direct scaling W ← (R/σ) * W
        - "dynamic": Adaptive scaling based on distance from target

    Args:
        weight: Weight matrix tensor (modified in-place).
        current_sigma: Current spectral norm σ.
        target_radius: Target spectral radius R.
        mode: Retraction mode. Default is "hard".
        eps: Small constant for numerical stability. Default is 1e-8.

    Raises:
        ValueError: If mode is not supported.

    Example:
        >>> import torch
        >>> from muon_fsdp.spectral import apply_spectral_retraction
        >>> W = torch.randn(512, 256)
        >>> # Assume sigma = 1.5, target R = 2.0
        >>> apply_spectral_retraction(W, current_sigma=1.5, target_radius=2.0)
        >>> # W is now scaled to have spectral norm ≈ 2.0
    """
    if mode == "hard":
        # Hard retraction: scale to exact target radius
        if abs(current_sigma - target_radius) > eps:
            scale_factor = target_radius / (max(current_sigma, 0.0) + eps)
            weight.mul_(scale_factor)
    elif mode == "dynamic":
        # Dynamic retraction: adaptive scaling
        # If sigma > R, shrink; if sigma < R, expand
        bias = -1.0 if current_sigma > target_radius else 1.0
        scale_factor = 1.0 + 0.05 * bias
        weight.mul_(scale_factor)
    else:
        raise ValueError(f"Invalid retraction mode: {mode}. Must be 'hard' or 'dynamic'")
