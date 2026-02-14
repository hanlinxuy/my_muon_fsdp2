"""Core numerical utilities for Muon FSDP2.

This module provides essential numerical operations used by the Muon optimizer,
including the Newton-Schulz iteration for computing orthogonal matrices.
"""

from typing import Optional

import torch


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Compute zeropower (orthogonal) matrix using Newton-Schulz iteration.

    This function implements the Newton-Schulz iteration to compute an orthogonal
    matrix that approximates the input matrix's "zeropower" (also known as the
    "orthogonalization" or "project onto orthogonal matrices"). This is a key
    component of the Muon optimizer for maintaining orthogonal weight matrices.

    The implementation uses a quintic polynomial iteration with coefficients
    (a, b, c) = (3.4445, -4.7750, 2.0315) that are optimized for faster
    convergence in the Muon optimizer context.

    Algorithm:
        X_{k+1} = a * X_k + (b * A + c * A^2) * X_k
        where A = X_k @ X_k^T
        and coefficients (a, b, c) = (3.4445, -4.7750, 2.0315)

    Args:
        G: Input matrix tensor of shape (m, n) where m >= n.
           If m < n, the matrix is transposed internally.
        steps: Number of Newton-Schulz iterations. Default is 5.
               More iterations provide higher precision but slower computation.
        dtype: Computation dtype. If None, uses the dtype of input tensor.
               Set to torch.bfloat16 for bfloat16 computation.

    Returns:
        Orthogonal matrix of the same shape as input.
        The output matrix W satisfies W @ W.T being close to identity.
        Output dtype matches the input tensor's dtype.

    Raises:
        ValueError: If input tensor has less than 2 dimensions.
        ValueError: If steps is negative.

    Example:
        >>> import torch
        >>> from muon_fsdp import zeropower_via_newtonschulz5
        >>> G = torch.randn(512, 512)
        >>> W = zeropower_via_newtonschulz5(G)
        >>> # Verify orthogonality: W @ W.T should be close to identity
        >>> orthogonality_error = (W @ W.T - torch.eye(512)).norm().item()
        >>> print(f"Orthogonality error: {orthogonality_error:.6f}")

    Note:
        - For best numerical stability, ensure the smaller dimension is first.
        - The input matrix is normalized before iteration to prevent overflow.
        - A small epsilon (1e-7) is added to prevent division by zero.
        - The quintic iteration produces an orthogonal-ish matrix suitable for
          Muon optimizer but may not achieve very low orthogonality error.
    """
    # Validate inputs
    if G.dim() < 2:
        raise ValueError(f"Input tensor must have at least 2 dimensions, got {G.dim()}")
    if steps < 0:
        raise ValueError(f"Number of steps must be non-negative, got {steps}")

    # Handle empty or trivial cases
    if G.numel() == 0:
        return G

    transposed = G.size(-2) < G.size(-1)

    # Determine computation dtype
    if dtype is None:
        dtype = G.dtype
    compute_dtype = torch.float32 if dtype == torch.bfloat16 else dtype

    # Convert to computation dtype for numerical stability
    X = G.to(compute_dtype)

    # Transpose if needed to ensure smaller dimension is first
    if transposed:
        X = X.T

    # Normalize input to prevent overflow
    X = X / (X.norm() + 1e-7)

    # Newton-Schulz iteration coefficients (quintic polynomial)
    # These coefficients maximize convergence speed for Muon optimizer
    a, b, c = 3.4445, -4.7750, 2.0315

    # Perform iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    # Transpose back if we transposed
    if transposed:
        X = X.T

    # Convert back to original dtype
    return X.to(dtype)
