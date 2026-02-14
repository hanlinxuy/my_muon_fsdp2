"""FSDP2 工具库 - 数值计算工具。"""

from __future__ import annotations

from typing import Optional

import torch


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """使用 Newton-Schulz 迭代计算正交矩阵。

    参数:
        G: 输入矩阵 (m, n)。
        steps: Newton-Schulz 迭代次数，默认 5。
        dtype: 计算数据类型，None 表示使用输入类型。

    返回:
        正交矩阵，形状与输入相同。
    """
    if G.dim() < 2:
        raise ValueError(f"输入张量必须至少有 2 个维度，得到 {G.dim()}")
    if steps < 0:
        raise ValueError(f"步数必须为非负数，得到 {steps}")

    if G.numel() == 0:
        return G

    transposed = G.size(-2) < G.size(-1)

    if dtype is None:
        dtype = G.dtype
    compute_dtype = torch.float32 if dtype == torch.bfloat16 else dtype

    X = G.to(compute_dtype)

    if transposed:
        X = X.T

    X = X / (X.norm() + 1e-7)

    a, b, c = 3.4445, -4.7750, 2.0315

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(dtype)


def power_iteration(
    A: torch.Tensor,
    num_iterations: int = 100,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """幂迭代计算最大特征值和特征向量。

    参数:
        A: 对称矩阵 (n, n)。
        num_iterations: 最大迭代次数。
        tol: 收敛容差。

    返回:
        (最大特征值, 对应特征向量)。
    """
    n = A.size(0)
    v = torch.randn(n, device=A.device, dtype=A.dtype)
    v = v / v.norm()

    for _ in range(num_iterations):
        v_new = A @ v
        v_new = v_new / v_new.norm()

        if (v - v_new).norm() < tol:
            break

        v = v_new

    eigenvalue = (v @ A @ v).item()
    return eigenvalue, v


def compute_spectral_norm(
    A: torch.Tensor,
    num_iterations: int = 100,
) -> torch.Tensor:
    """计算矩阵的谱范数。"""
    if A.dim() == 2:
        m, n = A.shape
        if m >= n:
            B = A.T @ A
        else:
            B = A @ A.T

        eigenvalue, _ = power_iteration(B, num_iterations)
        return torch.sqrt(torch.tensor(eigenvalue, device=A.device, dtype=A.dtype))

    return torch.linalg.norm(A, 2)


__all__ = [
    "zeropower_via_newtonschulz5",
    "power_iteration",
    "compute_spectral_norm",
]
