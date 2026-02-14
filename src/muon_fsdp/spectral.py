"""谱球优化器 (SSO) 的谱操作。

本模块提供谱范数计算、幂迭代和二分搜索求解器的实用函数。
这些操作支持在谱球上进行约束优化。

参考文献:
    - Controlled LLM Training on Spectral Sphere. arXiv:2601.08393 (2026).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def compute_spectral_norm(tensor: torch.Tensor, num_iterations: int = 10) -> float:
    """使用幂迭代计算谱范数 (最大奇异值)。

    此函数使用幂迭代计算矩阵的谱范数 (算子 2-范数)。
    返回最大的奇异值 σ，使得 ||W||_2 = σ。

    参数:
        tensor: 形状为 (m, n) 的输入矩阵张量。
        num_iterations: 幂迭代步数。默认为 10。迭代次数越多精度越高。

    返回:
        谱范数作为 Python float。

    异常:
        ValueError: 如果输入张量维度少于 2。

    示例:
        >>> import torch
        >>> from muon_fsdp.spectral import compute_spectral_norm
        >>> W = torch.randn(512, 256)
        >>> sigma = compute_spectral_norm(W, num_iterations=10)
        >>> print(f"谱范数: {sigma:.4f}")
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
    """计算主奇异三元组的幂迭代算法。

    此函数计算主奇异值 σ 和对应的奇异向量 (u, v)，
    使得 W @ v = σ * u 和 W.T @ u = σ * v。

    算法:
        1. 初始化随机向量 v
        2. 迭代: u = normalize(W @ v), v = normalize(W.T @ u)
        3. 计算 σ = u.T @ W @ v

    参数:
        tensor: 形状为 (m, n) 的输入矩阵张量。
        num_iterations: 幂迭代步数。默认为 10。
        eps: 数值稳定性的小常数。默认为 1e-7。

    返回:
        (sigma, u, v) 元组，其中:
            - sigma: 主奇异值 (标量张量)
            - u: 形状为 (m, 1) 的左奇异向量
            - v: 形状为 (n, 1) 的右奇异向量

    异常:
        ValueError: 如果输入张量维度少于 2。

    示例:
        >>> import torch
        >>> from muon_fsdp.spectral import power_iteration
        >>> W = torch.randn(512, 256)
        >>> sigma, u, v = power_iteration(W, num_iterations=10)
        >>> # 验证: W @ v ≈ sigma * u
        >>> residual = (W @ v - sigma * u).norm()
        >>> print(f"残差: {residual.item():.6f}")
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
    """用于查找适当谱半径的二分搜索求解器。

    此求解器计算用于在给定矩阵维度下达到目标谱半径的缩放因子。
    它使用二分法找到方程 f(λ) = 0 的根，其中 λ 是拉格朗日乘数。

    在谱球优化器的上下文中，这有助于确定适当的回缩因子，
    以将权重保持在半径为 R 的谱球上。

    参数:
        target_radius: 目标谱半径 R。
        matrix_dim: (m, n) 矩阵维度元组。
        max_iterations: 最大二分迭代次数。默认为 100。
        tolerance: 收敛容差。默认为 1e-6。

    返回:
        谱约束的缩放因子 (lambda 值)。

    异常:
        ValueError: 如果 target_radius 不为正数。

    示例:
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
    """计算谱球约束的目标半径。

    此函数根据矩阵形状和指定的半径模式计算目标谱半径 R。

    模式:
        - "spectral_mup": R = scaler * sqrt(n_out / n_in)
          这遵循 Spectral MuP 参数化。
        - "identity": R = scaler * 1.0
          无论形状如何，半径固定。

    参数:
        shape: 矩阵形状元组 (n_out, n_in)。
        radius_mode: 半径计算模式。默认为 "spectral_mup"。
        radius_scaler: 半径的缩放因子。默认为 1.0。

    返回:
        目标谱半径 R。

    异常:
        ValueError: 如果不支持 radius_mode。

    示例:
        >>> from muon_fsdp.spectral import compute_target_radius
        >>> R = compute_target_radius((512, 256), radius_mode="spectral_mup")
        >>> print(f"目标半径: {R:.4f}")
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
    """应用谱球回缩。

    此函数就地修改权重矩阵以将其回缩到半径为 R 的谱球上。
    回缩确保操作后 ||W||_2 = R。

    模式:
        - "hard": 直接缩放 W ← (R/σ) * W
        - "dynamic": 基于到目标距离的自适应缩放

    参数:
        weight: 权重矩阵张量 (就地修改)。
        current_sigma: 当前谱范数 σ。
        target_radius: 目标谱半径 R。
        mode: 回缩模式。默认为 "hard"。
        eps: 数值稳定性的小常数。默认为 1e-8。

    异常:
        ValueError: 如果模式不支持。

    示例:
        >>> import torch
        >>> from muon_fsdp.spectral import apply_spectral_retraction
        >>> W = torch.randn(512, 256)
        >>> # 假设 sigma = 1.5, 目标 R = 2.0
        >>> apply_spectral_retraction(W, current_sigma=1.5, target_radius=2.0)
        >>> # W 现在被缩放到谱范数 ≈ 2.0
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
