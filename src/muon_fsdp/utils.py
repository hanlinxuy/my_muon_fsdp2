"""Muon FSDP2 核心数值工具。

本模块提供 Muon 优化器所需的基本数值操作，包括用于计算正交矩阵的
Newton-Schulz 迭代。
"""

from typing import Optional

import torch


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """使用 Newton-Schulz 迭代计算零功率 (正交) 矩阵。

    本函数实现 Newton-Schulz 迭代，用于计算输入矩阵的"零功率"
    (也称为"正交化"或"投影到正交矩阵")。这是 Muon 优化器
    保持权重矩阵正交的关键组件。

    实现使用五次多项式迭代，系数 (a, b, c) = (3.4445, -4.7750, 2.0315)，
    针对 Muon 优化器中更快的收敛进行了优化。

    算法:
        X_{k+1} = a * X_k + (b * A + c * A^2) * X_k
        其中 A = X_k @ X_k^T
        系数 (a, b, c) = (3.4445, -4.7750, 2.0315)

    参数:
        G: 形状为 (m, n) 的输入矩阵张量，其中 m >= n。
           如果 m < n，矩阵会在内部转置。
        steps: Newton-Schulz 迭代次数。默认为 5。
               迭代次数越多精度越高但计算越慢。
        dtype: 计算数据类型。如果为 None，则使用输入张量的数据类型。
               设置为 torch.bfloat16 用于 bfloat16 计算。

    返回:
        与输入形状相同的正交矩阵。
        输出矩阵 W 满足 W @ W.T 接近单位矩阵。
        输出数据类型与输入张量的数据类型相同。

    异常:
        ValueError: 如果输入张量维度少于 2。
        ValueError: 如果 steps 为负数。

    示例:
        >>> import torch
        >>> from muon_fsdp import zeropower_via_newtonschulz5
        >>> G = torch.randn(512, 512)
        >>> W = zeropower_via_newtonschulz5(G)
        >>> # 验证正交性: W @ W.T 应该接近单位矩阵
        >>> orthogonality_error = (W @ W.T - torch.eye(512)).norm().item()
        >>> print(f"正交性误差: {orthogonality_error:.6f}")

    注意:
        - 为获得最佳数值稳定性，请确保较小的维度在前。
        - 输入矩阵在迭代前进行归一化以防止溢出。
        - 添加小 epsilon (1e-7) 以防止除零。
        - 五次迭代产生适合 Muon 优化器的近似正交矩阵，
          可能无法达到非常低的正交性误差。
    """
    # 验证输入
    if G.dim() < 2:
        raise ValueError(f"输入张量必须至少有 2 个维度，得到 {G.dim()}")
    if steps < 0:
        raise ValueError(f"步数必须为非负数，得到 {steps}")

    # 处理空或平凡情况
    if G.numel() == 0:
        return G

    transposed = G.size(-2) < G.size(-1)

    # 确定计算数据类型
    if dtype is None:
        dtype = G.dtype
    compute_dtype = torch.float32 if dtype == torch.bfloat16 else dtype

    # 转换为计算数据类型以提高数值稳定性
    X = G.to(compute_dtype)

    # 如需要转置以确保较小维度在前
    if transposed:
        X = X.T

    # 归一化输入以防止溢出
    X = X / (X.norm() + 1e-7)

    # Newton-Schulz 迭代系数 (五次多项式)
    # 这些系数使 Muon 优化器的收敛速度最大化
    a, b, c = 3.4445, -4.7750, 2.0315

    # 执行迭代
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    # 如果转置过，则转置回来
    if transposed:
        X = X.T

    # 转换回原始数据类型
    return X.to(dtype)
