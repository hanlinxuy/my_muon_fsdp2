"""谱球优化器 (SSO) 实现。

本模块提供 SpectralSphereOptimizer 类,扩展了带有谱约束的 Muon 优化器。
SSO 强制权重矩阵保持特定的谱范数 (半径),从而更好地控制特征学习
并提高训练稳定性。

关键见解是将权重限制在固定半径 R 的谱球上,其中 ||W||_2 = R。优化过程如下:

1. 幂迭代计算谱范数 σ 和顶级奇异向量 (u, v)
2. 缩回到谱球: W ← (R/σ) * W
3. 应用 Newton-Schulz 正交化
4. 更新: W ← W - lr * 正交化更新

参考文献:
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
    """实现谱球优化器 (SSO)。

    SSO 通过添加谱约束扩展了 Muon 优化器,强制权重矩阵保持特定的谱范数。
    这提供了更好的特征学习控制和改进的训练稳定性,特别适用于大型语言模型。

    优化器应用:
    - 基于动量的梯度累积
    - 谱范数约束: ||W||_2 = R
    - 2D 矩阵的 Newton-Schulz 正交化
    - 基于矩阵维度的学习率缩放
    - 权重衰减
    - 可选的 Nesterov 动量

    算法:
        1. 计算谱范数: σ = ||W||_2 通过幂迭代
        2. 缩回到谱球: W ← (R/σ) * W
        3. 更新动量缓冲区: buf = beta * buf + (1 - beta) * grad
        4. 计算更新方向 (可选 Nesterov)
        5. 对于 2D 矩阵: 应用 Newton-Schulz 正交化
        6. 缩放学习率: lr * max(1, m/n)**0.5
        7. 应用权重衰减和参数更新

    参数:
        params: 要优化的参数的可迭代对象或定义参数组的字典。
        lr: 学习率 (默认: 0.02)。
        momentum: 动量系数 (默认: 0.95)。
        weight_decay: 权重衰减 (L2 惩罚) (默认: 0)。
        nesterov: 启用 Nesterov 动量 (默认: False)。
        ns_steps: Newton-Schulz 迭代次数 (默认: 5)。
        power_iteration_steps: 谱范数计算的幂迭代步数 (默认: 10)。
        radius_mode: 目标半径计算模式。选项:
            - "spectral_mup": R = sqrt(n_out / n_in) (默认)
            - "identity": R = 1.0
        radius_scaler: 目标半径的缩放因子 (默认: 1.0)。
        retract_mode: 谱球的回缩模式。选项:
            - "hard": 直接缩放 W ← (R/σ) * W (默认)
            - "dynamic": 基于到目标距离的自适应缩放

    示例:
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

    注意:
        - 只有 2D 矩阵 (权重矩阵) 会进行谱约束和 Newton-Schulz 正交化。
        - 1D 向量 (偏置) 和标量不进行谱约束更新。
        - 学习率缩放有助于平衡不同层大小的更新。
        - 谱约束有助于控制特征学习并提高稳定性。
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
        """初始化 SpectralSphereOptimizer。

        参数:
            params: 要优化的参数。
            lr: 学习率。
            momentum: 动量系数。
            weight_decay: 权重衰减系数。
            nesterov: 是否使用 Nesterov 动量。
            ns_steps: Newton-Schulz 迭代次数。
            power_iteration_steps: 幂迭代步数。
            radius_mode: 目标半径计算模式。
            radius_scaler: 目标半径的缩放因子。
            retract_mode: 谱球的回缩模式。
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
        """从 pickle 恢复优化器状态。"""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("power_iteration_steps", 10)
            group.setdefault("radius_mode", "spectral_mup")
            group.setdefault("radius_scaler", 1.0)
            group.setdefault("retract_mode", "hard")

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """执行单步优化。

        参数:
            closure: 重新评估模型并返回损失的闭包。

        返回:
            如果提供了 closure 则返回损失值，否则返回 None。
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
        """获取所有 2D 参数的谱范数信息。

        返回:
            将参数 ID 映射到其谱范数信息的字典:
            {
                param_id: {
                    "current_sigma": float,
                    "target_radius": float,
                    "initial_sigma": float (如果可用),
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
