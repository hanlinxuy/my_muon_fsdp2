"""Muon 优化器实现。

本模块提供 MuonOptimizer 类，实现了带有动量、权重衰减和针对 2D
权重矩阵的 Newton-Schulz 正交化的 Muon 优化算法。
"""

from typing import Any, Optional

import torch
from torch.optim.optimizer import Optimizer

from .utils import zeropower_via_newtonschulz5


class MuonOptimizer(Optimizer):
    """实现 Muon 优化器。

    Muon 是一种通过 Newton-Schulz 迭代保持权重矩阵正交的优化算法。
    对于使用 2D 权重矩阵的深度神经网络训练特别有效。

    优化器应用:
    - 基于动量的梯度累积
    - 2D 矩阵的 Newton-Schulz 正交化
    - 基于矩阵维度的学习率缩放
    - 权重衰减
    - 可选的 Nesterov 动量

    算法:
        1. 更新动量缓冲区: buf = beta * buf + (1 - beta) * grad
        2. 计算更新方向 (可选 Nesterov)
        3. 对于 2D 矩阵: 应用 Newton-Schulz 正交化
        4. 缩放学习率: lr * max(1, m/n)**0.5 其中 m = min(dim_in, dim_out)
        5. 应用权重衰减和参数更新

    参数:
        params: 要优化的参数的可迭代对象或定义参数组的字典。
        lr: 学习率 (默认: 0.02)。
        momentum: 动量系数 (默认: 0.95)。
        weight_decay: 权重衰减 (L2 惩罚) (默认: 0)。
        nesterov: 启用 Nesterov 动量 (默认: False)。
        ns_steps: Newton-Schulz 迭代次数 (默认: 5)。

    示例:
        >>> from muon_fsdp import MuonOptimizer
        >>> model = torch.nn.Linear(512, 512)
        >>> optimizer = MuonOptimizer(model.parameters(), lr=0.02, momentum=0.95)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    注意:
        - 只有 2D 矩阵 (权重矩阵) 会进行 Newton-Schulz 正交化。
        - 1D 向量 (偏置) 和标量不进行正交化。
        - 学习率缩放有助于平衡不同层大小的更新。
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
        """初始化 MuonOptimizer。

        参数:
            params: 要优化的参数。
            lr: 学习率。
            momentum: 动量系数。
            weight_decay: 权重衰减系数。
            nesterov: 是否使用 Nesterov 动量。
            ns_steps: Newton-Schulz 迭代次数。
        """
        if lr < 0.0:
            raise ValueError(f"无效的学习率: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"无效的动量值: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"无效的 weight_decay 值: {weight_decay}")
        if ns_steps < 0:
            raise ValueError(f"无效的 ns_steps 值: {ns_steps}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov 动量需要正动量")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """从 pickle 恢复优化器状态。"""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

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

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MuonOptimizer 不支持稀疏梯度")

                # 需要时初始化状态
                state = self.state[p]
                if len(state) == 0:
                    # 初始化动量缓冲区
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]

                # 更新动量缓冲区: buf = beta * buf + (1 - beta) * grad
                # 使用 lerp 提高效率: buf.lerp_(grad, 1 - beta)
                buf.lerp_(grad, 1 - momentum)

                # 计算更新方向
                if nesterov:
                    # Nesterov: update = grad + beta * buf
                    update = grad + momentum * buf
                else:
                    # 标准动量: update = buf
                    update = buf.clone()

                # 对于 2D 矩阵应用 Newton-Schulz 正交化
                if p.dim() == 2:
                    # 计算正交化更新
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                    # 应用学习率缩放: max(1, m/n)**0.5
                    # 其中 m = min(dim_in, dim_out), n = max(dim_in, dim_out)
                    m, n = p.shape
                    min_dim = min(m, n)
                    max_dim = max(m, n)
                    lr_scale = max(1.0, min_dim / max_dim) ** 0.5
                    update.mul_(lr_scale)

                # 应用权重衰减: p = p * (1 - lr * weight_decay)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # 应用更新: p = p - lr * update
                p.add_(update, alpha=-lr)

        return loss

    def state_dict(self) -> dict[str, Any]:
        """返回优化器状态的字典。

        返回:
            包含以下内容的字典:
            - state: 参数 ID 到其状态 (动量缓冲区) 的映射
            - param_groups: 包含超参数的参数组列表
        """
        return super().state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """从状态字典加载优化器状态。

        参数:
            state_dict: 从 state_dict() 返回的优化器状态。
        """
        super().load_state_dict(state_dict)
