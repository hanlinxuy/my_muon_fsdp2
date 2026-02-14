"""FSDP2 工具库 - Muon 优化器实现。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base import FSDPCompatibleOptimizer
from .numerical import zeropower_via_newtonschulz5


class MuonOptimizer(FSDPCompatibleOptimizer):
    """Muon 优化器实现。

    Muon 是一种通过 Newton-Schulz 迭代保持权重矩阵正交的优化算法。
    对于使用 2D 权重矩阵的深度神经网络训练特别有效。
    """

    def __init__(
        self,
        model: nn.Module,
        params: Optional[List[nn.Parameter]] = None,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        ns_steps: int = 5,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        """初始化 MuonOptimizer。

        参数:
            model: 要优化的模型。
            params: 要优化的参数列表，None 表示使用 model.parameters()。
            lr: 学习率，默认 0.02。
            momentum: 动量系数，默认 0.95。
            weight_decay: 权重衰减系数，默认 0.0。
            nesterov: 是否使用 Nesterov 动量，默认 False。
            ns_steps: Newton-Schulz 迭代次数，默认 5。
            gradient_accumulation_steps: 梯度累积步数，默认 1。
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

        super().__init__(
            model=model,
            params=params,
            defaults=defaults,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["momentum_buffer"] = torch.zeros_like(p)

    def _step_impl(self) -> None:
        """执行 Muon 优化步骤。"""
        from .dtensor import is_dtensor, get_local_tensor

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            params = group["params"]

            gathered_grads = self._gather_gradients(params)
            updates: List[Optional[torch.Tensor]] = []

            for p, grad in zip(params, gathered_grads):
                if grad is None:
                    updates.append(None)
                    continue

                state = self.state[p]

                grad_tensor = grad
                if is_dtensor(grad_tensor):
                    grad_tensor = get_local_tensor(grad_tensor)

                grad_matrix = grad_tensor.reshape(grad_tensor.shape[0], -1).contiguous()
                buf_shape = grad_matrix.shape

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros(
                        buf_shape, device=grad_matrix.device, dtype=grad_matrix.dtype
                    )

                buf = state["momentum_buffer"]
                if is_dtensor(buf):
                    buf = get_local_tensor(buf)

                if buf.shape != buf_shape:
                    buf = torch.zeros(buf_shape, device=grad_matrix.device, dtype=grad_matrix.dtype)

                if weight_decay != 0:
                    full_p = p
                    if is_dtensor(p):
                        full_p = get_local_tensor(p)
                    full_p = full_p.reshape(full_p.shape[0], -1).contiguous()
                    grad_matrix = grad_matrix + weight_decay * full_p

                buf.lerp_(grad_matrix, 1 - momentum)

                if nesterov:
                    update = grad_matrix + momentum * buf
                else:
                    update = buf.clone()

                if p.dim() == 2:
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    m, n = p.shape
                    min_dim = min(m, n)
                    max_dim = max(m, n)
                    lr_scale = max(1.0, min_dim / max_dim) ** 0.5
                    update.mul_(lr_scale)

                update.mul_(-lr)
                update = update.reshape(p.shape)

                updates.append(update)

            scattered_updates = self._scatter_updates(params, updates)
            self._apply_updates(params, scattered_updates)


__all__ = ["MuonOptimizer"]
