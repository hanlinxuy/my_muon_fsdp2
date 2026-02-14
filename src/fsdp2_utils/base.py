"""FSDP2 工具库 - 优化器抽象基类。"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from .dtensor import get_local_tensor, get_shard_dim, is_dtensor
from .comm import (
    all_gather_grads,
    all_reduce_sum,
    get_rank,
    get_world_size,
    is_distributed,
    scatter_updates,
)
from .fsdp_utils import (
    collect_fsdp_modules,
    get_local_grads,
    get_local_params,
    has_fsdp_modules,
    unshard_model,
)

logger = logging.getLogger(__name__)


class FSDPCompatibleOptimizer(Optimizer):
    """FSDP2 兼容的优化器基类。

    提供 FSDP2 环境下优化器所需的通用功能：
    - DTensor 参数处理
    - 梯度全收集
    - 更新分片
    - 梯度累积支持
    - 状态字典管理
    """

    def __init__(
        self,
        model: nn.Module,
        params: Optional[List[nn.Parameter]] = None,
        defaults: Optional[Dict[str, Any]] = None,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        """初始化 FSDP 兼容优化器。

        参数:
            model: 要优化的模型。
            params: 要优化的参数列表，如果为 None 则使用 model.parameters()。
            defaults: 优化器默认超参数字典。
            gradient_accumulation_steps: 梯度累积步数。
        """
        self.model = model
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._step_count = 0

        if params is None:
            params = list(model.parameters())

        params = [p for p in params if p.requires_grad]
        if not params:
            raise ValueError("优化器收到空参数列表")

        self.fsdp_modules = collect_fsdp_modules(model)
        if self.fsdp_modules:
            logger.info(f"检测到 {len(self.fsdp_modules)} 个 FSDP 模块")

        if defaults is None:
            defaults = {}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """执行单步优化。

        子类应重写 _step_impl 方法来实现具体的优化逻辑。

        参数:
            closure: 重新评估模型并返回损失的闭包。

        返回:
            如果提供了 closure 则返回损失值，否则返回 None。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        should_update = (self._step_count % self.gradient_accumulation_steps) == 0

        if should_update:
            self._step_impl()

        return loss

    def _step_impl(self) -> None:
        """执行实际的优化步骤。

        子类必须重写此方法来实现具体的优化逻辑。

        实现应遵循的步骤:
        1. 收集所有需要的梯度
        2. 对每个参数组应用优化逻辑
        3. 更新参数

        可使用的辅助方法:
        - self._gather_gradients(): 收集梯度
        - self._scatter_updates(): 分片更新
        - self._apply_updates(): 应用更新
        """
        raise NotImplementedError("子类必须实现 _step_impl 方法")

    def _gather_gradients(
        self,
        params: List[nn.Parameter],
    ) -> List[Optional[torch.Tensor]]:
        """收集所有进程的梯度。

        对于 DTensor 参数，会全收集梯度；对于普通参数，直接返回局部梯度。

        参数:
            params: 参数列表。

        返回:
            收集的梯度列表，None 表示该参数没有梯度。
        """
        local_grads: List[Optional[torch.Tensor]] = []

        for p in params:
            if p.grad is None:
                local_grads.append(None)
                continue

            grad = p.grad
            if is_dtensor(p):
                local_grad = get_local_tensor(grad)
            else:
                local_grad = grad

            local_grads.append(local_grad)

        if is_distributed() and get_world_size() > 1:
            non_none_grads = [g for g in local_grads if g is not None]
            if non_none_grads:
                gathered_non_none = all_gather_grads(non_none_grads)
                gathered_iter = iter(gathered_non_none)
                gathered_grads = [
                    next(gathered_iter) if g is not None else None for g in local_grads
                ]
            else:
                gathered_grads = local_grads
        else:
            gathered_grads = local_grads

        return gathered_grads

    def _scatter_updates(
        self,
        params: List[nn.Parameter],
        updates: List[Optional[torch.Tensor]],
    ) -> List[Optional[torch.Tensor]]:
        """将更新分片回各个进程。

        参数:
            params: 参数列表。
            updates: 完整更新列表。

        返回:
            分片后的更新列表。
        """
        world_size = get_world_size()
        rank = get_rank()

        if world_size <= 1:
            return updates

        non_none_updates = [u for u in updates if u is not None]
        if not non_none_updates:
            return updates

        scattered_non_none = scatter_updates(non_none_updates)
        scattered_iter = iter(scattered_non_none)

        return [next(scattered_iter) if u is not None else None for u in updates]

    def _apply_updates(
        self,
        params: List[nn.Parameter],
        updates: List[Optional[torch.Tensor]],
    ) -> None:
        """将更新应用到参数。

        参数:
            params: 参数列表。
            updates: 更新列表。
        """
        for p, update in zip(params, updates):
            if update is None:
                continue

            local_p = get_local_tensor(p)
            if local_p.shape != update.shape:
                update = update.reshape(local_p.shape)

            local_p.copy_(local_p + update)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """清零所有优化参数的梯度。"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        """返回优化器状态的字典。"""
        return {
            "state": self.state,
            "param_groups": self.param_groups,
            "step_count": self._step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """从字典加载优化器状态。"""
        self._step_count = state_dict.get("step_count", 0)
        super().load_state_dict(state_dict)


__all__ = ["FSDPCompatibleOptimizer"]
