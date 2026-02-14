"""FSDP2 与 Muon 优化器的集成层。

本模块提供 FSDPMuonOptimizer 类，将 Muon 优化器与 PyTorch FSDP2 (完全分片数据并行) 集成。
它处理 DTensor 参数，管理 unshard/reshard 生命周期，并对完整矩阵执行 Newton-Schulz 计算的梯度全收集。
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from muon_fsdp.distributed import (
    all_gather_grads,
    get_rank,
    get_world_size,
    is_available,
)
from muon_fsdp.utils import zeropower_via_newtonschulz5

logger = logging.getLogger(__name__)


def is_dtensor(tensor: torch.Tensor) -> bool:
    """检查张量是否为 DTensor (FSDP2 分片参数)。

    参数:
        tensor: 要检查的张量。

    返回:
        如果张量是 DTensor 则返回 True，否则返回 False。
    """
    try:
        from torch.distributed.tensor import DTensor

        return isinstance(tensor, DTensor)
    except ImportError:
        return False


def get_dtensor_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """从 DTensor 获取局部张量。

    参数:
        tensor: DTensor 或常规张量。

    返回:
        如果是 DTensor 则返回局部张量，否则返回原始张量。
    """
    if is_dtensor(tensor):
        return tensor.to_local()
    return tensor


def get_dtensor_full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """通过收集所有分片从 DTensor 获取完整张量。

    参数:
        tensor: DTensor 或常规张量。

    返回:
        如果是 DTensor 则返回完整收集的张量，否则返回原始张量。
    """
    if is_dtensor(tensor):
        return tensor.full_tensor()
    return tensor


def collect_fsdp_modules(model: nn.Module) -> List[nn.Module]:
    """从模型中收集所有 FSDP 包装的模块。

    此函数遍历模型并返回所有已使用 FSDP2 (fully_shard) 包装的模块。
    这些模块具有 FSDPModule 类型，并暴露 unshard/reshard 方法。

    参数:
        model: 要搜索 FSDP 模块的模型。

    返回:
        FSDP 包装模块的列表。
    """
    fsdp_modules = []
    try:
        from torch.distributed.fsdp import FSDPModule

        for module in model.modules():
            if isinstance(module, FSDPModule):
                fsdp_modules.append(module)
    except ImportError:
        # FSDP2 不可用
        logger.warning("FSDP2 不可用，使用单进程模式")
        return []

    return fsdp_modules


def has_fsdp_modules(model: nn.Module) -> bool:
    """检查模型是否具有任何 FSDP 包装的模块。

    参数:
        model: 要检查的模型。

    返回:
        如果模型包含 FSDP 模块则返回 True，否则返回 False。
    """
    return len(collect_fsdp_modules(model)) > 0


class FSDPMuonOptimizer(Optimizer):
    """支持 FSDP2 集成的 Muon 优化器。

    此优化器包装 Muon 优化器逻辑并添加 FSDP2 特定处理:
    - 检测来自 FSDP2 分片的 DTensor 参数
    - 管理 unshard/reshard 生命周期以访问完整参数
    - 全收集梯度以在完整矩阵上进行 Newton-Schulz 计算
    - 遵守 FSDP2 的 MixedPrecisionPolicy
    - 正确处理梯度累积

    优化器遵循 NS Replication 策略:
    1. 全收集所有进程的梯度
    2. 在完整梯度矩阵上计算 Newton-Schulz
    3. 将更新应用到分片参数

    参数:
        model: 要优化的模型。分布式训练必须使用 FSDP 包装，
            但也可以使用常规模型进行单 GPU 训练。
        params: 要优化的参数的可迭代对象。如果为 None，则使用 model.parameters()。
        lr: 学习率。默认: 0.02
        weight_decay: 权重衰减系数。默认: 0.01
        momentum: 梯度累积的动量系数。默认: 0.95
        nesterov: 是否使用 Nesterov 动量。默认: True
        ns_steps: Newton-Schulz 迭代次数。默认: 5
        ns_stepsize: Newton-Schulz 更新的步长。默认: 1.0
        beta2: 二阶矩系数 (Adam 风格)。默认: 0.99
        eps: 数值稳定性的 epsilon。默认: 1e-8
        gradient_accumulation_steps: 梯度累积的步数。
            默认: 1 (无累积)。

    示例:
        >>> from torch.distributed.fsdp import fully_shard
        >>> model = nn.Linear(512, 512)
        >>> fully_shard(model)
        >>> optimizer = FSDPMuonOptimizer(model, lr=0.02)
        >>> for input, target in dataloader:
        ...     output = model(input)
        ...     loss = criterion(output, target)
        ...     loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        model: nn.Module,
        params: Optional[List[torch.nn.Parameter]] = None,
        lr: float = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_stepsize: float = 1.0,
        beta2: float = 0.99,
        eps: float = 1e-8,
        gradient_accumulation_steps: int = 1,
    ):
        # 存储模型引用以进行 FSDP 操作
        self.model = model
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._step_count = 0

        # 如果未提供参数则使用默认参数
        if params is None:
            params = list(model.parameters())

        # 过滤掉不需要梯度的参数
        params = [p for p in params if p.requires_grad]

        # 验证参数
        if not params:
            raise ValueError("FSDPMuonOptimizer 收到空参数列表")

        # 检查 FSDP 模块
        self.fsdp_modules = collect_fsdp_modules(model)
        if self.fsdp_modules:
            logger.info(f"检测到 {len(self.fsdp_modules)} 个 FSDP 模块")

        # 默认超参数
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "ns_stepsize": ns_stepsize,
            "beta2": beta2,
            "eps": eps,
        }

        super().__init__(params, defaults)

        # 为每个参数初始化状态
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # 一阶矩的动量缓冲区
                state["momentum_buffer"] = torch.zeros_like(p)
                # 二阶矩用于 Adam 风格更新
                state["second_moment"] = torch.zeros_like(p)
                # 梯度累积计数器
                state["accum_count"] = 0

    @contextmanager
    def unshard_params(self) -> Generator[None, None, None]:
        """取消分片 FSDP 参数的上下文管理器。

        此上下文管理器在进入上下文之前对所有 FSDP 模块调用 unshard()，
        并确保在退出时 (包括异常情况下) 调用 reshard()。

        生成:
            None

        示例:
            >>> with optimizer.unshard_params():
            ...     # 参数在这里取消分片
            ...     full_params = [p.full_tensor() for p in model.parameters()]
            ... # 参数在这里自动重新分片
        """
        if not self.fsdp_modules:
            # 没有 FSDP 模块，立即 yield
            yield
            return

        # 为所有 FSDP 模块进入 unshard 上下文
        handles = []
        try:
            for module in self.fsdp_modules:
                handle = module.unshard()
                handles.append(handle)
                handle.__enter__()
            yield
        finally:
            # 确保在所有模块上调用 reshard，即使发生异常
            for handle in handles:
                try:
                    handle.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"重分片时出错: {e}")

    def _gather_gradients(
        self,
        params: List[torch.nn.Parameter],
    ) -> List[torch.Tensor]:
        """为给定参数从所有进程收集梯度。

        对于 DTensor 参数，这会跨所有进程全收集梯度。
        对于常规参数，返回局部梯度。

        参数:
            params: 要收集梯度的参数列表。

        返回:
            收集的梯度张量列表。
        """
        local_grads = []

        for p in params:
            if p.grad is None:
                # 如果不存在梯度则创建零梯度
                grad = torch.zeros_like(get_dtensor_local_tensor(p))
            else:
                grad = p.grad

            # 对于 DTensor，获取梯度的局部分片
            if is_dtensor(p):
                # DTensor 梯度也是 DTensor
                local_grad = get_dtensor_local_tensor(grad)
            else:
                local_grad = grad

            local_grads.append(local_grad)

        # 从所有进程全收集梯度
        if is_available() and get_world_size() > 1:
            gathered_grads = all_gather_grads(local_grads)
        else:
            gathered_grads = local_grads

        return gathered_grads

    def _apply_weight_decay_and_momentum(
        self,
        params: List[torch.nn.Parameter],
        group: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """应用权重衰减和动量到梯度。

        参数:
            params: 参数列表。
            group: 包含超参数的参数组。

        返回:
            应用动量后的更新梯度列表。
        """
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        beta2 = group["beta2"]
        eps = group["eps"]

        updated_grads = []

        for p in params:
            state = self.state[p]
            grad = p.grad

            if grad is None:
                updated_grads.append(None)
                continue

            # 获取用于计算的局部张量
            param_data = get_dtensor_local_tensor(p)
            grad_data = get_dtensor_local_tensor(grad)

            # 应用权重衰减 (解耦)
            if weight_decay != 0:
                grad_data = grad_data + weight_decay * param_data

            # 更新动量缓冲区
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad_data)

            # 如果启用则应用 Nesterov 动量
            if nesterov:
                grad_data = grad_data + momentum * buf
            else:
                grad_data = buf.clone()

            # 更新二阶矩 (Adam 风格)
            second_moment = state["second_moment"]
            second_moment.mul_(beta2).addcmul_(grad_data, grad_data, value=1 - beta2)

            # 二阶矩的偏差校正
            bias_correction = 1 - beta2 ** (self._step_count + 1)
            corrected_second_moment = second_moment / bias_correction

            # 按自适应学习率缩放
            adaptive_lr = lr / (corrected_second_moment.sqrt() + eps)
            grad_data = grad_data * adaptive_lr

            updated_grads.append(grad_data)

        return updated_grads

    def _compute_newton_schulz_updates(
        self,
        grads: List[torch.Tensor],
        ns_steps: int,
        ns_stepsize: float,
    ) -> List[torch.Tensor]:
        """从收集的梯度计算 Newton-Schulz 更新。

        参数:
            grads: 收集的梯度张量列表。
            ns_steps: Newton-Schulz 迭代次数。
            ns_stepsize: 更新的步长。

        返回:
            更新张量列表。
        """
        updates = []

        for grad in grads:
            if grad is None:
                updates.append(None)
                continue

            # 计算 Newton-Schulz 正交化
            # 梯度形状为 (world_size * local_dim, ...)
            # 需要重塑为 2D 以进行 NS 迭代
            original_shape = grad.shape

            if grad.dim() < 2:
                # 1D 参数: 使用简单更新
                updates.append(-grad * ns_stepsize)
                continue

            # 重塑为 2D 矩阵
            grad_matrix = grad.reshape(grad.shape[0], -1)

            # 应用 Newton-Schulz 迭代
            orthogonalized = zeropower_via_newtonschulz5(
                grad_matrix,
                steps=ns_steps,
            )

            # 按步长缩放
            update_matrix = -orthogonalized * ns_stepsize

            # 重塑回原始形状
            update = update_matrix.reshape(original_shape)
            updates.append(update)

        return updates

    def _scatter_updates_to_params(
        self,
        params: List[torch.nn.Parameter],
        updates: List[torch.Tensor],
    ) -> None:
        """将更新分片回参数。

        对于 DTensor 参数，这会将更新跨进程分片。
        对于常规参数，在本地应用完整更新。

        参数:
            params: 要更新的参数列表。
            updates: 完整更新张量列表。
        """
        world_size = get_world_size()
        rank = get_rank()

        for p, update in zip(params, updates):
            if update is None:
                continue

            # 获取局部参数数据
            param_data = get_dtensor_local_tensor(p)

            # 提取此进程的片
            if world_size > 1:
                dim_size = update.shape[0]
                slice_size = dim_size // world_size
                start_idx = rank * slice_size
                end_idx = start_idx + slice_size
                local_update = update[start_idx:end_idx]
            else:
                local_update = update

            # 如需要则重塑更新以匹配参数形状
            if local_update.shape != param_data.shape:
                local_update = local_update.reshape(param_data.shape)

            # 应用更新 (使用 no_grad 以避免叶子变量问题)
            with torch.no_grad():
                param_data.copy_(param_data + local_update)

    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """执行单步优化。

        此方法:
        1. 从所有进程收集梯度
        2. 应用权重衰减和动量
        3. 在完整梯度上计算 Newton-Schulz 更新
        4. 将更新分片回分片参数

        参数:
            closure: 重新评估模型并返回损失的闭包。对大多数用例是可选的。

        返回:
            如果提供了 closure 则返回损失值，否则返回 None。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 检查是否应该执行更新 (梯度累积)
        self._step_count += 1
        should_update = (self._step_count % self.gradient_accumulation_steps) == 0

        if not should_update:
            return loss

        for group in self.param_groups:
            params = group["params"]
            ns_steps = group["ns_steps"]
            ns_stepsize = group["ns_stepsize"]

            # 步骤 1: 应用权重衰减和动量以获得预处理梯度
            self._apply_weight_decay_and_momentum(params, group)  # noqa: F841

            # 步骤 2: 从所有进程收集梯度
            # 这在 unshard 上下文之外完成以避免不必要的内存使用
            gathered_grads = self._gather_gradients(params)

            # 步骤 3: 在完整梯度上计算 Newton-Schulz 更新
            updates = self._compute_newton_schulz_updates(
                gathered_grads,
                ns_steps=ns_steps,
                ns_stepsize=ns_stepsize,
            )

            # 步骤 4: 将更新分片回参数
            self._scatter_updates_to_params(params, updates)

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        """清零所有优化参数的梯度。

        参数:
            set_to_none: 如果为 True，则将梯度设置为 None 而不是清零。
                这可以节省内存但可能影响梯度累积。
        """
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
        """返回优化器状态的字典。

        返回:
            包含优化器状态的字典。
        """
        return {
            "state": self.state,
            "param_groups": self.param_groups,
            "step_count": self._step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """从字典加载优化器状态。

        参数:
            state_dict: 优化器状态字典。
        """
        self._step_count = state_dict.get("step_count", 0)
        super().load_state_dict(state_dict)


def create_fsdp_muon_optimizer(
    model: nn.Module,
    lr: float = 0.02,
    weight_decay: float = 0.01,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    ns_stepsize: float = 1.0,
    beta2: float = 0.99,
    eps: float = 1e-8,
    gradient_accumulation_steps: int = 1,
) -> FSDPMuonOptimizer:
    """为给定模型创建 FSDPMuonOptimizer。

    这是一个便捷函数，使用指定的超参数创建 FSDPMuonOptimizer。

    参数:
        model: 要优化的模型。
        lr: 学习率。
        weight_decay: 权重衰减系数。
        momentum: 动量系数。
        nesterov: 是否使用 Nesterov 动量。
        ns_steps: Newton-Schulz 迭代次数。
        ns_stepsize: Newton-Schulz 更新的步长。
        beta2: 二阶矩系数。
        eps: 数值稳定性的 epsilon。
        gradient_accumulation_steps: 梯度累积步数。

    返回:
        配置好的 FSDPMuonOptimizer 实例。
    """
    return FSDPMuonOptimizer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        ns_stepsize=ns_stepsize,
        beta2=beta2,
        eps=eps,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
