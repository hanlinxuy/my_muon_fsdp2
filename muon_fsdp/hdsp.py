"""HDSP (Hybrid Data Sharding Parallel) 优化器集成

本模块提供 HDSPMuonOptimizer 类,实现混合数据分片并行(Hybrid Data Sharding Parallel)。
HDSP 是 2D 并行策略：节点内使用 FSDP 分片，节点间使用数据并行复制。

主要特性：
- 支持 DeviceMesh 2D 配置 (dp_replicate, fsdp_shard)
- 组内梯度聚合（节点内 all-gather，节点间不通信）
- 与 FSDPMuonOptimizer 保持 API 兼容
- 支持混合精度训练
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from muon_fsdp.distributed import (
    get_rank,
    get_world_size,
    is_available,
)
from muon_fsdp.fsdp import (
    FSDPMuonOptimizer,
    get_dtensor_local_tensor,
    has_fsdp_modules,
)
from muon_fsdp.utils import zeropower_via_newtonschulz5

logger = logging.getLogger(__name__)


def is_dtensor(tensor: torch.Tensor) -> bool:
    """检查是否为 DTensor (FSDP2 分片参数)"""
    try:
        from torch.distributed.tensor import DTensor

        return isinstance(tensor, DTensor)
    except ImportError:
        return False


def create_device_mesh_2d(
    dp_size: int,
    fsdp_size: int,
) -> Optional[Any]:
    """创建 2D DeviceMesh 用于 HSDP 配置

    Args:
        dp_size: 数据并行维度大小（节点数）
        fsdp_size: FSDP 分片维度大小（每节点 GPU 数）

    Returns:
        DeviceMesh 对象，或 None（如果不可用）
    """
    if not is_available():
        return None

    try:
        import torch.distributed as dist
        from torch.distributed.device_mesh import DeviceMesh

        if not dist.is_initialized():
            logger.warning("分布式未初始化，无法创建 DeviceMesh")
            return None

        # 创建 2D 设备网格
        # 维度 0: 数据并行 (replicate)
        # 维度 1: FSDP 分片 (shard)
        mesh = DeviceMesh(
            device_type="cuda",
            mesh=[list(range(dp_size * fsdp_size))],
            mesh_dim_names=("dp_replicate", "fsdp_shard"),
        )
        return mesh
    except Exception as e:
        logger.warning(f"创建 DeviceMesh 失败: {e}")
        return None


def get_hsdp_groups(
    mesh: Any,
) -> Tuple[Optional[Any], Optional[Any]]:
    """从 DeviceMesh 获取 HSDP 进程组

    Args:
        mesh: 2D DeviceMesh 对象

    Returns:
        (dp_group, fsdp_group) 元组
    """
    try:
        dp_group = mesh.get_group("dp_replicate")
        fsdp_group = mesh.get_group("fsdp_shard")
        return dp_group, fsdp_group
    except Exception:
        return None, None


def gather_grads_group(
    grads: List[torch.Tensor],
    group: Any,
) -> List[torch.Tensor]:
    """在指定进程组内聚合梯度

    Args:
        grads: 本地梯度列表
        group: 进程组

    Returns:
        聚合后的梯度列表
    """
    if group is None or not is_available():
        return grads

    try:
        import torch.distributed as dist

        world_size = dist.get_world_size(group)
        if world_size == 1:
            return grads

        gathered = []
        for grad in grads:
            # All-gather
            tensor_list = [torch.zeros_like(grad) for _ in range(world_size)]
            dist.all_gather(tensor_list, grad.contiguous(), group=group)
            # 拼接所有分片
            gathered.append(torch.cat(tensor_list, dim=0))
        return gathered
    except Exception as e:
        logger.warning(f"组内梯度聚合失败: {e}")
        return grads


class HDSPMuonOptimizer(Optimizer):
    """HSDP 模式的 Muon 优化器

    实现 Hybrid Data Sharding Parallel (HSDP) 策略：
    - 节点内: 使用 FSDP 分片，梯度在节点内 all-gather
    - 节点间: 数据并行，参数复制，不跨节点通信

    继承自 FSDPMuonOptimizer，扩展支持 2D DeviceMesh。

    Args:
        model: 要优化的模型（必须使用 FSDP2 包装）
        device_mesh: 2D DeviceMesh 对象
        dp_group: 数据并行进程组（可选）
        fsdp_group: FSDP 分片进程组（可选）
        params: 要优化的参数列表
        lr: 学习率，默认 0.02
        weight_decay: 权重衰减，默认 0.01
        momentum: 动量系数，默认 0.95
        nesterov: 是否使用 Nesterov 动量，默认 True
        ns_steps: Newton-Schulz 迭代次数，默认 5
        ns_stepsize: Newton-Schulz 步长，默认 1.0
        beta2: 二阶矩系数（Adam 风格），默认 0.99
        eps: 数值稳定性 epsilon，默认 1e-8
        gradient_accumulation_steps: 梯度累积步数，默认 1

    Example:
        >>> from torch.distributed.device_mesh import DeviceMesh
        >>> mesh = DeviceMesh("cuda", [[0, 1], [2, 3]],
        ...                    mesh_dim_names=("dp_replicate", "fsdp_shard"))
        >>> model = nn.Linear(512, 512)
        >>> fully_shard(model, device_mesh=mesh, sharding_strategy=...)
        >>> optimizer = HDSPMuonOptimizer(model, device_mesh=mesh, lr=0.02)
    """

    def __init__(
        self,
        model: nn.Module,
        device_mesh: Optional[Any] = None,
        dp_group: Optional[Any] = None,
        fsdp_group: Optional[Any] = None,
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
        self.model = model
        self.device_mesh = device_mesh
        self.dp_group = dp_group
        self.fsdp_group = fsdp_group
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._step_count = 0

        # 尝试自动检测 HSDP 配置
        if device_mesh is None and is_available():
            device_mesh = self._detect_device_mesh()

        if device_mesh is not None:
            dp_group, fsdp_group = get_hsdp_groups(device_mesh)
            if self.dp_group is None:
                self.dp_group = dp_group
            if self.fsdp_group is None:
                self.fsdp_group = fsdp_group

        # 默认参数
        if params is None:
            params = list(model.parameters())
        params = [p for p in params if p.requires_grad]

        if not params:
            raise ValueError("HDSPMuonOptimizer 收到空参数列表")

        # 记录 HSDP 配置
        if self.dp_group is not None and self.fsdp_group is not None:
            logger.info("检测到 HSDP 配置: 节点内分片 + 节点间复制")

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

        # 初始化状态
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["momentum_buffer"] = torch.zeros_like(p)
                state["second_moment"] = torch.zeros_like(p)
                state["accum_count"] = 0

    def _detect_device_mesh(self) -> Optional[Any]:
        """自动检测 DeviceMesh 配置"""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            for module in self.model.modules():
                if isinstance(module, FSDP):
                    if hasattr(module, "_device_mesh"):
                        return module._device_mesh
        except Exception:
            pass
        return None

    @contextmanager
    def unshard_params(self) -> Generator[None, None, None]:
        """上下文管理器：取消分片参数

        在节点内取消 FSDP 分片，使可以访问完整参数。
        """
        if not has_fsdp_modules(self.model):
            yield
            return

        handles = []
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            for module in self.model.modules():
                if isinstance(module, FSDP):
                    handle = module.unshard()
                    handles.append(handle)
                    handle.__enter__()
            yield
        finally:
            for handle in handles:
                try:
                    handle.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"重新分片时出错: {e}")

    def _gather_gradients(
        self,
        params: List[torch.nn.Parameter],
    ) -> List[torch.Tensor]:
        """在 HSDP 组内聚合梯度

        关键区别于 FSDP：
        - 只在 FSDP 组内（节点内）进行 all-gather
        - 不跨节点聚合（节点间数据并行）
        """
        local_grads = []

        for p in params:
            if p.grad is None:
                grad = torch.zeros_like(get_dtensor_local_tensor(p))
            else:
                grad = p.grad

            if is_dtensor(p):
                local_grad = get_dtensor_local_tensor(grad)
            else:
                local_grad = grad

            local_grads.append(local_grad)

        # HSDP: 只在 FSDP 组内聚合（节点内）
        if self.fsdp_group is not None and is_available():
            gathered_grads = gather_grads_group(local_grads, self.fsdp_group)
        else:
            gathered_grads = local_grads

        return gathered_grads

    def step(self, closure: Any = None) -> Optional[float]:
        """执行优化器步骤

        1. 聚合梯度（在 FSDP 组内）
        2. 应用权重衰减和动量
        3. Newton-Schulz 正交化
        4. 更新参数
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            params = group["params"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            ns_stepsize = group["ns_stepsize"]
            beta2 = group["beta2"]
            eps = group["eps"]

            # 聚合梯度
            gathered_grads = self._gather_gradients(params)

            # 处理每个参数
            for i, p in enumerate(params):
                if p not in self.state:
                    continue

                state = self.state[p]
                grad = gathered_grads[i] if i < len(gathered_grads) else None

                if grad is None:
                    continue

                # 更新累积计数
                state["accum_count"] = state.get("accum_count", 0) + 1
                if state["accum_count"] < self.gradient_accumulation_steps:
                    continue

                # 重置累积计数
                state["accum_count"] = 0

                # 动量更新
                momentum_buffer = state["momentum_buffer"]
                if momentum > 0:
                    momentum_buffer.mul_(momentum).add_(grad, alpha=1 - momentum)
                    if nesterov:
                        grad = grad + momentum_buffer * momentum
                    else:
                        grad = momentum_buffer

                # 权重衰减
                if weight_decay > 0 and p.dim() > 1:
                    p.mul_(1 - lr * weight_decay)

                # 只有 2D 参数需要正交化
                if p.dim() > 1:
                    # 学习率缩放: lr * sqrt(d_out / d_in)
                    d_out, d_in = p.shape
                    lr_scaled = lr * (max(d_out, d_in) / min(d_out, d_in)) ** 0.5

                    # Newton-Schulz 正交化
                    grad_ortho = zeropower_via_newtonschulz5(
                        grad.to(torch.float32),
                        steps=ns_steps,
                    ).to(grad.dtype)

                    # 应用更新
                    p.add_(grad_ortho, alpha=-lr_scaled * ns_stepsize)
                else:
                    # 1D 参数直接更新
                    p.add_(grad, alpha=-lr)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """清零梯度"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad is not None:
                            p.grad.grad = None
                        p.grad.detach_()
                        p.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        """返回优化器状态字典"""
        state = {
            "step_count": self._step_count,
            "param_groups": self.param_groups,
        }

        # 简化状态（避免存储完整梯度）
        for group in state["param_groups"]:
            group["params"] = []

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载优化器状态"""
        self._step_count = state_dict.get("step_count", 0)
        super().load_state_dict(state_dict)


class HSDPConfig:
    """HSDP 配置数据类

    用于简化 HSDP 优化器的创建和配置。

    Attributes:
        dp_size: 数据并行大小（节点数）
        fsdp_size: FSDP 分片大小（每节点 GPU 数）
        device_mesh: 2D DeviceMesh 对象
        lr: 学习率
        momentum: 动量系数
        weight_decay: 权重衰减
        ns_steps: Newton-Schulz 迭代次数

    Example:
        >>> config = HSDPConfig(
        ...     dp_size=2,
        ...     fsdp_size=4,
        ...     lr=0.02,
        ...     momentum=0.95,
        ... )
        >>> optimizer = HDSPMuonOptimizer(model, **config.to_dict())
    """

    def __init__(
        self,
        dp_size: int = 1,
        fsdp_size: int = 1,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        nesterov: bool = True,
    ):
        self.dp_size = dp_size
        self.fsdp_size = fsdp_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.nesterov = nesterov

    def create_device_mesh(self) -> Optional[Any]:
        """根据配置创建 DeviceMesh"""
        return create_device_mesh_2d(self.dp_size, self.fsdp_size)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "dp_size": self.dp_size,
            "fsdp_size": self.fsdp_size,
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "ns_steps": self.ns_steps,
            "nesterov": self.nesterov,
        }
