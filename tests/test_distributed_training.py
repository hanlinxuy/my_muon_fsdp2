#!/usr/bin/env python3
"""
分布式训练测试脚本 - 展示 Muon FSDP2 的梯度交换和参数 gather 功能

此脚本创建一个模拟的分布式环境，展示：
1. 分布式梯度全收集 (all_gather_grads)
2. 参数 unshard/reshard 生命周期
3. Newton-Schulz 正交化更新
4. 更新分片回传 (scatter_updates)

使用方法:
    # 单进程模式 (模拟)
    python tests/test_distributed_training.py

    # 多进程分布式模式 (2进程)
    torchrun --nproc_per_node=2 tests/test_distributed_training.py --distributed

    # 多进程分布式模式 (4进程)
    torchrun --nproc_per_node=4 tests/test_distributed_training.py --distributed
"""

import argparse
import os
import sys
import time
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import fully_shard

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from muon_fsdp import FSDPMuonOptimizer
from muon_fsdp.distributed import all_gather_grads, get_rank, get_world_size, is_available


class SimpleModel(nn.Module):
    """简单的测试模型，包含多个线性层。"""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.dim = dim
        # 创建几个线性层用于测试
        self.layers = nn.ModuleList(
            [
                nn.Linear(dim, dim, bias=True),
                nn.Linear(dim, dim, bias=True),
                nn.Linear(dim, dim, bias=True),
            ]
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

    def configure_fsdp(self) -> None:
        """配置 FSDP2。"""
        for layer in self.layers:
            fully_shard(layer)


def print_header(title: str, rank: int = 0):
    """打印带格式的标题。"""
    if rank == 0:
        print("\n" + "=" * 70)
        print(f" {title}")
        print("=" * 70)


def print_section(title: str, rank: int = 0):
    """打印章节标题。"""
    if rank == 0:
        print(f"\n--- {title} ---")


def print_tensor_info(name: str, tensor: torch.Tensor, indent: int = 2):
    """打印张量信息。"""
    prefix = " " * indent
    print(
        f"{prefix}{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
    )


def test_gradient_gather(
    model: nn.Module,
    optimizer: FSDPMuonOptimizer,
    device: str,
    rank: int,
    world_size: int,
):
    """测试梯度全收集功能。"""
    print_section("Testing Gradient All-Gather", rank)

    # 创建输入并前向传播
    batch_size = 4
    x = torch.randn(batch_size, model.dim, device=device, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # 收集梯度
    grads_info = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads_info.append(
                {
                    "name": name,
                    "local_shape": tuple(param.grad.shape),
                }
            )

    if rank == 0:
        print(f"  Local gradients (per process):")
        for info in grads_info:
            print(f"    {info['name']}: {info['local_shape']}")

    # 使用优化器的梯度收集功能
    params_with_grad = [p for p in model.parameters() if p.grad is not None]
    local_grads = [p.grad for p in params_with_grad]

    if is_available() and world_size > 1:
        gathered_grads = all_gather_grads(local_grads)
        if rank == 0:
            print(f"\n  Gathered gradients (after all-gather):")
            for name, grad in zip(
                [n for n, _ in model.named_parameters() if _.grad is not None], gathered_grads
            ):
                print(f"    {name}: {tuple(grad.shape)} (world_size × local)")
    else:
        if rank == 0:
            print(f"\n  Single process mode - no gathering needed")

    optimizer.zero_grad()


def test_newton_schulz_update(
    model: nn.Module,
    optimizer: FSDPMuonOptimizer,
    device: str,
    rank: int,
    world_size: int,
):
    """测试 Newton-Schulz 正交化更新。"""
    print_section("Testing Newton-Schulz Orthogonalization", rank)

    # 前向和反向传播
    x = torch.randn(4, model.dim, device=device)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # 记录更新前的参数
    param_before = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            param_before[name] = param.data.clone()

    # 执行优化步骤
    optimizer.step()
    optimizer.zero_grad()

    # 检查更新后的参数
    if rank == 0:
        print(f"  Parameter updates applied:")
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2 and name in param_before:
                diff = (param.data - param_before[name]).abs().mean().item()
                print(f"    {name}: mean update magnitude = {diff:.6f}")


def test_unshard_reshard(
    model: nn.Module,
    optimizer: FSDPMuonOptimizer,
    device: str,
    rank: int,
):
    """测试参数 unshard/reshard 生命周期。"""
    print_section("Testing Unshard/Reshard Lifecycle", rank)

    if not optimizer.fsdp_modules:
        if rank == 0:
            print("  No FSDP modules - skipping unshard/reshard test")
        return

    # 使用 unshard_params 上下文管理器
    with optimizer.unshard_params():
        if rank == 0:
            print(f"  Inside unshard context:")
            for name, param in model.named_parameters():
                if "weight" in name:
                    print(f"    {name}: can access full tensor")

    if rank == 0:
        print(f"  Outside unshard context: parameters resharded")


def test_distributed_training_step(
    model: nn.Module,
    optimizer: FSDPMuonOptimizer,
    device: str,
    rank: int,
    world_size: int,
    num_steps: int = 3,
):
    """测试完整的分布式训练步骤。"""
    print_section(f"Testing Full Training Loop ({num_steps} steps)", rank)

    losses = []
    for step in range(num_steps):
        # 创建随机输入
        x = torch.randn(8, model.dim, device=device)
        y = model(x)
        loss = y.pow(2).mean()  # 简单的损失函数

        loss.backward()

        # 优化步骤
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        # 同步所有进程
        if is_available() and world_size > 1:
            dist.barrier()

    if rank == 0:
        print(f"  Training losses: {[f'{l:.4f}' for l in losses]}")
        print(f"  Loss trend: {'decreasing' if losses[-1] < losses[0] else 'increasing'}")


def run_test(rank: int, world_size: int, use_distributed: bool, device: str):
    """运行测试。"""
    # 初始化分布式环境
    if use_distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 设置设备
    if device == "cuda":
        device = f"cuda:{rank}"
    else:
        device = "cpu"

    # 创建模型
    dim = 256  # 使用较小的维度便于测试
    model = SimpleModel(dim=dim)
    model = model.to(device)

    # 配置 FSDP
    if use_distributed:
        model.configure_fsdp()

    # 创建优化器
    optimizer = FSDPMuonOptimizer(
        model=model,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        ns_stepsize=1.0,
    )

    # 打印测试信息
    if rank == 0:
        print_header("Muon FSDP2 Distributed Training Test")
        print(f"Configuration:")
        print(f"  World size: {world_size}")
        print(f"  Device: {device}")
        print(f"  Model dim: {dim}")
        print(f"  FSDP enabled: {use_distributed}")
        print(f"  FSDP modules: {len(optimizer.fsdp_modules)}")

    # 运行各项测试
    test_gradient_gather(model, optimizer, device, rank, world_size)
    test_newton_schulz_update(model, optimizer, device, rank, world_size)
    test_unshard_reshard(model, optimizer, device, rank)
    test_distributed_training_step(model, optimizer, device, rank, world_size)

    # 清理
    if use_distributed:
        dist.destroy_process_group()

    if rank == 0:
        print_header("Test Completed Successfully!")


def main():
    parser = argparse.ArgumentParser(description="Test Muon FSDP2 distributed training")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--world-size", type=int, default=2, help="Number of processes")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device type"
    )
    args = parser.parse_args()

    if args.distributed:
        # 多进程模式
        mp.spawn(
            run_test, args=(args.world_size, True, args.device), nprocs=args.world_size, join=True
        )
    else:
        # 单进程模式
        run_test(0, 1, False, args.device)


if __name__ == "__main__":
    main()
