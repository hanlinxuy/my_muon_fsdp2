#!/usr/bin/env python3
"""演示 Muon FSDP2 分布式训练功能的测试脚本。

此脚本展示：
1. 梯度全收集 (all_gather_grads) - 跨进程收集梯度
2. Newton-Schulz 正交化 - 在完整梯度上计算更新
3. 参数更新分片 - 将更新分片回各个进程

注意：由于当前环境限制（macOS + CPU），无法运行真正的多进程 FSDP2，
但单进程模式可以展示所有功能逻辑。
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from muon_fsdp import FSDPMuonOptimizer
from muon_fsdp.distributed import all_gather_grads, get_world_size, get_rank


def print_section(title):
    """打印章节标题。"""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def test_gradient_all_gather():
    """测试梯度全收集功能。"""
    print_section("Test 1: Gradient All-Gather")

    # 模拟多个进程的本地梯度
    world_size = 4
    local_dim = 64
    feature_dim = 128

    print(f"\n模拟 {world_size} 个进程，每个进程有本地梯度:")
    print(f"  本地梯度形状: ({local_dim}, {feature_dim})")

    # 创建模拟的本地梯度
    local_grad = torch.randn(local_dim, feature_dim)
    print(f"  本地梯度 (rank 0): {tuple(local_grad.shape)}")

    # 模拟 all-gather 后的结果
    gathered_shape = (world_size * local_dim, feature_dim)
    print(f"\nAll-gather 后:")
    print(f"  收集的梯度形状: {gathered_shape}")
    print(f"  包含所有 {world_size} 个进程的梯度")

    # 实际测试 all_gather_grads 函数（单进程模式）
    grads = [local_grad]
    result = all_gather_grads(grads)
    print(f"\n实际调用 all_gather_grads():")
    print(f"  输入: {tuple(grads[0].shape)}")
    print(f"  输出: {tuple(result[0].shape)}")
    print(f"  (单进程模式下直接返回原梯度)")


def test_newton_schulz_computation():
    """测试 Newton-Schulz 正交化计算。"""
    print_section("Test 2: Newton-Schulz Orthogonalization")

    from muon_fsdp.utils import zeropower_via_newtonschulz5

    # 创建一个梯度矩阵
    grad = torch.randn(128, 128)
    print(f"\n输入梯度矩阵: {tuple(grad.shape)}")

    # 应用 Newton-Schulz 迭代
    ns_steps = 5
    orthogonalized = zeropower_via_newtonschulz5(grad, steps=ns_steps)

    print(f"\nNewton-Schulz 迭代 ({ns_steps} steps):")
    print(f"  输出正交矩阵: {tuple(orthogonalized.shape)}")

    # 验证正交性
    identity = orthogonalized @ orthogonalized.T
    error = torch.norm(identity - torch.eye(128)) / torch.norm(torch.eye(128))
    print(f"\n正交性验证:")
    print(f"  ||G @ G^T - I|| / ||I|| = {error.item():.6f}")
    print(f"  (值越小越接近正交)")


def test_optimizer_step():
    """测试完整的优化器步骤。"""
    print_section("Test 3: Complete Optimizer Step")

    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.GELU(),
        nn.Linear(256, 128),
    )

    # 创建优化器
    optimizer = FSDPMuonOptimizer(
        model=model,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        ns_steps=5,
    )

    print(f"\n模型结构:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    print(f"  FSDP modules: {len(optimizer.fsdp_modules)}")

    # 模拟训练步骤
    print(f"\n执行训练步骤:")

    # 前向传播
    x = torch.randn(8, 128)
    y = model(x)
    loss = y.pow(2).mean()
    print(f"  初始 loss: {loss.item():.4f}")

    # 反向传播
    loss.backward()

    # 检查梯度
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    print(f"  梯度范数范围: [{min(grad_norms):.4f}, {max(grad_norms):.4f}]")

    # 优化步骤
    optimizer.step()
    optimizer.zero_grad()

    # 验证更新
    y2 = model(x)
    loss2 = y2.pow(2).mean()
    print(f"  更新后 loss: {loss2.item():.4f}")
    print(f"  Loss 变化: {loss.item() - loss2.item():.4f} ({'下降' if loss2 < loss else '上升'})")


def test_distributed_flow():
    """展示分布式训练流程。"""
    print_section("Test 4: Distributed Training Flow")

    print("""
在真正的多进程分布式训练中，流程如下：

┌─────────────────────────────────────────────────────────────┐
│  Rank 0          Rank 1          Rank 2          Rank 3     │
│    │               │               │               │        │
│    ▼               ▼               ▼               ▼        │
│ ┌──────┐       ┌──────┐       ┌──────┐       ┌──────┐       │
│ │Grad 0│       │Grad 1│       │Grad 2│       │Grad 3│       │
│ │(64,128)│     │(64,128)│     │(64,128)│     │(64,128)│     │
│ └──┬───┘       └──┬───┘       └──┬───┘       └──┬───┘       │
│    │               │               │               │        │
│    └───────────────┴───────────────┴───────────────┘        │
│                    │                                        │
│                    ▼                                        │
│            all_gather_grads()                               │
│                    │                                        │
│                    ▼                                        │
│            ┌──────────────┐                                 │
│            │ Gathered Grad│                                 │
│            │ (256, 128)   │                                 │
│            └──────┬───────┘                                 │
│                   │                                         │
│                   ▼                                         │
│            zeropower_via_newtonschulz5()                    │
│                   │                                         │
│                   ▼                                         │
│            ┌──────────────┐                                 │
│            │ Orthogonal   │                                 │
│            │ Update       │                                 │
│            └──────┬───────┘                                 │
│                   │                                         │
│                   ▼                                         │
│            scatter_updates()                                │
│                   │                                         │
│    ┌──────────────┼──────────────┐                         │
│    ▼              ▼              ▼              ▼          │
│ ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐           │
│ │Update│     │Update│     │Update│     │Update│           │
│ │(64,128)│   │(64,128)│   │(64,128)│   │(64,128)│         │
│ └──────┘     └──────┘     └──────┘     └──────┘           │
└─────────────────────────────────────────────────────────────┘

关键步骤:
1. 每个进程计算本地梯度 (sharded)
2. all_gather_grads() 收集所有梯度
3. Newton-Schulz 在完整梯度上计算正交更新
4. 更新分片回各个进程
    """)


def main():
    print("\n" + "=" * 60)
    print(" Muon FSDP2 分布式训练功能测试")
    print("=" * 60)
    print(f"\n环境信息:")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  当前进程数: {get_world_size()}")
    print(f"  当前 rank: {get_rank()}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")

    # 运行所有测试
    test_gradient_all_gather()
    test_newton_schulz_computation()
    test_optimizer_step()
    test_distributed_flow()

    print("\n" + "=" * 60)
    print(" 所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
