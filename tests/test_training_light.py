#!/usr/bin/env python3
"""轻量级训练测试 - 展示 Muon 优化器的训练效果。"""

import torch
import torch.nn as nn
import time

from muon_fsdp import MuonOptimizer


class TinyModel(nn.Module):
    """小型测试模型。"""

    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
        )

    def forward(self, x):
        return self.net(x)


def test_training():
    """测试训练效果。"""
    print("=" * 60)
    print(" Muon 优化器训练测试")
    print("=" * 60)

    # 配置
    dim = 256
    batch_size = 16
    num_steps = 10
    device = "cpu"

    # 创建模型
    model = TinyModel(dim=dim).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型信息:")
    print(f"  参数量: {num_params:,}")
    print(f"  维度: {dim}")
    print(f"  设备: {device}")

    # 创建优化器
    optimizer = MuonOptimizer(
        model.parameters(),
        lr=0.02,
        momentum=0.95,
        weight_decay=0.01,
        ns_steps=5,
    )
    print(f"\n优化器配置:")
    print(f"  学习率: 0.02")
    print(f"  动量: 0.95")
    print(f"  NS 迭代: 5")

    # 生成固定随机数据
    torch.manual_seed(42)
    x = torch.randn(batch_size, dim, device=device)
    target = torch.randn(batch_size, dim // 2, device=device)

    print(f"\n训练 {num_steps} 步...")
    print(f"{'Step':>6} {'Loss':>12} {'Time(ms)':>10}")
    print("-" * 32)

    losses = []
    times = []

    for step in range(num_steps):
        start = time.time()

        # 前向
        output = model(x)
        loss = nn.functional.mse_loss(output, target)

        # 反向
        optimizer.zero_grad()
        loss.backward()

        # 优化步骤
        optimizer.step()

        elapsed = (time.time() - start) * 1000

        losses.append(loss.item())
        times.append(elapsed)

        print(f"{step + 1:>6} {loss.item():>12.6f} {elapsed:>10.2f}")

    # 统计
    print("\n训练统计:")
    print(f"  初始 loss: {losses[0]:.6f}")
    print(f"  最终 loss: {losses[-1]:.6f}")
    print(f"  Loss 下降: {losses[0] - losses[-1]:.6f} ({(1 - losses[-1] / losses[0]) * 100:.1f}%)")
    print(f"  平均耗时: {sum(times) / len(times):.2f} ms/步")
    print(f"  总耗时: {sum(times) / 1000:.2f} 秒")

    print("\n" + "=" * 60)
    print(" 训练测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_training()
