"""单 GPU 示例 - 展示如何在非分布式环境下使用 fsdp2_utils。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


def main() -> int:
    print("=" * 50)
    print("单 GPU 示例 - fsdp2_utils")
    print("=" * 50)

    # 1. 导入模块
    from fsdp2_utils import MuonOptimizer, zeropower_via_newtonschulz5
    from fsdp2_utils.dtensor import is_dtensor
    from fsdp2_utils.comm import is_distributed, get_world_size

    print(f"\n[环境信息]")
    print(f"  分布式可用: {is_distributed()}")
    print(f"  World Size: {get_world_size()}")

    # 2. 创建模型
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.GELU(),
        nn.Linear(512, 256),
    )

    print(f"\n[模型信息]")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    for name, param in model.named_parameters():
        print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

    # 3. 创建 Muon 优化器
    optimizer = MuonOptimizer(
        model,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.01,
        nesterov=False,
        ns_steps=5,
    )

    print(f"\n[优化器信息]")
    print(f"  类型: MuonOptimizer")
    print(f"  学习率: 0.02")
    print(f"  动量: 0.95")
    print(f"  Newton-Schulz 步数: 5")

    # 4. 创建虚拟数据
    batch_size = 32
    x = torch.randn(batch_size, 256)
    y = torch.randn(batch_size, 256)

    # 5. 训练循环
    print(f"\n[训练步骤]")
    model.train()
    for step in range(3):
        optimizer.zero_grad()

        output = model(x)
        loss = nn.functional.mse_loss(output, y)

        loss.backward()
        optimizer.step()

        print(f"  Step {step + 1}: loss = {loss.item():.6f}")

    # 6. 测试 Newton-Schulz 正交化
    print(f"\n[Newton-Schulz 测试]")
    G = torch.randn(512, 512)
    W = zeropower_via_newtonschulz5(G, steps=5)
    orthogonality = (W @ W.T - torch.eye(512)).norm().item()
    print(f"  正交性误差: {orthogonality:.6f}")

    # 7. 测试 DTensor 检查（非分布式环境应返回 False）
    print(f"\n[DTensor 检查]")
    test_tensor = torch.randn(4, 4)
    print(f"  普通张量 is_dtensor: {is_dtensor(test_tensor)}")

    print("\n" + "=" * 50)
    print("示例完成!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
