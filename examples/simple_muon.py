"""简单 Muon 优化器使用示例

本示例展示如何在简单的 MLP 模型上使用 Muon 优化器进行训练。
适合初学者理解 Muon 的基本用法。

特点：
- 简单的多层感知机 (MLP) 模型
- 完整的训练循环演示
- 详细的注释说明每个步骤
- 可在 CPU 上运行

使用方法:
    python examples/simple_muon.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from muon_fsdp import MuonOptimizer


class SimpleMLP(nn.Module):
    """简单的多层感知机模型

    用于演示 Muon 优化器的基本使用方法。
    模型包含两个线性层和一个激活函数。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_dummy_dataset(
    n_samples: int = 1000,
    input_dim: int = 128,
    output_dim: int = 10,
    noise: float = 0.1,
) -> DataLoader:
    """创建虚拟数据集用于训练演示

    Args:
        n_samples: 样本数量
        input_dim: 输入维度
        output_dim: 输出维度（分类数）
        noise: 标签噪声比例

    Returns:
        DataLoader 对象
    """
    # 生成随机数据
    x = torch.randn(n_samples, input_dim)
    # 生成线性标签（添加噪声）
    true_w = torch.randn(input_dim, output_dim)
    y = x @ true_w + torch.randn(n_samples, output_dim) * noise
    # 转换为分类标签
    y = y.argmax(dim=1)

    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def train_with_muon(
    model: nn.Module,
    train_loader: DataLoader,
    lr: float = 0.02,
    epochs: int = 10,
    momentum: float = 0.95,
    weight_decay: float = 0.01,
    ns_steps: int = 5,
) -> None:
    """使用 Muon 优化器训练模型

    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        lr: 学习率
        epochs: 训练轮数
        momentum: 动量系数
        weight_decay: 权重衰减
        ns_steps: Newton-Schulz 迭代次数
    """
    # 创建 Muon 优化器
    # Muon 优化器会自动处理 2D 权重矩阵的正交化
    optimizer = MuonOptimizer(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        ns_steps=ns_steps,
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    print(f"开始训练: {epochs} 轮, 学习率: {lr}")
    print("-" * 50)

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # 打印训练进度
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")

    print("-" * 50)
    print("训练完成!")


def demo_muon_optimizer_behavior():
    """演示 Muon 优化器的核心行为

    展示 Muon 如何保持权重矩阵的正交性。
    """
    print("\n=== Muon 优化器行为演示 ===\n")

    # 创建一个小模型用于演示
    model = nn.Linear(100, 100)
    optimizer = MuonOptimizer(model.parameters(), lr=0.02, ns_steps=5)

    # 训练几步
    for i in range(5):
        x = torch.randn(10, 100)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

    # 检查权重矩阵的正交性
    # 正交矩阵满足: W @ W^T ≈ I
    w = model.weight.data
    gram_matrix = w @ w.T
    identity = torch.eye(w.shape[0])
    orthogonality_error = (gram_matrix - identity).norm().item()

    print(f"权重矩阵形状: {w.shape}")
    print(f"正交性误差 (||W @ W^T - I||): {orthogonality_error:.6f}")
    print(f"理想值: 接近 0 (误差 < 0.1 表示良好的正交性)")

    if orthogonality_error < 0.1:
        print("✓ 权重矩阵保持了良好的正交性!")
    else:
        print("✗ 正交性误差较大，可能需要调整参数")


def main():
    """主函数"""
    print("=" * 50)
    print("简单 Muon 优化器使用示例")
    print("=" * 50)

    # 设置随机种子以保证可复现性
    torch.manual_seed(42)

    # 配置参数
    input_dim = 128
    hidden_dim = 256
    output_dim = 10
    n_samples = 1000
    epochs = 10
    lr = 0.02
    momentum = 0.95
    weight_decay = 0.01

    print(f"\n配置:")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏维度: {hidden_dim}")
    print(f"  输出维度: {output_dim}")
    print(f"  样本数量: {n_samples}")
    print(f"  训练轮数: {epochs}")
    print(f"  学习率: {lr}")
    print(f"  动量: {momentum}")
    print(f"  权重衰减: {weight_decay}")

    # 创建模型
    model = SimpleMLP(input_dim, hidden_dim, output_dim)

    # 打印模型参数数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,}")

    # 创建数据集
    train_loader = create_dummy_dataset(
        n_samples=n_samples,
        input_dim=input_dim,
        output_dim=output_dim,
    )

    # 训练模型
    train_with_muon(
        model=model,
        train_loader=train_loader,
        lr=lr,
        epochs=epochs,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # 演示 Muon 的正交化行为
    demo_muon_optimizer_behavior()

    print("\n" + "=" * 50)
    print("示例完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
