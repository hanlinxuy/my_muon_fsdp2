"""HSDP 训练示例

展示如何使用 HDSPMuonOptimizer 进行混合数据分片并行训练。
HSDP = 节点内 FSDP 分片 + 节点间数据并行

使用方式:
    # 单节点多 GPU (2节点 x 2 GPU = 4 GPU)
    torchrun --nproc_per_node=4 examples/hdsp_training.py

    # 多节点 (2节点，每节点2 GPU)
    torchrun --nnodes=2 --nproc_per_node=2 examples/hdsp_training.py
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset

from muon_fsdp import HDSPMuonOptimizer, HSDPConfig
from muon_fsdp.hdsp import create_device_mesh_2d, get_hsdp_groups


def setup_distributed():
    """初始化分布式环境"""
    if not dist.is_available():
        return False, 0, 1

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return True, rank, world_size


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_hsdp_device_mesh(world_size: int, dp_size: int = None):
    """创建 HSDP 2D DeviceMesh

    Args:
        world_size: 总进程数
        dp_size: 数据并行大小（节点数），自动计算如果未提供

    Returns:
        DeviceMesh 或 None
    """
    if dp_size is None:
        import math

        dp_size = int(math.sqrt(world_size))

    fsdp_size = world_size // dp_size

    if fsdp_size < 1:
        fsdp_size = 1

    mesh = create_device_mesh_2d(dp_size, fsdp_size)
    return mesh, dp_size, fsdp_size


class SimpleTransformer(nn.Module):
    """简单的 Transformer 模型"""

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)


def create_dummy_data(n_samples: int = 200, seq_len: int = 32, vocab_size: int = 1000):
    """创建虚拟训练数据"""
    x = torch.randint(0, vocab_size, (n_samples, seq_len))
    y = torch.randint(0, vocab_size, (n_samples, seq_len))
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)


def train_hsdp(
    model: nn.Module,
    train_loader: DataLoader,
    device_mesh,
    dp_group,
    fsdp_group,
    rank: int,
    lr: float = 0.02,
    epochs: int = 5,
):
    """使用 HSDP 训练模型

    Args:
        model: HSDP 包装的模型
        train_loader: 训练数据
        device_mesh: 2D DeviceMesh
        dp_group: 数据并行进程组
        fsdp_group: FSDP 分片进程组
        rank: 当前进程 rank
        lr: 学习率
        epochs: 训练轮数
    """
    optimizer = HDSPMuonOptimizer(
        model,
        device_mesh=device_mesh,
        dp_group=dp_group,
        fsdp_group=fsdp_group,
        lr=lr,
        momentum=0.95,
        weight_decay=0.01,
    )

    criterion = nn.CrossEntropyLoss()

    print(f"[Rank {rank}] HSDP 训练开始: {epochs} 轮")
    print(f"[Rank {rank}] DeviceMesh: {device_mesh}")

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    if rank == 0:
        print("训练完成!")


def train_single_gpu(model, train_loader, lr=0.02, epochs=5):
    """单 GPU 训练（回退模式）"""
    from muon_fsdp import MuonOptimizer

    optimizer = MuonOptimizer(model.parameters(), lr=lr, momentum=0.95)
    criterion = nn.CrossEntropyLoss()

    print(f"单 GPU 模式训练: {epochs} 轮")

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

    print("训练完成!")


def main():
    parser = argparse.ArgumentParser(description="HSDP 训练示例")
    parser.add_argument("--lr", type=float, default=0.02, help="学习率")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--dp_size", type=int, default=None, help="数据并行大小")
    args = parser.parse_args()

    print("=" * 50)
    print("HSDP 训练示例")
    print("=" * 50)

    torch.manual_seed(42)

    is_distributed, rank, world_size = setup_distributed()

    if is_distributed:
        torch.cuda.set_device(rank)

    config = {
        "vocab_size": 500,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
    }

    train_loader = create_dummy_data(
        n_samples=100,
        seq_len=16,
        vocab_size=config["vocab_size"],
    )

    if is_distributed:
        print(f"\n分布式模式: {world_size} GPUs")

        device_mesh, dp_size, fsdp_size = create_hsdp_device_mesh(world_size, args.dp_size)
        dp_group, fsdp_group = get_hsdp_groups(device_mesh)

        if rank == 0:
            print(f"DP size: {dp_size}, FSDP size: {fsdp_size}")

        model = SimpleTransformer(**config).cuda()

        train_hsdp(
            model=model,
            train_loader=train_loader,
            device_mesh=device_mesh,
            dp_group=dp_group,
            fsdp_group=fsdp_group,
            rank=rank,
            lr=args.lr,
            epochs=args.epochs,
        )
    else:
        print("\n单 GPU 模式")
        model = SimpleTransformer(**config)

        train_single_gpu(
            model=model,
            train_loader=train_loader,
            lr=args.lr,
            epochs=args.epochs,
        )

    cleanup_distributed()

    print("\n" + "=" * 50)
    print("示例完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
