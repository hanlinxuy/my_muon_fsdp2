"""FSDP2 分布式训练示例

展示如何使用 FSDPMuonOptimizer 进行多 GPU 分布式训练。
如果没有多 GPU 环境，会自动回退到单 GPU 模式。

特点：
- 自动检测分布式环境
- 单 GPU 回退支持
- FSDP2 模型包装演示
- 分布式训练循环

使用方式:
    # 单 GPU 模式
    python examples/fsdp_distributed.py

    # 多 GPU 分布式模式 (需要 2+ GPUs)
    torchrun --nproc_per_node=2 examples/fsdp_distributed.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed.fsdp import fully_shard

from muon_fsdp import FSDPMuonOptimizer, MuonOptimizer


def setup_distributed():
    """初始化分布式环境

    检测是否在分布式环境中运行，如果是则初始化，
    否则返回单 GPU 模式。

    Returns:
        tuple: (is_distributed, rank, world_size)
    """
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return True, rank, world_size
    elif dist.is_available() and dist.is_nccl_available():
        if torch.cuda.device_count() >= 2:
            dist.init_process_group(backend="nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            return True, rank, world_size
    return False, 0, 1


def cleanup_distributed(is_distributed: bool):
    """清理分布式环境"""
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


class SimpleTransformer(nn.Module):
    """简单的 Transformer 模型用于演示 FSDP2"""

    def __init__(
        self,
        vocab_size: int = 1000,
        max_seq_len: int = 128,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        return x


def create_dummy_data(
    n_samples: int = 500,
    seq_len: int = 32,
    vocab_size: int = 1000,
) -> DataLoader:
    """创建虚拟训练数据"""
    x = torch.randint(0, vocab_size, (n_samples, seq_len))
    y = torch.randint(0, vocab_size, (n_samples, seq_len))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def apply_fsdp2(model: nn.Module) -> nn.Module:
    """使用 FSDP2 包装模型

    对模型的每一层应用 fully_shard 包装，
    使其可以在多 GPU 上分布式训练。
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            apply_fsdp2(module)
        else:
            fully_shard(module)
    fully_shard(model)
    return model


def train_fsdp(
    model: nn.Module,
    train_loader: DataLoader,
    is_distributed: bool,
    rank: int,
    world_size: int,
    lr: float = 0.02,
    epochs: int = 5,
) -> None:
    """使用 FSDP2 分布式训练

    Args:
        model: FSDP2 包装的模型
        train_loader: 训练数据
        is_distributed: 是否分布式模式
        rank: 当前进程 rank
        world_size: 总进程数
        lr: 学习率
        epochs: 训练轮数
    """
    optimizer = FSDPMuonOptimizer(
        model,
        lr=lr,
        momentum=0.95,
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"[Rank {rank}] 开始训练: {epochs} 轮")
    print(f"[Rank {rank}] 分布式模式: {is_distributed}, World size: {world_size}")

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
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


def train_single_gpu(
    model: nn.Module,
    train_loader: DataLoader,
    lr: float = 0.02,
    epochs: int = 5,
) -> None:
    """单 GPU 训练（回退模式）

    当没有多 GPU 环境时使用 MuonOptimizer 进行训练。
    """
    optimizer = MuonOptimizer(
        model.parameters(),
        lr=lr,
        momentum=0.95,
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"单 GPU 模式训练: {epochs} 轮")

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
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    print("训练完成!")


def main():
    """主函数"""
    print("=" * 50)
    print("FSDP2 分布式训练示例")
    print("=" * 50)

    torch.manual_seed(42)

    is_distributed, rank, world_size = setup_distributed()

    if is_distributed:
        torch.cuda.set_device(rank)

    config = {
        "vocab_size": 1000,
        "max_seq_len": 32,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 512,
    }

    if rank == 0:
        print(f"\n配置: {config}")
        print(f"分布式: {is_distributed}, World size: {world_size}")

    train_loader = create_dummy_data(
        n_samples=200,
        seq_len=config["max_seq_len"],
        vocab_size=config["vocab_size"],
    )

    if is_distributed:
        model = SimpleTransformer(**config).cuda()
        model = apply_fsdp2(model)

        train_fsdp(
            model=model,
            train_loader=train_loader,
            is_distributed=is_distributed,
            rank=rank,
            world_size=world_size,
            epochs=3,
        )
    else:
        print("\n未检测到多 GPU 环境，使用单 GPU 模式")
        model = SimpleTransformer(**config)

        train_single_gpu(
            model=model,
            train_loader=train_loader,
            epochs=3,
        )

    cleanup_distributed(is_distributed)

    print("\n" + "=" * 50)
    print("示例完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
