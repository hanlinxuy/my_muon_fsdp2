#!/usr/bin/env python3
"""多进程分布式训练测试。"""

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import os
import sys

sys.path.insert(0, ".")
from muon_fsdp import FSDPMuonOptimizer
from torch.distributed.fsdp import fully_shard


def run_worker(rank, world_size):
    """Worker function for distributed training test."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 创建模型
    model = nn.Sequential(
        nn.Linear(128, 128),
        nn.GELU(),
        nn.Linear(128, 128),
    )

    # 配置 FSDP
    for layer in model:
        if isinstance(layer, nn.Linear):
            fully_shard(layer)

    # 创建优化器
    optimizer = FSDPMuonOptimizer(
        model=model,
        lr=0.02,
        ns_steps=3,
    )

    # 前向传播
    x = torch.randn(4, 128)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()

    if rank == 0:
        print(f"Rank {rank}: FSDP modules = {len(optimizer.fsdp_modules)}")
        print(f"Rank {rank}: Initial loss = {loss.item():.4f}")

    # 优化步骤
    optimizer.step()
    optimizer.zero_grad()

    # 再次前向
    y2 = model(x)
    loss2 = y2.pow(2).mean()

    if rank == 0:
        print(f"Rank {rank}: After step loss = {loss2.item():.4f}")
        print(f"Rank {rank}: Loss decreased: {loss2.item() < loss.item()}")

    dist.destroy_process_group()


def main():
    world_size = 2
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
    print("\nDistributed test completed!")


if __name__ == "__main__":
    main()
