"""Mac 环境下 FSDP2 验证示例 - 使用 Gloo 后端。

这个示例展示如何在 Mac 上使用 Gloo 后端进行 FSDP2 的原型验证。
适用于调试 Sharding 策略、查看参数分片情况、测试 checkpoint 保存加载等。

注意：在 Mac 上只建议用 CPU + Gloo 做逻辑验证，不要追求速度。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mac 环境下 FSDP2 验证示例")
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        choices=["gloo"],
        help="分布式后端 (Mac 上只支持 gloo)",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="cpu",
        choices=["cpu"],
        help="设备类型 (Mac 上推荐用 cpu)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="tiny",
        choices=["tiny", "small"],
        help="模型大小",
    )
    return parser.parse_args()


def create_model(model_size: str) -> nn.Module:
    """创建测试模型。"""
    if model_size == "tiny":
        return nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )
    else:
        return nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )


def init_distributed(backend: str) -> None:
    """初始化分布式环境。"""
    if not torch.distributed.is_available():
        raise RuntimeError("PyTorch distributed 不可用")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    logger.info(f"Rank {rank}/{world_size}, Local Rank {local_rank}")

    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    logger.info(f"Rank {rank}: 分布式环境初始化成功 (backend={backend})")


def main() -> int:
    args = parse_args()

    # 初始化分布式
    init_distributed(args.backend)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # 导入 FSDP2 相关模块（放在这里避免在非分布式环境导入失败）
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
    except ImportError as e:
        logger.error(f"FSDP2 导入失败: {e}")
        torch.distributed.destroy_process_group()
        return 1

    # 创建设备网格
    logger.info(f"Rank {rank}: 创建设备网格 (device_type={args.device_type})")
    mesh = init_device_mesh(
        args.device_type,
        (world_size,),
        mesh_dim_names=("dp",),
    )

    # 创建模型
    logger.info(f"Rank {rank}: 创建模型 (size={args.model_size})")
    model = create_model(args.model_size)

    if rank == 0:
        logger.info(f"原始模型结构:\n{model}")
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"原始模型参数量: {total_params:,}")

    # 应用 FSDP2 分片
    logger.info(f"Rank {rank}: 应用 FSDP2 分片")
    sharded_model = fully_shard(
        model,
        mesh=mesh,
    )

    if rank == 0:
        logger.info(f"分片后模型结构:\n{sharded_model}")

    # 验证前向传播
    logger.info(f"Rank {rank}: 验证前向传播")
    batch_size = 32
    input_dim = 128 if args.model_size == "tiny" else 256
    input_data = torch.randn(batch_size, input_dim)
    output = sharded_model(input_data)

    logger.info(f"Rank {rank}: 前向传播成功，输出形状: {output.shape}")

    # 验证反向传播
    logger.info(f"Rank {rank}: 验证反向传播")
    loss = output.sum()
    loss.backward()

    logger.info(f"Rank {rank}: 反向传播成功")

    # 检查参数和梯度
    logger.info(f"Rank {rank}: 检查参数和梯度")
    for name, param in sharded_model.named_parameters():
        if param.requires_grad:
            logger.info(
                f"Rank {rank}: 参数 {name}: "
                f"shape={param.shape}, "
                f"grad={'存在' if param.grad is not None else 'None'}, "
                f"is_dtensor={hasattr(param, 'placements')}"
            )

    # 验证优化器
    logger.info(f"Rank {rank}: 测试优化器")
    try:
        from fsdp2_utils import MuonOptimizer

        optimizer = MuonOptimizer(
            sharded_model,
            lr=0.02,
            momentum=0.95,
        )

        optimizer.step()
        optimizer.zero_grad()

        logger.info(f"Rank {rank}: 优化器执行成功")
    except ImportError as e:
        logger.warning(f"无法导入优化器: {e}")

    # 清理
    logger.info(f"Rank {rank}: 清理")
    torch.distributed.destroy_process_group()

    if rank == 0:
        logger.info("=" * 50)
        logger.info("验证完成！")
        logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
