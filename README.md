# FSDP2 Utils

通用 PyTorch FSDP2 工具库 - 提供 FSDP2 下的通用操作和优化器抽象。

## 概述

这是一个通用的 FSDP2 工具库，不仅包含 Muon 优化器，还提供：
- DTensor 操作工具
- 分布式通信原语
- FSDP 模块管理
- 优化器抽象基类（方便移植其他优化器到 FSDP2）

### 核心特性

- **通用 FSDP2 工具**：DTensor 操作、梯度收集、更新分片
- **Newton-Schulz 正交化**：保持权重矩阵正交，确保训练稳定
- **FSDP2 集成**：支持分片参数处理和梯度全收集
- **优化器基类**：轻松实现自定义 FSDP2 兼容优化器
- **Mac 原型验证**：支持 CPU + Gloo 后端在 Mac 上做分布式逻辑验证
- **纯 PyTorch 实现**：无 torch.compile，无自定义 CUDA 扩展

## 安装

### 环境要求

- Python 3.12+
- PyTorch 2.9.0 - 2.10.0

### 使用 uv（推荐）

本项目推荐使用 [uv](https://github.com/astral-sh/uv) 作为包管理器，配置清华 PyPI 镜像为主源，华为云镜像为辅源。

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目
git clone https://github.com/hanlinxuy/muon_fsdp.git
cd muon_fsdp

# 使用 uv 创建虚拟环境并安装（自动使用清华+华为镜像）
uv venv
uv pip install -e ".[dev]"

# 或使用 uv 同步依赖
uv pip sync requirements.txt
```

### 使用 pip

```bash
# 从源码安装
git clone https://github.com/hanlinxuy/muon_fsdp.git
cd muon_fsdp
pip install -e ".[dev]"

# 或仅安装核心包
pip install -e .
```

### 配置镜像源（可选）

如果使用 pip，建议配置国内镜像源加速下载：

```bash
# 清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 或华为源
pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
```

## 快速开始

### 单 GPU 训练

```python
import torch
import torch.nn as nn
from muon_fsdp import MuonOptimizer

# 创建模型
model = nn.Sequential(
    nn.Linear(512, 2048),
    nn.GELU(),
    nn.Linear(2048, 512),
)

# 创建优化器
optimizer = MuonOptimizer(
    model.parameters(),
    lr=0.02,
    momentum=0.95,
    weight_decay=0.01,
)

# 训练循环
for input, target in dataloader:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

### FSDP2 分布式训练

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from muon_fsdp import FSDPMuonOptimizer

# 创建模型
model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8),
    num_layers=6,
)

# 使用 FSDP2 包装
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)

# 创建优化器（自动处理分片参数）
optimizer = FSDPMuonOptimizer(
    model=model,
    lr=0.02,
    weight_decay=0.01,
    momentum=0.95,
)

# 训练循环
for input, target in dataloader:
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 混合 Muon + AdamW

为获得最佳性能，对 2D 权重矩阵使用 Muon，对 1D 参数（偏置、层归一化）使用 AdamW：

```python
from examples.mixed_adamw import create_hybrid_optimizer

# 自动分离 2D 和 1D 参数
optimizer = create_hybrid_optimizer(
    model=model,
    muon_lr=0.02,        # 用于权重矩阵
    adamw_lr=1e-3,       # 用于偏置和层归一化
    adamw_weight_decay=0.01,
)

# 训练循环相同
```

## 架构

### 核心组件

```
muon_fsdp/
├── optimizer.py          # 单 GPU 训练用 MuonOptimizer
├── fsdp.py              # 分布式训练用 FSDPMuonOptimizer
├── distributed.py        # 通信工具（all-gather, scatter）
└── utils.py             # Newton-Schulz 迭代
```

### 工作原理

1. **Newton-Schulz 迭代**：对权重矩阵进行正交化，保持稳定的奇异值分布
2. **动量更新**：使用可配置动量累积梯度
3. **学习率缩放**：根据矩阵维度自动缩放学习率
4. **FSDP2 集成**：
   - 全收集梯度用于 Newton-Schulz 计算
   - 在完整矩阵上计算正交化更新
   - 将更新分片回分片参数

## API 参考

### MuonOptimizer

```python
from muon_fsdp import MuonOptimizer

optimizer = MuonOptimizer(
    params,              # 可迭代的参数
    lr=0.02,            # 学习率
    momentum=0.95,       # 动量系数
    weight_decay=0.0,    # 权重衰减（L2 惩罚）
    nesterov=False,     # 启用 Nesterov 动量
    ns_steps=5,         # Newton-Schulz 迭代次数
)
```

### FSDPMuonOptimizer

```python
from muon_fsdp import FSDPMuonOptimizer

optimizer = FSDPMuonOptimizer(
    model,               # FSDP 包装的模型
    params=None,         # 可选的参数列表
    lr=0.02,
    weight_decay=0.01,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,
    ns_stepsize=1.0,
    beta2=0.99,         # 二阶矩系数
    eps=1e-8,           # 数值稳定性
    gradient_accumulation_steps=1,
)
```

## 示例

| 示例 | 说明 |
|------|------|
| `examples/train_gpt.py` | GPT 风格语言模型训练 |
| `examples/mixed_adamw.py` | 混合 Muon + AdamW 优化器 |

运行示例：

```bash
# 单 GPU
python examples/train_gpt.py --model gpt2 --epochs 3

# FSDP2 分布式（4 GPU）
torchrun --nproc_per_node=4 examples/train_gpt.py --model gpt2 --fsdp --epochs 3
```

## 超参数

### 推荐设置

| 模型规模 | 学习率 | 动量 | 权重衰减 |
|----------|--------|------|----------|
| 小型 (<1B) | 0.02 | 0.95 | 0.01 |
| 中型 (1-10B) | 0.02 | 0.95 | 0.01 |
| 大型 (>10B) | 0.015 | 0.95 | 0.01 |

### Newton-Schulz 步数

- **默认**：5 次迭代
- **大型模型**：5-7 次迭代，收敛更好
- **小型模型**：3-5 次迭代，效率更高

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行并查看覆盖率
pytest tests/ --cov=muon_fsdp

# 运行特定测试文件
pytest tests/test_optimizer.py -v
```

## 环境要求

- Python 3.10+
- PyTorch 2.10+
- CUDA 11.8+ / ROCm 5.4+（GPU 训练需要）
- torcheval（可选，用于指标计算）

## 性能优化建议

1. **梯度累积**：使用 `gradient_accumulation_steps` 获得更大的有效批量大小
2. **混合精度**：启用 bf16/fp16 以节省内存并加速训练
3. **梯度检查点**：对大型模型使用以减少内存使用
4. **优化器状态卸载**：不支持 - Muon 需要 GPU 计算

## 局限性

- **CPU 卸载**：不支持（Muon 需要 GPU 进行 Newton-Schulz 计算）
- **稀疏梯度**：不支持
- **张量并行**：第二阶段功能（尚未实现）

## 参考资料

- [Muon: Matrix-based Orthogonalization for Neural Networks](https://github.com/microsoft/muon)
- [PyTorch FSDP2](https://pytorch.org/docs/stable/fsdp.html)
- [Newton-Schulz Iteration for Matrix Square Root](https://en.wikipedia.org/wiki/Newton%E2%80%93Schulz_method)

## 许可证

MIT License - 详见 LICENSE 文件。
