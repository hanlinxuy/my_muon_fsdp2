# 常见问题 (FAQ)

本文档收集了 Muon FSDP 优化器的常见问题及解决方案。

## 目录

- [安装问题](#安装问题)
- [常见错误和解决方案](#常见错误和解决方案)
- [性能问题](#性能问题)
- [分布式训练问题](#分布式训练问题)
- [兼容性问题](#兼容性问题)
- [其他问题](#其他问题)

---

## 安装问题

### Q: 安装时提示找不到 `torch.distributed.fsdp`

**错误信息：**
```
ImportError: cannot import name 'fully_shard' from 'torch.distributed.fsdp'
```

**解决方案：**

1. 检查 PyTorch 版本：
```bash
python -c "import torch; print(torch.__version__)"
```

2. 确保使用 PyTorch 2.10+：
```bash
pip install --upgrade torch>=2.10.0
```

3. 如果使用 conda：
```bash
conda install pytorch>=2.10.0 torchvision torchaudio -c pytorch
```

**注意：** FSDP2 API 在 PyTorch 2.10 中稳定，旧版本可能不支持。

---

### Q: 安装时 `pip install -e ".[dev]"` 失败

**可能原因：**
- Python 版本过低（需要 3.10+）
- 缺少编译工具
- 网络问题

**解决方案：**

1. 检查 Python 版本：
```bash
python --version  # 需要 3.10+
```

2. 分步安装：
```bash
# 仅安装核心包
pip install -e .

# 手动安装开发依赖
pip install pytest pytest-cov torch
```

3. 使用国内镜像源：
```bash
pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### Q: CLI 命令 `muon-fsdp` 无法运行

**错误信息：**
```
command not found: muon-fsdp
```

**解决方案：**

此问题已知，`cli.py` 尚未实现。目前需要直接使用 Python API：

```python
# 直接导入使用，而不是通过 CLI
from muon_fsdp import MuonOptimizer, FSDPMuonOptimizer

optimizer = MuonOptimizer(model.parameters(), lr=0.02)
```

---

## 常见错误和解决方案

### Q: RuntimeError: Sparse gradients are not supported

**错误信息：**
```
RuntimeError: Sparse gradients are not supported by Muon optimizer
```

**原因：** Muon 优化器不支持稀疏梯度（例如嵌入层使用稀疏梯度时）。

**解决方案：**

1. 对嵌入层使用密集梯度：
```python
import torch.nn as nn

# 禁用稀疏梯度
embedding = nn.Embedding(vocab_size, d_model, sparse=False)
```

2. 使用混合优化器（Muon 用于权重，AdamW 用于嵌入）：
```python
from examples.mixed_adamw import create_hybrid_optimizer

# 自动处理稀疏梯度层
optimizer = create_hybrid_optimizer(model, muon_lr=0.02, adamw_lr=1e-3)
```

---

### Q: CUDA out of memory (OOM)

**错误信息：**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**解决方案：**

1. **减少微批次大小（micro batch size）**
```python
batch_size = 1  # 减小到 1 或 2
```

2. **启用梯度检查点（gradient checkpointing）**
```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)
```

3. **使用混合精度（bf16/fp16）**
```python
from torch.cuda.amp import autocast

with autocast(dtype=torch.bfloat16):
    output = model(input)
```

4. **增加梯度累积步数**
```python
optimizer = FSDPMuonOptimizer(
    model,
    gradient_accumulation_steps=8,  # 增加
)
```

5. **使用梯度裁剪**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### Q: 梯度爆炸或 NaN 损失

**症状：** 训练过程中损失突然变为 `nan` 或 `inf`。

**解决方案：**

1. **降低学习率**
```python
optimizer = MuonOptimizer(model.parameters(), lr=0.01)  # 从 0.02 降低
```

2. **增加权重衰减**
```python
optimizer = MuonOptimizer(model.parameters(), weight_decay=0.02)  # 从 0.01 增加
```

3. **启用梯度裁剪**
```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

4. **使用 bf16 而非 fp16**
```python
# bf16 数值稳定性更好
with autocast(dtype=torch.bfloat16):
    output = model(input)
```

5. **增加 Newton-Schulz 迭代次数**
```python
optimizer = MuonOptimizer(model.parameters(), ns_steps=7)  # 从 5 增加
```

---

### Q: 优化器状态不匹配错误

**错误信息：**
```
RuntimeError: Error(s) in loading state_dict for MuonOptimizer
```

**解决方案：**

1. 确保优化器配置一致：
```python
# 保存和加载时使用相同参数
optimizer = MuonOptimizer(
    model.parameters(),
    lr=0.02,       # 必须一致
    momentum=0.95,  # 必须一致
    ns_steps=5,    # 必须一致
)

# 保存
torch.save(optimizer.state_dict(), "optimizer.pt")

# 加载
optimizer.load_state_dict(torch.load("optimizer.pt"))
```

2. 检查模型参数顺序：
```python
# 确保加载时的模型结构与保存时完全相同
model = load_model()  # 相同的模型结构
```

---

## 性能问题

### Q: 训练速度很慢

**可能原因：** GPU 利用率低、通信开销大、数据加载慢。

**解决方案：**

1. **使用 NCCL 后端（多 GPU）**
```python
import torch.distributed as dist

dist.init_process_group(backend="nccl", init_method="env://")
```

2. **启用 FSDP2 向后预取**
```python
from torch.distributed.fsdp import BackwardPrefetch, fully_shard

fully_shard(
    layer,
    backward_prefetch=BackwardPrefetch.FULL_SHARD,
    forward_prefetch=True,
)
```

3. **优化数据加载**
```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=4,  # 增加 worker 数量
    pin_memory=True,  # 启用 pin_memory
    prefetch_factor=2,
)
```

4. **使用性能分析器**
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    # 训练步骤
    optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

### Q: 单 GPU 训练比多 GPU 还快

**可能原因：** 通信开销过大，或模型太小不适合分布式训练。

**解决方案：**

1. **增加模型规模**
```python
# 小模型（< 100M 参数）不适合 FSDP
# 增加层数、隐藏层大小、或序列长度
```

2. **减少通信频率**
```python
# 增加梯度累积步数
optimizer = FSDPMuonOptimizer(
    model,
    gradient_accumulation_steps=8,  # 增加
)
```

3. **检查网络带宽**
```bash
# 测试 GPU 间带宽
nvidia-smi nvlink --status
```

4. **对小模型使用单 GPU**
```python
# 小模型直接使用 MuonOptimizer
optimizer = MuonOptimizer(model.parameters(), lr=0.02)
```

---

### Q: 内存占用很高

**症状：** 即使减少 batch size，内存仍然不够。

**解决方案：**

1. **启用梯度检查点**
```python
from torch.utils.checkpoint import checkpoint

# 对 Transformer 块使用
def forward(self, x):
    return checkpoint(self._forward, x, use_reentrant=False)
```

2. **减少序列长度**
```python
seq_length = 512  # 从 2048 减少到 512
```

3. **使用更小的模型**
```python
# 减少隐藏层大小
hidden_size = 768  # 从 1024 减少到 768
```

4. **监控内存使用**
```python
import torch

# 峰值内存
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# 当前内存
print(f"Current memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

---

## 分布式训练问题

### Q: 分布式训练时梯度不同步

**症状：** 不同 GPU 上的模型参数不一致，或训练结果不稳定。

**解决方案：**

1. **确保正确初始化分布式环境**
```python
import torch.distributed as dist

# 所有进程使用相同的初始化方法
dist.init_process_group(
    backend="nccl",
    init_method="env://",  # 确保使用相同方法
    world_size=4,
    rank=0,
)
```

2. **使用相同的随机种子**
```python
import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 所有进程使用相同种子
set_seed(42)
```

3. **确保数据加载器正确分片**
```python
from torch.utils.data import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=True,  # 确保所有进程批次大小相同
)

dataloader = DataLoader(dataset, sampler=sampler, ...)
```

---

### Q: FSDP2 参数分片错误

**错误信息：**
```
RuntimeError: Expected fully_sharded but got ...
```

**解决方案：**

1. **正确使用 `fully_shard`**
```python
from torch.distributed.fsdp import fully_shard

# 逐层包装
for layer in model.layers:
    fully_shard(layer)

# 最后包装整个模型
fully_shard(model)
```

2. **确保使用 FSDPMuonOptimizer**
```python
# FSDP 模型必须使用 FSDPMuonOptimizer
optimizer = FSDPMuonOptimizer(model=model, lr=0.02)

# 不要使用 MuonOptimizer（用于单 GPU）
# optimizer = MuonOptimizer(model.parameters(), lr=0.02)  # 错误
```

3. **使用 `unshard_params` 上下文**
```python
with optimizer.unshard_params():
    # 参数在这里是完整的
    full_params = [p for p in model.parameters()]
    # ...
# 参数在这里自动恢复为分片状态
```

---

### Q: 分布式训练时 NCCL 错误

**错误信息：**
```
RuntimeError: NCCL error in: ...
```

**可能原因：** NCCL 版本不匹配、网络配置问题、多进程启动错误。

**解决方案：**

1. **检查 NCCL 版本**
```bash
python -c "import torch; print(torch.cuda.nccl.version())"
```

2. **使用 `torchrun` 启动**
```bash
# 正确启动方式
torchrun --nproc_per_node=4 --master_port=29500 train.py

# 不要使用 python + 手动多进程（容易出错）
```

3. **设置环境变量**
```bash
# 增加超时时间
export NCCL_TIMEOUT=3600

# 启用调试（排查问题）
export NCCL_DEBUG=INFO
```

4. **确保所有 GPU 可见**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## 兼容性问题

### Q: CPU 训练不支持

**错误信息：**
```
RuntimeError: CPU offload is not supported
```

**原因：** Newton-Schulz 迭代需要 GPU 计算，不支持 CPU。

**解决方案：**

1. **使用 GPU 训练**
```python
# 确保模型和输入在 GPU 上
model = model.cuda()
input = input.cuda()
```

2. **如果只有 CPU，考虑使用其他优化器**
```python
# CPU 训练使用 AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

---

### Q: 与其他优化器混合使用

**问题：** 如何对模型的不同部分使用不同的优化器？

**解决方案：**

```python
from examples.mixed_adamw import create_hybrid_optimizer

# 2D 权重使用 Muon，1D 参数使用 AdamW
optimizer = create_hybrid_optimizer(
    model,
    muon_lr=0.02,        # 用于 nn.Linear, nn.Conv2d
    adamw_lr=1e-3,       # 用于偏置、层归一化
    adamw_weight_decay=0.01,
)

# 训练循环不变
for batch in dataloader:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

### Q: 与 PyTorch Lightning 集成

**问题：** 如何在 PyTorch Lightning 中使用 Muon 优化器？

**解决方案：**

```python
import pytorch_lightning as pl
from muon_fsdp import MuonOptimizer, FSDPMuonOptimizer

class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def configure_optimizers(self):
        # 单 GPU
        return MuonOptimizer(
            self.model.parameters(),
            lr=0.02,
            momentum=0.95,
            weight_decay=0.01,
        )

        # FSDP（需要 FSDP 策略）
        # return FSDPMuonOptimizer(
        #     self.model,
        #     lr=0.02,
        #     weight_decay=0.01,
        # )
```

---

### Q: 与 Hugging Face Transformers 集成

**问题：** 如何在 Hugging Face 模型中使用 Muon？

**解决方案：**

```python
from transformers import AutoModelForCausalLM
from muon_fsdp import MuonOptimizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 创建优化器
optimizer = MuonOptimizer(
    model.parameters(),
    lr=0.02,
    momentum=0.95,
    weight_decay=0.01,
)

# 正常训练
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
```

**注意：** Hugging Face 模型的嵌入层默认使用稀疏梯度，需要禁用或使用混合优化器。

---

## 其他问题

### Q: 如何选择超参数？

**推荐设置：**

| 参数 | 小型模型 (<1B) | 中型模型 (1-10B) | 大型模型 (>10B) |
|------|----------------|------------------|-----------------|
| 学习率 | 0.02 | 0.02 | 0.015 |
| 动量 | 0.95 | 0.95 | 0.95 |
| 权重衰减 | 0.01 | 0.01 | 0.01 |
| NS 步数 | 3-5 | 5 | 5-7 |

**调整建议：**

- **损失不稳定** → 降低学习率、增加权重衰减
- **收敛慢** → 增加学习率、减少动量
- **数值不稳定（NaN）** → 使用 bf16、增加梯度裁剪

---

### Q: 如何监控训练？

**代码示例：**

```python
# 监控损失
loss_history = []
ema_loss = None

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()
    optimizer.step()

    # EMA 平滑
    ema_loss = 0.1 * loss.item() + 0.9 * ema_loss if ema_loss else loss.item()
    loss_history.append(ema_loss)

    if step % 100 == 0:
        print(f"Step {step}, Loss: {ema_loss:.4f}")

# 监控内存
import torch
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

---

### Q: 如何调试训练？

**调试技巧：**

1. **小模型测试**
```python
# 先用小模型验证流程
model = nn.Linear(10, 10)  # 极小模型
```

2. **单步测试**
```python
# 只运行一个批次
for batch in dataloader:
    optimizer.step()
    break  # 只运行一步
```

3. **打印中间值**
```python
# 打印梯度统计
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean={param.grad.mean():.4f}, std={param.grad.std():.4f}")
```

4. **使用断点**
```python
import pdb; pdb.set_trace()  # 添加断点
```

---

### Q: 代码示例在哪里？

**可用示例：**

| 文件 | 说明 |
|------|------|
| `examples/train_gpt.py` | GPT 风格语言模型训练 |
| `examples/mixed_adamw.py` | 混合 Muon + AdamW 优化器 |

**运行示例：**

```bash
# 单 GPU
python examples/train_gpt.py --model gpt2 --epochs 3

# FSDP2 分布式（4 GPU）
torchrun --nproc_per_node=4 examples/train_gpt.py --model gpt2 --fsdp --epochs 3
```

---

### Q: 如何报告 Bug？

**报告步骤：**

1. **收集环境信息**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sys; print(f'Python: {sys.version}')"
nvidia-smi
```

2. **创建最小复现代码**
```python
# 最简复现脚本
import torch
import torch.nn as nn
from muon_fsdp import MuonOptimizer

# ... 复现代码
```

3. **在 GitHub Issues 报告**
   - 标题：简洁描述问题
   - 正文：包含环境信息、复现代码、错误信息、期望行为

---

### Q: 有哪些已知的限制？

**当前限制：**

1. **CPU 训练不支持** - Newton-Schulz 需要 GPU
2. **稀疏梯度不支持** - 嵌入层需要使用密集梯度或混合优化器
3. **张量并行未实现** - 仅支持 FSDP2 数据并行
4. **优化器状态卸载不支持** - 必须在 GPU 上

**未来计划：**

- 支持张量并行
- 支持 CPU offload（部分功能）
- 更多优化算法

---

## 获取帮助

- **文档**：查看 `docs/api.md` 和 `docs/performance.md`
- **示例**：查看 `examples/` 目录
- **GitHub Issues**：报告 Bug 或提出功能请求
- **测试**：运行 `pytest tests/` 验证安装

---

## 参考资源

- [Muon 论文](https://arxiv.org/abs/2307.00000)
- [PyTorch FSDP2 文档](https://pytorch.org/docs/stable/fsdp.html)
- [Newton-Schulz 迭代](https://en.wikipedia.org/wiki/Newton%E2%80%93Schulz_method)
- [PyTorch 性能调优指南](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
