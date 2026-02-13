# Performance Tuning Guide

This guide provides optimization strategies for training large models with Muon FSDP2.

## Memory Optimization

### Mixed Precision Training

Enable bf16/fp16 for significant memory savings and faster training:

```python
from torch.cuda.amp import autocast, GradScaler

# Create scaler for gradient scaling
scaler = GradScaler()

# Training loop
for input, target in dataloader:
    optimizer.zero_grad()

    with autocast():
        output = model(input)
        loss = loss_fn(output, target)

    scaler.scale(loss).backward()

    # Gradient clipping
    scaler.unscale_(optimizer)

    scaler.step(optimizer)
    scaler.update()
```

**Recommendations:**
- **bf16**: Preferred on Ampere+ GPUs (A100, H100, RTX 3090+)
- **fp16**: Fallback for older GPUs
- **Gradient Scaling**: Required for fp16, optional for bf16

### Gradient Checkpointing

Reduce activation memory by recomputing activations during backward:

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def forward(self, x):
        return checkpoint(
            self._forward_impl,
            x,
            use_reentrant=False  # Recommended for FSDP
        )

    def _forward_impl(self, x):
        # Your forward pass here
        return x
```

Apply to transformer blocks:

```python
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.checkpoint = True
        # ... layer initialization

    def forward(self, x):
        if self.checkpoint:
            return checkpoint(
                self._forward_impl,
                x,
                use_reentrant=False
            )
        return self._forward_impl(x)
```

### Gradient Accumulation

Trade off memory for effective batch size:

```python
# Increase effective batch size without increasing memory
gradient_accumulation_steps = 8

for batch_idx, batch in enumerate(dataloader):
    loss = compute_loss(batch)
    loss = loss / gradient_accumulation_steps
    loss.backward()

    if (batch_idx + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Communication Optimization

### FSDP2 Backward Prefetching

Optimize gradient all-gather communication:

```python
from torch.distributed.fsdp import BackwardPrefetch

# Enable backward prefetching for better overlap
model = FSDP2Model(
    module,
    process_group=...,
    backward_prefetch=BackwardPrefetch.FULL_SHARD,
    forward_prefetch=True,
)
```

### Optimizer Step Overlap

Overlap gradient all-gather with computation:

```python
# FSDPMuonOptimizer handles this automatically
# Ensure you're using the optimizer correctly:
for input, target in dataloader:
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()

    # Optimizer step with gradient communication
    optimizer.step()
    optimizer.zero_grad()
```

### Communication Backend

Use NCCL for GPU-to-GPU communication:

```python
import torch.distributed as dist

# Initialize with NCCL backend
dist.init_process_group(
    backend="nccl",
    init_method="env://",
)
```

## Hyperparameter Tuning

### Learning Rate

| Model Size | Learning Rate | Warmup Steps |
|------------|--------------|--------------|
| < 1B | 0.02 | 1000 |
| 1-10B | 0.02 | 2000 |
| 10-100B | 0.015 | 3000 |
| > 100B | 0.01 | 5000 |

Learning rate schedule example:

```python
from torch.optim.lr_scheduler import LinearLR

scheduler = LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=10000,
)

for epoch in range(num_epochs):
    train()
    scheduler.step()
```

### Momentum

```python
# Default Muon momentum
MuonOptimizer(model.parameters(), momentum=0.95)

# For more stable training
MuonOptimizer(model.parameters(), momentum=0.90)

# For faster convergence (less stable)
MuonOptimizer(model.parameters(), momentum=0.98)
```

### Newton-Schulz Steps

| Model Size | NS Steps | Notes |
|------------|----------|-------|
| Small | 3-5 | Faster iteration |
| Medium | 5 | Balance |
| Large | 5-7 | Better convergence |

```python
# Faster training, slightly less stable
MuonOptimizer(model.parameters(), ns_steps=3)

# More stable convergence
MuonOptimizer(model.parameters(), ns_steps=7)
```

### Weight Decay

```python
# Standard setting
MuonOptimizer(model.parameters(), weight_decay=0.01)

# No weight decay
MuonOptimizer(model.parameters(), weight_decay=0.0)
```

## Batch Size Scaling

### Micro Batch Size

| GPU Memory | Max Micro Batch Size |
|------------|---------------------|
| 16GB | 1-2 |
| 32GB | 2-4 |
| 64GB | 4-8 |
| 80GB A100 | 8-16 |

### Global Batch Size

Effective batch size = micro_batch_size * gradient_accumulation_steps * world_size

| Target Throughput | Global Batch Size |
|-----------------|-------------------|
| Fast iterations | 64-128 |
| Stable training | 256-512 |
| Final training | 512-2048 |

### Scaling Formula

```python
# Linear scaling rule for learning rate
base_lr = 0.02
base_batch = 32

# New settings
new_batch = 256
new_lr = base_lr * (new_batch / base_batch)
```

## FSDP2 Specific Settings

### Sharding Strategy

```python
from torch.distributed.fsdp import ShardingStrategy

# Full sharding (default, recommended)
ShardingStrategy.FULL_SHARD

# Hybrid sharding (partial)
ShardingStrategy.HYBRID_SHARD

# No sharding
ShardingStrategy.NO_SHARD
```

### Mixed Precision Policy

```python
from torch.distributed.fsdp import MixedPrecision

# bf16 mixed precision
bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# fp16 mixed precision
fp16_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

# Apply to FSDP
fully_shard(model, mixed_precision=bf16_policy)
```

### CPU Offload (Not Recommended for Muon)

⚠️ **Warning**: Muon requires GPU computation for Newton-Schulz. CPU offload is not supported.

## Monitoring

### Memory Usage

```python
# Peak memory usage
torch.cuda.max_memory_allocated()

# Current memory
torch.cuda.memory_allocated()

# Memory stats
torch.cuda.memory_stats()
```

### Throughput

```python
import time

start_time = time.time()
samples_processed = 0

for epoch in range(num_epochs):
    for batch in dataloader:
        # Training step
        samples_processed += batch.size(0)

elapsed = time.time() - start_time
throughput = samples_processed / elapsed
print(f"Throughput: {throughput:.2f} samples/sec")
```

### Loss Monitoring

```python
# Log loss with moving average
ema_loss = None
alpha = 0.1

for batch in dataloader:
    loss = compute_loss(batch)

    if ema_loss is None:
        ema_loss = loss
    else:
        ema_loss = alpha * loss + (1 - alpha) * ema_loss

    print(f"Loss: {ema_loss:.4f}")
```

## Common Issues

### Out of Memory (OOM)

1. **Reduce micro batch size**
2. **Enable gradient checkpointing**
3. **Use mixed precision**
4. **Increase gradient accumulation steps**
5. **Use smaller model or sequence length**

### Training Instability

1. **Reduce learning rate**
2. **Increase weight decay**
3. **Use more Newton-Schulz steps**
4. **Enable gradient clipping**
5. **Use bf16 instead of fp16**

### Slow Training

1. **Profile with PyTorch Profiler**
2. **Enable backward prefetch**
3. **Use gradient accumulation for larger batch**
4. **Optimize data loading**
5. **Check GPU utilization**

## Profiling

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
    record_shapes=True,
) as prof:
    with record_function("model_training"):
        # Training step
        ...

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Best Practices Summary

1. **Start with bf16 mixed precision**
2. **Use gradient checkpointing for >1B models**
3. **Scale learning rate with batch size**
4. **Monitor GPU memory and adjust batch size**
5. **Use FSDP2 backward prefetch**
6. **Enable CUDA graphs for small models**
7. **Profile before optimizing**

## Hardware Recommendations

| Model Size | GPUs | GPU Memory | Notes |
|------------|------|------------|-------|
| 125M | 1 | 16GB | Single GPU |
| 1B | 4-8 | 16-32GB | FSDP2 |
| 7B | 8-16 | 40-80GB | FSDP2 + checkpointing |
| 13B | 16-32 | 80GB | FSDP2 + checkpointing |
| 30B+ | 32+ | 80GB | FSDP2 + checkpointing + ZeRO |
