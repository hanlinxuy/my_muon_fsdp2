# Muon FSDP2 优化器实现计划

## TL;DR

> **目标**: 实现支持 PyTorch FSDP2 的 Muon 优化器，解决 FSDP2 切分权重与 Muon 需要完整矩阵的矛盾
> 
> **阶段 1**: 基础 Muon + FSDP2 (NS Replication 策略)
> **阶段 2**: 添加 Spectral Sphere Optimizer (SSO) 支持
> **阶段 3**: Fused-FSDP Overlap 优化 (低优先级)
> 
> **交付物**:
> - `muon_fsdp/optimizer.py` - 核心优化器实现
> - `muon_fsdp/distributed.py` - 分布式通信工具
> - `muon_fsdp/sso.py` - Spectral Sphere Optimizer
> - `tests/` - 完整的测试套件
> - `examples/` - 使用示例
> 
> **Estimated Effort**: Large (3-4 weeks)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Core Muon → FSDP Integration → Testing → SSO

---

## Context

### 原始需求
用户需要在 PyTorch FSDP2 环境下使用 Muon 优化器。核心挑战是：
- FSDP2 将权重切分到多个 GPU
- Muon 优化器需要完整矩阵进行 Newton-Schulz 正交化
- 需要在优化器步骤中重建完整矩阵视图

### 技术调研总结

**Muon 核心算法**:
- Newton-Schulz 迭代: `X_{k+1} = a·X_k + (b·A + c·A²)·X_k`, 其中 `A = X_k @ X_k^T`
- 需要完整矩阵计算 Gram matrix
- 5 次迭代通常足够，可用 bfloat16 运行

**FSDP2 集成策略**:
1. **NS Replication**: All-gather 梯度后计算 NS (推荐，已有参考实现)
2. **All-to-All**: Microsoft Dion 使用的方法，最 scalable
3. **Fused-FSDP Overlap**: 最优但复杂，需要深入 FSDP2 内部

**SSO 扩展**:
- 在 Muon 基础上添加谱约束
- 需要 power iteration 计算奇异向量
- 纯 PyTorch 实现，无 MLX 依赖

### 决策记录

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 实现策略 | 先基础 Muon，后 SSO | 降低复杂度，确保基础功能稳定 |
| FSDP 策略 | NS Replication | 实现简单，已有验证 |
| 开发环境 | CPU PyTorch + CI | macOS 无 GPU 环境 |
| Overlap 优化 | 低优先级 | 性能优化，后续添加 |

---

## Work Objectives

### 核心目标
实现一个生产级的 Muon 优化器，支持：
1. PyTorch FSDP2 分布式训练
2. 混合精度训练 (bf16/fp16)
3. 与 AdamW 的混合使用 (embedding/layernorm 用 AdamW)
4. 完整的 checkpoint 支持

### 具体交付物

**Phase 1 - 基础 Muon + FSDP2**:
- [ ] `muon_fsdp/optimizer.py` - MuonOptimizer 类
- [ ] `muon_fsdp/distributed.py` - 分布式通信工具
- [ ] `muon_fsdp/utils.py` - Newton-Schulz 实现
- [ ] `tests/test_optimizer.py` - 单元测试
- [ ] `tests/test_distributed.py` - 分布式测试
- [ ] `examples/train_gpt.py` - GPT 训练示例

**Phase 2 - SSO 支持**:
- [ ] `muon_fsdp/sso.py` - SpectralSphereOptimizer 类
- [ ] `muon_fsdp/spectral.py` - 谱操作工具 (power iteration, SVD)
- [ ] `tests/test_sso.py` - SSO 测试
- [ ] `examples/train_sso.py` - SSO 训练示例

**Phase 3 - 高级特性** (低优先级):
- [ ] Fused-FSDP Overlap 实现
- [ ] Triton kernel 优化
- [ ] Tensor Parallel 支持

### Definition of Done

**Phase 1 完成标准**:
- [ ] 在单 GPU 上训练 small GPT (10M params) 收敛
- [ ] 在 2-4 GPU FSDP2 上训练收敛，loss 曲线与单 GPU 一致
- [ ] 内存使用与 AdamW 相当或更优
- [ ] 所有测试通过

**Phase 2 完成标准**:
- [ ] SSO 在单 GPU 上训练收敛
- [ ] SSO 在 FSDP2 上训练收敛
- [ ] 与 Muon 的对比实验显示稳定性提升

### Must Have
- PyTorch 2.10+ 兼容
- FSDP2 支持 (fully_shard API)
- Newton-Schulz 正交化
- Momentum 支持
- Learning rate scaling
- Weight decay
- Checkpoint save/load

### Must NOT Have (Guardrails)
- 不支持 FSDP CPU offload (Muon 必须在 GPU 上计算)
- 不支持稀疏梯度
- Phase 1 不实现 TP (Tensor Parallel) 支持
- 不实现 Fused-FSDP Overlap (Phase 3 再做)

---

## Verification Strategy

### 测试策略

**单元测试** (CPU 可运行):
- Newton-Schulz 迭代正确性
- 正交化性质验证
- Learning rate scaling 公式
- State dict save/load

**分布式测试** (需要 GPU):
- FSDP2 集成测试
- 多 GPU 收敛性验证
- Communication pattern 验证

**集成测试** (需要 GPU):
- Small GPT 训练 (10M params)
- 与 AdamW baseline 对比
- 内存 profiling

### Agent-Executed QA Scenarios

**Scenario 1: 单 GPU Muon 训练**
```
Tool: Bash (Python)
Preconditions: PyTorch 2.10+ installed, CPU mode acceptable
Steps:
  1. python -c "
       import torch
       from muon_fsdp import MuonOptimizer
       
       # Create simple model
       model = torch.nn.Linear(100, 100)
       
       # Create optimizer
       optim = MuonOptimizer(model.parameters(), lr=0.02)
       
       # Forward + backward
       x = torch.randn(10, 100)
       loss = model(x).sum()
       loss.backward()
       
       # Step
       optim.step()
       
       print('✓ Muon optimizer step completed')
     "
Expected Result: 无错误，optimizer step 成功
Evidence: 终端输出
```

**Scenario 2: Newton-Schulz 正交化验证**
```
Tool: Bash (Python)
Steps:
  1. python -c "
       import torch
       from muon_fsdp.utils import zeropower_via_newtonschulz5
       
       # Random matrix
       G = torch.randn(50, 100)
       
       # Orthogonalize
       O = zeropower_via_newtonschulz5(G, steps=5)
       
       # Check orthogonality: O @ O^T ≈ I
       I = torch.eye(50)
       error = torch.norm(O @ O.T - I)
       
       assert error < 0.1, f'Orthogonality error too large: {error}'
       print(f'✓ Orthogonality error: {error:.6f}')
     "
Expected Result: 正交化误差 < 0.1
Evidence: 终端输出
```

**Scenario 3: FSDP2 集成测试** (需要多 GPU)
```
Tool: Bash (torchrun)
Preconditions: 2+ GPUs available
Steps:
  1. torchrun --nproc_per_node=2 tests/test_fsdp2_integration.py
     
     # Test script performs:
     # - Initialize distributed
     # - Create FSDP2 wrapped model
     # - Train for 10 steps
     # - Verify all ranks have same loss
     
Expected Result: 所有 ranks 收敛，loss 一致
Evidence: 训练日志
```

**Scenario 4: Checkpoint 保存/加载**
```
Tool: Bash (Python)
Steps:
  1. python -c "
       import torch
       from muon_fsdp import MuonOptimizer
       
       model = torch.nn.Linear(100, 100)
       optim = MuonOptimizer(model.parameters(), lr=0.02)
       
       # Train a bit
       for _ in range(5):
           loss = model(torch.randn(10, 100)).sum()
           loss.backward()
           optim.step()
       
       # Save
       state = optim.state_dict()
       torch.save(state, '/tmp/muon_state.pt')
       
       # Load
       optim2 = MuonOptimizer(model.parameters(), lr=0.02)
       optim2.load_state_dict(torch.load('/tmp/muon_state.pt'))
       
       # Verify
       for p in model.parameters():
           assert torch.allclose(
               optim.state[p]['momentum_buffer'],
               optim2.state[p]['momentum_buffer']
           )
       
       print('✓ Checkpoint save/load works')
     "
Expected Result: Checkpoint 保存和加载成功
Evidence: 终端输出
```

---

## Execution Strategy

### Wave 1: 核心实现 (可并行)

**Task 1: 项目结构和基础工具**
- 创建 `muon_fsdp/` 包结构
- 实现 `utils.py` - Newton-Schulz 迭代
- 实现 `distributed.py` - 分布式通信抽象
- 设置 pytest 和 CI 配置

**Task 2: MuonOptimizer 核心**
- 实现基础 MuonOptimizer 类
- 支持 momentum, weight decay
- 实现 learning rate scaling
- 单 GPU 测试通过

**Task 3: FSDP2 集成**
- 实现 FSDP2 兼容的分布式逻辑
- 支持 all-gather 梯度
- 处理 DTensor 参数
- 多 GPU 测试通过

### Wave 2: 测试和文档 (依赖 Wave 1)

**Task 4: 完整测试套件**
- 单元测试覆盖所有功能
- 分布式测试 (mock 和真实)
- 集成测试 (small GPT)

**Task 5: 示例和文档**
- GPT 训练示例
- API 文档
- 性能调优指南

**Task 6: SSO 实现 (Phase 2)**
- 实现 SpectralSphereOptimizer
- Power iteration 和 bisection solver
- 与 Muon 的对比实验

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 (基础工具) | None | 2, 3 | None |
| 2 (Muon核心) | 1 | 4 | 3 |
| 3 (FSDP集成) | 1 | 4 | 2 |
| 4 (测试) | 2, 3 | 5, 6 | None |
| 5 (文档) | 4 | None | 6 |
| 6 (SSO) | 4 | None | 5 |

---

## TODOs

### Task 1: 项目结构和基础工具

**What to do**:
- [x] 创建项目目录结构
- [x] 实现 `muon_fsdp/utils.py` - Newton-Schulz 迭代
- [x] 实现 `muon_fsdp/distributed.py` - 分布式通信工具
- [x] 创建 `setup.py` / `pyproject.toml`
- [x] 设置 pytest 配置
- [x] 创建 GitHub Actions CI (CPU 测试)

**Must NOT do**:
- 不要实现完整的优化器逻辑 (留给 Task 2)
- 不要添加复杂的分布式逻辑 (留给 Task 3)

**Recommended Agent Profile**:
- **Category**: `quick`
- **Skills**: None (基础项目设置)

**Parallelization**:
- **Can Run In Parallel**: NO (基础任务)
- **Parallel Group**: Wave 1
- **Blocks**: Task 2, Task 3
- **Blocked By**: None

**References**:
- `torch/optim/_muon.py` - PyTorch 官方 Muon 实现
- `microsoft/dion/dion/newton_schulz_triton.py` - Triton kernel 参考
- `https://github.com/KellerJordan/Muon` - 原始 Muon 实现

**Acceptance Criteria**:
- [ ] `python -c "import muon_fsdp"` 成功
- [ ] `pytest tests/test_utils.py` 通过
- [ ] CI pipeline 运行成功

**Agent-Executed QA**:
```
Scenario: 项目可以正确导入
  Tool: Bash
  Steps:
    1. pip install -e .
    2. python -c "from muon_fsdp.utils import zeropower_via_newtonschulz5; print('OK')"
  Expected Result: 输出 "OK"
  Evidence: 终端输出
```

**Commit**: YES
- Message: `feat: initial project structure and Newton-Schulz implementation`
- Files: `muon_fsdp/`, `setup.py`, `tests/`, `.github/workflows/`

---

### Task 2: MuonOptimizer 核心实现

**What to do**:
- [ ] 实现 `muon_fsdp/optimizer.py` - MuonOptimizer 类
- [ ] 继承 `torch.optim.Optimizer`
- [ ] 实现 `__init__` 参数: lr, momentum, weight_decay, nesterov
- [ ] 实现 `step()` 方法
- [ ] 实现 momentum buffer 管理
- [ ] 实现 learning rate scaling: `max(1, m/n)^0.5`
- [ ] 处理不同维度参数 (2D 矩阵 vs 1D 向量)

**Must NOT do**:
- 不要添加 FSDP 特定代码 (留给 Task 3)
- 不要处理分布式通信
- 不要优化性能 (Triton kernel 留给 Phase 3)

**Recommended Agent Profile**:
- **Category**: `ultrabrain`
- **Skills**: None (纯 PyTorch)
- **Reason**: 需要正确实现 Newton-Schulz 和优化器逻辑，涉及数值稳定性

**Parallelization**:
- **Can Run In Parallel**: YES (与 Task 3 并行)
- **Parallel Group**: Wave 1
- **Blocks**: Task 4
- **Blocked By**: Task 1

**References**:
- `torch/optim/_muon.py:Muon` - PyTorch 官方实现
- `torch/optim/sgd.py` - 参考 momentum 实现
- `microsoft/dion/dion/normuon.py` - 参考参数处理

**Acceptance Criteria**:
- [ ] 单 GPU 上训练 small model 收敛
- [ ] Newton-Schulz 输出正交矩阵 (误差 < 0.1)
- [ ] Momentum buffer 正确更新
- [ ] State dict 可保存/加载

**Agent-Executed QA**:
```
Scenario: Muon 优化器单步更新
  Tool: Bash (Python)
  Steps:
    1. python -c "
         import torch
         from muon_fsdp import MuonOptimizer
         
         model = torch.nn.Linear(100, 100)
         optim = MuonOptimizer(model.parameters(), lr=0.02)
         
         # Generate gradient
         x = torch.randn(10, 100)
         loss = model(x).sum()
         loss.backward()
         
         # Get initial weight
         w0 = model.weight.clone()
         
         # Step
         optim.step()
         
         # Verify weight changed
         assert not torch.allclose(model.weight, w0)
         print('✓ Weight updated')
       "
  Expected Result: 权重更新，无错误
  Evidence: 终端输出

Scenario: Momentum buffer 持久化
  Tool: Bash (Python)
  Steps:
    1. 创建 optimizer，执行 3 个 step
    2. 保存 state_dict
    3. 创建新 optimizer，加载 state_dict
    4. 验证 momentum buffer 相同
  Expected Result: Buffer 正确恢复
  Evidence: 终端输出
```

**Commit**: YES
- Message: `feat: implement MuonOptimizer core`
- Files: `muon_fsdp/optimizer.py`, `tests/test_optimizer.py`

---

### Task 3: FSDP2 集成

**What to do**:
- [ ] 实现 `muon_fsdp/fsdp.py` - FSDP2 兼容层
- [ ] 实现梯度 all-gather 逻辑
- [ ] 处理 DTensor 参数类型
- [ ] 支持 `fully_shard()` 包装后的模型
- [ ] 实现 `unshard()` / `reshard()` 管理
- [ ] 处理混合精度 (bf16/fp16)
- [ ] 支持 gradient accumulation

**Must NOT do**:
- 不要实现 Fused-FSDP Overlap (Phase 3)
- 不要支持 FSDP1 (flat parameter) - 仅支持 FSDP2
- 不要处理 Tensor Parallel

**Recommended Agent Profile**:
- **Category**: `ultrabrain`
- **Skills**: None
- **Reason**: 需要深入理解 FSDP2 内部机制，处理 DTensor 和分布式通信

**Parallelization**:
- **Can Run In Parallel**: YES (与 Task 2 并行)
- **Parallel Group**: Wave 1
- **Blocks**: Task 4
- **Blocked By**: Task 1

**References**:
- `torch/distributed/fsdp/_fully_shard.py` - FSDP2 实现
- `torch/distributed/tensor/_dtensor.py` - DTensor API
- `microsoft/dion/dion/normuon.py` - all-to-all 实现参考
- `one-covenant/templar/src/tplr/muon/muon_fsdp2.py` - DTensor 处理

**Acceptance Criteria**:
- [ ] FSDP2 包装后的模型可以正常训练
- [ ] 2 GPU 上 loss 曲线与单 GPU 一致
- [ ] 内存使用合理 (无 OOM)
- [ ] 支持混合精度训练

**Agent-Executed QA**:
```
Scenario: FSDP2 模型训练
  Tool: Bash (torchrun)
  Preconditions: 2 GPUs available
  Steps:
    1. torchrun --nproc_per_node=2 tests/test_fsdp2_basic.py
       
       # Test script:
       # - 创建 simple FSDP2 model
       # - 使用 MuonOptimizer
       # - Train 10 steps
       # - Verify loss decreases
       
  Expected Result: 训练成功，loss 下降
  Evidence: 训练日志

Scenario: DTensor 参数处理
  Tool: Bash (Python)
  Steps:
    1. python -c "
         import torch
         from torch.distributed.fsdp import fully_shard
         from muon_fsdp import MuonOptimizer
         
         model = torch.nn.Sequential(
             torch.nn.Linear(100, 100),
             torch.nn.Linear(100, 10)
         )
         
         # Apply FSDP2
         for layer in model:
             fully_shard(layer)
         fully_shard(model)
         
         # Create optimizer
         optim = MuonOptimizer(model.parameters())
         
         # Verify parameters are DTensors
         for p in model.parameters():
             assert isinstance(p, torch.distributed.tensor.DTensor)
         
         print('✓ FSDP2 + Muon setup successful')
       "
  Expected Result: DTensor 参数正确处理
  Evidence: 终端输出
```

**Commit**: YES
- Message: `feat: add FSDP2 integration`
- Files: `muon_fsdp/fsdp.py`, `tests/test_fsdp.py`

---

### Task 4: 完整测试套件

**What to do**:
- [ ] 单元测试: Newton-Schulz, optimizer step, state dict
- [ ] Mock 分布式测试 (单进程模拟多 GPU)
- [ ] 真实分布式测试 (需要 GPU)
- [ ] 集成测试: small GPT 训练
- [ ] 性能测试: 与 AdamW 对比
- [ ] 内存 profiling

**Must NOT do**:
- 不要测试大规模模型 (留给用户)
- 不要测试 TP (Phase 3)

**Recommended Agent Profile**:
- **Category**: `unspecified-high`
- **Skills**: None
- **Reason**: 需要编写全面的测试，包括 mock 分布式

**Parallelization**:
- **Can Run In Parallel**: NO (依赖 Task 2, 3)
- **Parallel Group**: Wave 2
- **Blocks**: Task 5, 6
- **Blocked By**: Task 2, Task 3

**References**:
- `pytest` best practices
- `torch.testing.assert_close`
- `torch.distributed.launch` / `torchrun`

**Acceptance Criteria**:
- [ ] `pytest tests/` 100% 通过 (CPU 测试)
- [ ] 分布式测试在 2 GPU 上通过
- [ ] Small GPT 训练收敛

**Agent-Executed QA**:
```
Scenario: 完整测试套件运行
  Tool: Bash
  Steps:
    1. pip install pytest pytest-cov
    2. pytest tests/ -v --cov=muon_fsdp
    
  Expected Result: 所有测试通过，覆盖率 > 80%
  Evidence: pytest 输出和 coverage 报告
```

**Commit**: YES
- Message: `test: comprehensive test suite`
- Files: `tests/`

---

### Task 5: 示例和文档

**What to do**:
- [ ] `examples/train_gpt.py` - GPT 训练完整示例
- [ ] `examples/mixed_adamw.py` - Muon + AdamW 混合使用
- [ ] `README.md` - 项目介绍和快速开始
- [ ] `docs/api.md` - API 文档
- [ ] `docs/performance.md` - 性能调优指南
- [ ] `docs/faq.md` - 常见问题

**Must NOT do**:
- 不要写 Phase 3 的文档 (Fused-FSDP Overlap)
- 不要写 SSO 文档 (留给 Task 6)

**Recommended Agent Profile**:
- **Category**: `writing`
- **Skills**: None

**Parallelization**:
- **Can Run In Parallel**: YES (与 Task 6 并行)
- **Parallel Group**: Wave 2
- **Blocks**: None
- **Blocked By**: Task 4

**Acceptance Criteria**:
- [ ] 示例代码可以直接运行
- [ ] README 包含安装、使用、示例
- [ ] API 文档完整

**Agent-Executed QA**:
```
Scenario: GPT 示例运行
  Tool: Bash
  Steps:
    1. python examples/train_gpt.py --model small --steps 100
    
  Expected Result: 训练成功，loss 下降
  Evidence: 训练日志
```

**Commit**: YES
- Message: `docs: add examples and documentation`
- Files: `examples/`, `README.md`, `docs/`

---

### Task 6: SSO 实现 (Phase 2)

**What to do**:
- [ ] 实现 `muon_fsdp/spectral.py` - 谱操作工具
  - Power iteration 计算奇异值/向量
  - Bisection solver 求解 Lagrange multiplier
- [ ] 实现 `muon_fsdp/sso.py` - SpectralSphereOptimizer
  - 继承或组合 MuonOptimizer
  - 添加谱约束逻辑
  - 实现 retraction: W ← (R/σ)·W
- [ ] 实现 `tests/test_sso.py`
- [ ] 创建 `examples/train_sso.py`

**Must NOT do**:
- 不要修改 MuonOptimizer 核心逻辑 (保持独立)
- 不要添加 MLX 依赖 (纯 PyTorch)

**Recommended Agent Profile**:
- **Category**: `ultrabrain`
- **Skills**: None
- **Reason**: SSO 算法复杂，需要正确实现 power iteration 和 bisection

**Parallelization**:
- **Can Run In Parallel**: YES (与 Task 5 并行)
- **Parallel Group**: Wave 2
- **Blocks**: None
- **Blocked By**: Task 4

**References**:
- `https://github.com/Unakar/Megatron-LM/tree/SSO_main/emerging_optimizers/orthogonalized_optimizers/spectral_ball.py`
- `https://github.com/Unakar/Megatron-LM/tree/SSO_main/emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py`
- SSO paper: arXiv:2601.08393

**Acceptance Criteria**:
- [ ] SSO 在单 GPU 上训练收敛
- [ ] SSO 在 FSDP2 上训练收敛
- [ ] Power iteration 正确计算谱范数
- [ ] Bisection solver 收敛

**Agent-Executed QA**:
```
Scenario: SSO 谱约束验证
  Tool: Bash (Python)
  Steps:
    1. python -c "
         import torch
         from muon_fsdp import SpectralSphereOptimizer
         from muon_fsdp.spectral import compute_spectral_norm
         
         model = torch.nn.Linear(100, 100)
         optim = SpectralSphereOptimizer(model.parameters(), lr=0.02)
         
         # Train a few steps
         for _ in range(10):
             loss = model(torch.randn(10, 100)).sum()
             loss.backward()
             optim.step()
         
         # Check spectral norm is close to target
         sigma = compute_spectral_norm(model.weight)
         target_R = (100/100)**0.5  # sqrt(d_out/d_in)
         
         assert abs(sigma - target_R) < 0.1 * target_R
         print(f'✓ Spectral norm: {sigma:.4f}, target: {target_R:.4f}')
       "
  Expected Result: 谱范数接近目标值
  Evidence: 终端输出
```

**Commit**: YES
- Message: `feat: add Spectral Sphere Optimizer (SSO)`
- Files: `muon_fsdp/sso.py`, `muon_fsdp/spectral.py`, `tests/test_sso.py`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat: initial project structure and Newton-Schulz implementation` | `muon_fsdp/utils.py`, `setup.py` | `pytest tests/test_utils.py` |
| 2 | `feat: implement MuonOptimizer core` | `muon_fsdp/optimizer.py` | `pytest tests/test_optimizer.py` |
| 3 | `feat: add FSDP2 integration` | `muon_fsdp/fsdp.py` | `torchrun --nproc_per_node=2 tests/test_fsdp.py` |
| 4 | `test: comprehensive test suite` | `tests/` | `pytest tests/ --cov` |
| 5 | `docs: add examples and documentation` | `examples/`, `README.md` | 文档可读性检查 |
| 6 | `feat: add Spectral Sphere Optimizer (SSO)` | `muon_fsdp/sso.py`, `muon_fsdp/spectral.py` | `pytest tests/test_sso.py` |

---

## Success Criteria

### Verification Commands

```bash
# 1. 安装和基础功能
pip install -e .
python -c "from muon_fsdp import MuonOptimizer; print('✓ Import OK')"

# 2. 单元测试
pytest tests/ -v --cov=muon_fsdp

# 3. Newton-Schulz 正交化
python -c "
import torch
from muon_fsdp.utils import zeropower_via_newtonschulz5
G = torch.randn(50, 100)
O = zeropower_via_newtonschulz5(G)
error = torch.norm(O @ O.T - torch.eye(50))
assert error < 0.1, f'Error: {error}'
print(f'✓ Orthogonality error: {error:.6f}')
"

# 4. FSDP2 集成 (需要 2+ GPUs)
torchrun --nproc_per_node=2 tests/test_fsdp_integration.py

# 5. Small GPT 训练
python examples/train_gpt.py --model small --steps 1000
```

### Final Checklist

**Phase 1**:
- [ ] MuonOptimizer 单 GPU 训练收敛
- [ ] FSDP2 多 GPU 训练收敛
- [ ] 所有单元测试通过
- [ ] 分布式测试通过
- [ ] 文档完整

**Phase 2**:
- [ ] SSO 单 GPU 训练收敛
- [ ] SSO FSDP2 训练收敛
- [ ] 谱约束正确应用
- [ ] SSO 测试通过

**Phase 3** (未来):
- [ ] Fused-FSDP Overlap 实现
- [ ] Triton kernel 优化
- [ ] Tensor Parallel 支持

---

## 附录: 参考实现

### 1. PyTorch 官方 Muon
```python
# torch/optim/_muon.py
class Muon(Optimizer):
    # PyTorch 2.10+ 内置实现
    # 参考其实现细节
```

### 2. Microsoft Dion
- Repo: https://github.com/microsoft/dion
- Key files:
  - `dion/normuon.py` - FSDP2 分布式 Muon
  - `dion/newton_schulz_triton.py` - Triton kernel

### 3. SSO Megatron 实现
- Repo: https://github.com/Unakar/Megatron-LM/tree/SSO_main
- Key files:
  - `emerging_optimizers/orthogonalized_optimizers/spectral_ball.py`
  - `emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py`

### 4. FSDP2 参考
- PyTorch docs: https://pytorch.org/docs/stable/fsdp.html
- FSDP2 tutorial: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

---

## 开发环境建议

### macOS 开发 (无 GPU)

```bash
# 1. 安装 CPU PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. 安装项目
pip install -e ".[dev]"

# 3. 运行 CPU 测试
pytest tests/test_utils.py tests/test_optimizer.py -v

# 4. 使用 mock 测试分布式逻辑
pytest tests/test_distributed_mock.py -v
```

### CI 测试 (GPU)

```yaml
# .github/workflows/test-gpu.yml
# 在 GPU runner 上运行真实分布式测试
```

### 远程 GPU 测试

```bash
# 使用 ssh 连接到 GPU 服务器
ssh gpu-server

# 运行分布式测试
cd muon_fsdp
torchrun --nproc_per_node=4 tests/test_fsdp_integration.py
```

---

## 风险和缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| FSDP2 API 变化 | 中 | 高 | 关注 PyTorch 更新，使用稳定 API |
| 分布式测试困难 | 高 | 中 | 使用 mock 测试 + CI GPU runner |
| Newton-Schulz 数值不稳定 | 低 | 高 | 充分测试，参考成熟实现 |
| SSO bisection 不收敛 | 中 | 中 | 添加 fallback 逻辑 |
| 性能不如预期 | 中 | 低 | Phase 3 优化，使用 Triton |

---

## 后续工作 (Phase 3+)

1. **Fused-FSDP Overlap**: Hook 进 FSDP 前后向，重叠通信和计算
2. **Triton Kernel**: 优化 Newton-Schulz 迭代
3. **Tensor Parallel**: 支持 TP + FSDP 组合
4. **CPU Offload**: 研究 Muon 是否支持 (可能不行)
5. **大规模验证**: 在 1B+ 模型上验证

---

*Plan generated by Prometheus - Strategic Planning Consultant*
*Last updated: 2026-02-13*
