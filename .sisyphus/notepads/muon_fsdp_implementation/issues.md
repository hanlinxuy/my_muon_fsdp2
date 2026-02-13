## [2026-02-13T05:20:00Z] Task 1: Newton-Schulz 实现错误

### 问题描述
子代理实现的 `zeropower_via_newtonschulz5` 使用了错误的 Newton-Schulz 公式。

### 当前实现（错误）
```python
# 在 utils.py:93-94
for _ in range(steps):
    X = 1.5 * X - 0.5 * X @ X.T @ X
```

### 正确的实现
根据 Muon 论文和 PyTorch 官方实现，应该使用 quintic polynomial 迭代：
```python
# 正确的 Newton-Schulz 迭代
a, b, c = (3.4445, -4.7750, 2.0315)

for _ in range(steps):
    A = X @ X.T
    B = b * A + c * A @ A
    X = a * X + B @ X
```

### 参考
- PyTorch 官方 Muon: `torch/optim/_muon.py`
- Keller Jordan 实现: `https://github.com/KellerJordan/Muon`
- 计划文档第 34 行定义了正确的公式

### 影响
这是一个**算法级错误**，会完全破坏 Muon 优化器的正确性：
- 正交化性质会严重偏离
- 优化器收敛行为会不正确
- 所有测试都会失败或产生错误结果

### 解决方案
必须修复 `utils.py` 中的 Newton-Schulz 实现，使用正确的 quintic polynomial 公式。
