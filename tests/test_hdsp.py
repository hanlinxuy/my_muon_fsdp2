"""HDSP optimizer tests.

测试 HDSP (Hybrid Data Sharding Parallel) 优化器的功能。
"""

import pytest
import torch
import torch.nn as nn

from tests.mocks.hdsp_mock import (
    HSDPMockContext,
    MockDeviceMesh,
    MockHSDPModule,
    create_mock_hsdp_config,
    mock_create_device_mesh_2d,
    mock_gather_grads_group,
)


class TestDeviceMeshCreation:
    """测试 DeviceMesh 创建"""

    def test_create_2d_mesh(self):
        mesh = mock_create_device_mesh_2d(dp_size=2, fsdp_size=2)
        assert mesh is not None
        assert mesh.mesh_dim_names == ("dp_replicate", "fsdp_shard")

    def test_mesh_dimensions(self):
        mesh = mock_create_device_mesh_2d(dp_size=2, fsdp_size=4)
        assert len(mesh.mesh) == 2
        assert len(mesh.mesh[0]) == 4


class TestHSDPConfig:
    """测试 HSDP 配置"""

    def test_create_config(self):
        config = create_mock_hsdp_config(dp_size=2, fsdp_size=2)
        assert config["dp_size"] == 2
        assert config["fsdp_size"] == 2
        assert config["device_mesh"] is not None

    def test_config_groups(self):
        config = create_mock_hsdp_config(dp_size=2, fsdp_size=2)
        assert config["dp_group"] is not None
        assert config["fsdp_group"] is not None


class TestGradientGathering:
    """测试梯度聚合"""

    def test_gather_grads_group_single(self):
        grads = [torch.randn(10, 10)]
        result = mock_gather_grads_group(grads, None)
        assert len(result) == 1

    def test_gather_grads_multiple(self):
        grads = [torch.randn(10, 10), torch.randn(20, 30)]
        result = mock_gather_grads_group(grads, None)
        assert len(result) == 2


class TestHSDPMockModule:
    """测试 Mock HSDP Module"""

    def test_create_mock_module(self):
        model = nn.Linear(10, 10)
        mesh = mock_create_device_mesh_2d(2, 2)
        hsdp_module = MockHSDPModule(model, mesh)

        assert hsdp_module._module is not None
        assert hsdp_module._mesh is mesh

    def test_unshard_returns_context_manager(self):
        model = nn.Linear(10, 10)
        hsdp_module = MockHSDPModule(model, None)

        handle = hsdp_module.unshard()
        assert hasattr(handle, "__enter__")
        assert hasattr(handle, "__exit__")


class TestHSDPMockContext:
    """测试 HSDP Mock 上下文管理器"""

    def test_context_creation(self):
        with HSDPMockContext() as ctx:
            pass

    def test_world_size_mock(self):
        with HSDPMockContext(world_size=4) as ctx:
            import torch.distributed as dist

            assert dist.get_world_size() == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestHDSPOptimizerGPU:
    """需要 GPU 的 HDSP 优化器测试"""

    def test_hdsp_optimizer_init(self):
        from muon_fsdp.hdsp import HDSPMuonOptimizer

        model = nn.Linear(64, 64).cuda()
        optimizer = HDSPMuonOptimizer(
            model,
            lr=0.02,
            momentum=0.95,
        )

        assert optimizer is not None
        assert optimizer._step_count == 0

    def test_hdsp_optimizer_step(self):
        from muon_fsdp.hdsp import HDSPMuonOptimizer

        model = nn.Linear(64, 64).cuda()
        optimizer = HDSPMuonOptimizer(model, lr=0.02)

        x = torch.randn(8, 64).cuda()
        y = model(x)
        loss = y.sum()
        loss.backward()

        optimizer.step()

        assert optimizer._step_count == 1


class TestHSDPOptimizerBasic:
    """HDSP 优化器基础测试（CPU）"""

    def test_hdsp_optimizer_creation(self):
        from muon_fsdp.hdsp import HDSPMuonOptimizer

        model = nn.Linear(32, 32)
        optimizer = HDSPMuonOptimizer(
            model,
            lr=0.02,
            momentum=0.95,
            weight_decay=0.01,
        )

        assert optimizer is not None
        assert optimizer._step_count == 0
        assert len(optimizer.param_groups) == 1

    def test_hdsp_optimizer_state_initialized(self):
        from muon_fsdp.hdsp import HDSPMuonOptimizer

        model = nn.Linear(32, 32)
        optimizer = HDSPMuonOptimizer(model, lr=0.02)

        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state[p]
                assert "momentum_buffer" in state
                assert "second_moment" in state

    def test_hdsp_optimizer_step(self):
        from muon_fsdp.hdsp import HDSPMuonOptimizer

        model = nn.Linear(32, 32)
        optimizer = HDSPMuonOptimizer(model, lr=0.02)

        x = torch.randn(8, 32)
        y = model(x)
        loss = y.sum()
        loss.backward()

        initial_weight = model.weight.clone()
        optimizer.step()

        assert not torch.allclose(model.weight, initial_weight)
        assert optimizer._step_count == 1

    def test_hdsp_optimizer_zero_grad(self):
        from muon_fsdp.hdsp import HDSPMuonOptimizer

        model = nn.Linear(32, 32)
        optimizer = HDSPMuonOptimizer(model, lr=0.02)

        x = torch.randn(8, 32)
        y = model(x)
        loss = y.sum()
        loss.backward()

        assert model.weight.grad is not None
        optimizer.zero_grad()
        assert model.weight.grad is None

    def test_hdsp_optimizer_state_dict(self):
        from muon_fsdp.hdsp import HDSPMuonOptimizer

        model = nn.Linear(32, 32)
        optimizer = HDSPMuonOptimizer(model, lr=0.02)

        x = torch.randn(8, 32)
        for _ in range(3):
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()

        state_dict = optimizer.state_dict()
        assert "step_count" in state_dict
        assert state_dict["step_count"] == 3

    def test_hdsp_optimizer_load_state_dict(self):
        from muon_fsdp.hdsp import HDSPMuonOptimizer

        model = nn.Linear(32, 32)
        optimizer = HDSPMuonOptimizer(model, lr=0.02)

        for _ in range(3):
            y = model(torch.randn(8, 32))
            y.sum().backward()
            optimizer.step()

        state_dict = optimizer.state_dict()

        model2 = nn.Linear(32, 32)
        optimizer2 = HDSPMuonOptimizer(model2, lr=0.02)
        optimizer2.load_state_dict(state_dict)

        assert optimizer2._step_count == 3


class TestHSDPConfigClass:
    """测试 HSDPConfig 类"""

    def test_config_creation(self):
        from muon_fsdp.hdsp import HSDPConfig

        config = HSDPConfig(
            dp_size=2,
            fsdp_size=4,
            lr=0.02,
            momentum=0.95,
        )

        assert config.dp_size == 2
        assert config.fsdp_size == 4
        assert config.lr == 0.02
        assert config.momentum == 0.95

    def test_config_to_dict(self):
        from muon_fsdp.hdsp import HSDPConfig

        config = HSDPConfig(dp_size=2, fsdp_size=4, lr=0.02)
        d = config.to_dict()

        assert d["dp_size"] == 2
        assert d["fsdp_size"] == 4
        assert d["lr"] == 0.02


class TestHSDPWithMockContext:
    """使用 Mock 上下文的集成测试"""

    def test_optimizer_with_mock_context(self):
        from muon_fsdp.hdsp import HDSPMuonOptimizer

        with HSDPMockContext():
            model = nn.Linear(32, 32)
            optimizer = HDSPMuonOptimizer(
                model,
                lr=0.02,
                momentum=0.95,
            )

            assert optimizer is not None

            x = torch.randn(8, 32)
            y = model(x)
            loss = y.sum()
            loss.backward()

            optimizer.step()

            assert optimizer._step_count == 1


class TestHSDPDifferentParamShapes:
    """测试不同参数形状"""

    def test_2d_weight_matrix(self):
        from muon_fsdp.hdsp import HDSPMuonOptimizer

        model = nn.Linear(64, 128)
        optimizer = HDSPMuonOptimizer(model, lr=0.02)

        x = torch.randn(8, 64)
        y = model(x)
        loss = y.sum()
        loss.backward()

        optimizer.step()

        weight = model.weight.data
        gram = weight @ weight.T
        identity = torch.eye(weight.shape[0])
        error = (gram - identity).norm()

        assert error < 1.0

    def test_1d_bias(self):
        from muon_fsdp.hdsp import HDSPMuonOptimizer

        model = nn.Linear(64, 128)
        optimizer = HDSPMuonOptimizer(model, lr=0.02)

        x = torch.randn(8, 64)
        y = model(x)
        loss = y.sum()
        loss.backward()

        initial_bias = model.bias.clone()
        optimizer.step()

        assert not torch.allclose(model.bias, initial_bias)


class TestHSDPMomentum:
    """测试动量功能"""

    def test_momentum_accumulation(self):
        from muon_fsdp.hdsp import HDSPMuonOptimizer

        model = nn.Linear(32, 32)
        optimizer = HDSPMuonOptimizer(model, lr=0.02, momentum=0.95)

        for _ in range(5):
            x = torch.randn(8, 32)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()

        assert optimizer._step_count == 5


class TestHSDPWeightDecay:
    """测试权重衰减"""

    def test_weight_decay_applied(self):
        from muon_fsdp.hdsp import HDSPMuonOptimizer

        model = nn.Linear(32, 32)
        optimizer = HDSPMuonOptimizer(model, lr=0.02, weight_decay=0.1)

        initial_norm = model.weight.norm()

        for _ in range(3):
            x = torch.randn(8, 32)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()

        final_norm = model.weight.norm()
        assert final_norm < initial_norm
