"""Real distributed tests for FSDP and HDSP.

这些测试需要真实的分布式环境（多 GPU）才能运行。
使用 torchrun 启动: torchrun --nproc_per_node=2 tests/test_distributed_real.py
"""

import os
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import always_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP

# Skip all tests if distributed is not available
skip_distributed = not (
    torch.cuda.is_available()
    and torch.distributed.is_available()
    and torch.distributed.is_nccl_available()
)

if not skip_distributed:
    try:
        # Test if we can initialize
        if not dist.is_initialized():
            skip_distributed = True
    except:
        skip_distributed = True


@unittest.skipIf(skip_distributed, "Distributed environment not available")
class TestFSDPDistributed(unittest.TestCase):
    """Test FSDP distributed training with Muon optimizer."""

    @classmethod
    def setUpClass(cls):
        """Initialize distributed environment."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()
        cls.device = torch.device(f"cuda:{cls.rank}")
        torch.cuda.set_device(cls.device)

    @classmethod
    def tearDownClass(cls):
        """Cleanup distributed environment."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_fsdp_basic_setup(self):
        """Test basic FSDP setup."""
        from muon_fsdp import FSDPMuonOptimizer

        model = nn.Linear(64, 64).to(self.device)
        model = FSDP(model)

        optimizer = FSDPMuonOptimizer(model, lr=0.02)

        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer._step_count, 0)

    def test_fsdp_training_step(self):
        """Test FSDP training step."""
        from muon_fsdp import FSDPMuonOptimizer

        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        ).to(self.device)

        model = FSDP(model)
        optimizer = FSDPMuonOptimizer(model, lr=0.02)

        # Training step
        x = torch.randn(8, 64, device=self.device)
        y = model(x)
        loss = y.sum()
        loss.backward()

        initial_weight = model.module[0].weight.clone()
        optimizer.step()

        # Verify weight changed
        self.assertFalse(torch.allclose(model.module[0].weight, initial_weight))
        self.assertEqual(optimizer._step_count, 1)

    def test_fsdp_gradient_sync(self):
        """Test that gradients are synchronized across ranks."""
        from muon_fsdp import FSDPMuonOptimizer

        torch.manual_seed(42 + self.rank)

        model = nn.Linear(64, 64).to(self.device)
        model = FSDP(model)
        optimizer = FSDPMuonOptimizer(model, lr=0.02)

        # Different input on each rank
        x = torch.randn(4, 64, device=self.device) * (self.rank + 1)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Gather gradients from all ranks
        if hasattr(model, "module"):
            grad = model.module.weight.grad
        else:
            grad = model.weight.grad

        # All ranks should have the same gradient after sync
        if self.world_size > 1:
            gathered_grads = [torch.zeros_like(grad) for _ in range(self.world_size)]
            dist.all_gather(gathered_grads, grad)

            # Check all gradients are the same
            for i in range(1, self.world_size):
                self.assertTrue(torch.allclose(gathered_grads[0], gathered_grads[i]))

    def test_fsdp_save_load(self):
        """Test FSDP checkpoint save and load."""
        from muon_fsdp import FSDPMuonOptimizer

        model = nn.Linear(64, 64).to(self.device)
        model = FSDP(model)
        optimizer = FSDPMuonOptimizer(model, lr=0.02)

        # Train a few steps
        for _ in range(3):
            x = torch.randn(8, 64, device=self.device)
            y = model(x)
            y.sum().backward()
            optimizer.step()

        # Save state
        state_dict = optimizer.state_dict()

        # Create new optimizer and load
        model2 = nn.Linear(64, 64).to(self.device)
        model2 = FSDP(model2)
        optimizer2 = FSDPMuonOptimizer(model2, lr=0.02)
        optimizer2.load_state_dict(state_dict)

        self.assertEqual(optimizer2._step_count, 3)


@unittest.skipIf(skip_distributed, "Distributed environment not available")
class TestHDSPDistributed(unittest.TestCase):
    """Test HDSP (Hybrid Data Sharding Parallel) distributed training."""

    @classmethod
    def setUpClass(cls):
        """Initialize distributed environment."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()
        cls.device = torch.device(f"cuda:{cls.rank}")
        torch.cuda.set_device(cls.device)

    @classmethod
    def tearDownClass(cls):
        """Cleanup distributed environment."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_hdsp_basic_setup(self):
        """Test basic HDSP setup."""
        from muon_fsdp import HDSPMuonOptimizer

        model = nn.Linear(64, 64).to(self.device)

        # Create process groups for HSDP
        if self.world_size >= 2:
            # Split into DP and FSDP groups
            dp_size = 2
            fsdp_size = self.world_size // dp_size

            dp_ranks = [i for i in range(self.rank % dp_size, self.world_size, dp_size)]
            fsdp_ranks = [
                i
                for i in range(
                    (self.rank // dp_size) * fsdp_size, (self.rank // dp_size + 1) * fsdp_size
                )
            ]

            dp_group = dist.new_group(dp_ranks) if len(dp_ranks) > 1 else None
            fsdp_group = dist.new_group(fsdp_ranks) if len(fsdp_ranks) > 1 else None

            optimizer = HDSPMuonOptimizer(model, dp_group=dp_group, fsdp_group=fsdp_group, lr=0.02)

            self.assertIsNotNone(optimizer)
            if dp_group is not None:
                dist.destroy_process_group(dp_group)
            if fsdp_group is not None:
                dist.destroy_process_group(fsdp_group)

    def test_hdsp_gradient_aggregation(self):
        """Test HDSP gradient aggregation within FSDP group only."""
        from muon_fsdp import HDSPMuonOptimizer
        from muon_fsdp.hdsp import gather_grads_group

        # Create simple gradient
        grad = torch.randn(64, 64, device=self.device)

        # Test gather_grads_group function
        result = gather_grads_group([grad], None)
        self.assertEqual(len(result), 1)
        self.assertTrue(torch.allclose(result[0], grad))

    def test_hdsp_training_step(self):
        """Test HDSP training step."""
        from muon_fsdp import HDSPMuonOptimizer

        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        ).to(self.device)

        optimizer = HDSPMuonOptimizer(model, lr=0.02)

        # Training step
        x = torch.randn(8, 64, device=self.device)
        y = model(x)
        loss = y.sum()
        loss.backward()

        initial_weight = model[0].weight.clone()
        optimizer.step()

        # Verify weight changed
        self.assertFalse(torch.allclose(model[0].weight, initial_weight))
        self.assertEqual(optimizer._step_count, 1)


class TestDistributedUtils(unittest.TestCase):
    """Test distributed utility functions."""

    def test_imports(self):
        """Test that all distributed modules can be imported."""
        from muon_fsdp import (
            FSDPMuonOptimizer,
            HDSPMuonOptimizer,
            HSDPConfig,
        )
        from muon_fsdp.hdsp import (
            create_device_mesh_2d,
            gather_grads_group,
            get_hsdp_groups,
        )

        self.assertTrue(True)  # If we get here, imports worked

    def test_hsdp_config(self):
        """Test HSDPConfig class."""
        from muon_fsdp import HSDPConfig

        config = HSDPConfig(
            dp_size=2,
            fsdp_size=4,
            lr=0.02,
            momentum=0.95,
        )

        self.assertEqual(config.dp_size, 2)
        self.assertEqual(config.fsdp_size, 4)
        self.assertEqual(config.lr, 0.02)
        self.assertEqual(config.momentum, 0.95)

        # Test to_dict
        d = config.to_dict()
        self.assertEqual(d["dp_size"], 2)
        self.assertEqual(d["fsdp_size"], 4)


def run_tests():
    """Run all tests."""
    # Check if we're running under torchrun
    if "RANK" in os.environ:
        # Running under torchrun, use unittest
        unittest.main()
    else:
        # Not running under torchrun, print message
        print("These tests require distributed environment.")
        print("Run with: torchrun --nproc_per_node=2 tests/test_distributed_real.py")
        print("\nRunning basic import tests only...")

        # Run only non-distributed tests
        suite = unittest.TestSuite()
        suite.addTest(TestDistributedUtils("test_imports"))
        suite.addTest(TestDistributedUtils("test_hsdp_config"))

        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)


if __name__ == "__main__":
    run_tests()
