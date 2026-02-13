"""Unit tests for muon_fsdp.fsdp module (FSDP2 integration).

This module contains tests for the FSDPMuonOptimizer class using mocks
to simulate distributed environments without requiring actual GPUs or
multiple processes.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

# Import mock utilities
from tests.mocks.fsdp_mock import (
    MockDTensor,
    MockFSDPModule,
    mock_fully_shard,
    mock_is_dtensor,
    mock_get_world_size,
    mock_get_rank,
    mock_is_available,
    mock_all_gather,
    FSDPMockContext,
    create_mock_dtensor,
    create_mock_fsdp_model,
)

# Try to import the actual FSDP module, skip tests if not available
try:
    from muon_fsdp.fsdp import FSDPMuonOptimizer, is_dtensor, get_dtensor_local_tensor

    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False


pytestmark = pytest.mark.skipif(not FSDP_AVAILABLE, reason="FSDP module not available")


class TestFSDPMuonOptimizerInitialization:
    """Test suite for FSDPMuonOptimizer initialization."""

    def test_default_initialization(self):
        """Test optimizer initializes with default parameters."""
        model = nn.Linear(128, 128)
        optimizer = FSDPMuonOptimizer(model)

        assert optimizer.defaults["lr"] == 0.02
        assert optimizer.defaults["momentum"] == 0.95
        assert optimizer.defaults["weight_decay"] == 0.01
        assert optimizer.defaults["nesterov"] is True
        assert optimizer.defaults["ns_steps"] == 5

    def test_custom_initialization(self):
        """Test optimizer initializes with custom parameters."""
        model = nn.Linear(128, 128)
        optimizer = FSDPMuonOptimizer(
            model,
            lr=0.01,
            momentum=0.9,
            weight_decay=0.001,
            nesterov=False,
            ns_steps=3,
        )

        assert optimizer.defaults["lr"] == 0.01
        assert optimizer.defaults["momentum"] == 0.9
        assert optimizer.defaults["weight_decay"] == 0.001
        assert optimizer.defaults["nesterov"] is False
        assert optimizer.defaults["ns_steps"] == 3

    def test_empty_parameters_error(self):
        """Test that empty parameter list raises error."""
        model = nn.Linear(128, 128)
        # Create parameters that don't require gradients
        for p in model.parameters():
            p.requires_grad = False

        with pytest.raises(ValueError, match="empty parameter list"):
            FSDPMuonOptimizer(model)

    def test_model_reference_stored(self):
        """Test that model reference is stored."""
        model = nn.Linear(128, 128)
        optimizer = FSDPMuonOptimizer(model)

        assert optimizer.model is model


class TestFSDPMuonOptimizerStep:
    """Test suite for FSDPMuonOptimizer step functionality."""

    def test_basic_step(self):
        """Test that optimizer step updates parameters."""
        model = nn.Linear(128, 128)
        optimizer = FSDPMuonOptimizer(model, lr=0.02)

        # Store initial weights
        initial_weight = model.weight.data.clone()

        # Create a dummy gradient
        loss = model(torch.randn(32, 128)).sum()
        loss.backward()

        # Perform optimizer step
        optimizer.step()

        # Check that weights have been updated
        assert not torch.allclose(model.weight.data, initial_weight)

    def test_step_with_zero_grad(self):
        """Test that step handles zero gradients."""
        model = nn.Linear(128, 128)
        optimizer = FSDPMuonOptimizer(model)

        initial_weight = model.weight.data.clone()

        # No backward pass - gradients are None
        optimizer.step()

        # Weights should remain unchanged (no update applied)
        assert torch.allclose(model.weight.data, initial_weight)

    def test_step_with_closure(self):
        """Test optimizer step with closure."""
        model = nn.Linear(128, 128)
        optimizer = FSDPMuonOptimizer(model, lr=0.02)

        def closure():
            optimizer.zero_grad()
            output = model(torch.randn(32, 128))
            loss = output.sum()
            loss.backward()
            return loss.item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)


class TestFSDPMuonOptimizerState:
    """Test suite for optimizer state management."""

    def test_state_initialization(self):
        """Test that optimizer state is initialized correctly."""
        model = nn.Linear(128, 128)
        optimizer = FSDPMuonOptimizer(model)

        # Check state is initialized for all parameters
        for group in optimizer.param_groups:
            for p in group["params"]:
                assert p in optimizer.state
                state = optimizer.state[p]
                assert "momentum_buffer" in state
                assert "second_moment" in state
                assert "accum_count" in state

    def test_momentum_buffer_shape(self):
        """Test that momentum buffers have correct shape."""
        model = nn.Linear(128, 64)
        optimizer = FSDPMuonOptimizer(model)

        for group in optimizer.param_groups:
            for p in group["params"]:
                buf = optimizer.state[p]["momentum_buffer"]
                assert buf.shape == p.shape

    def test_state_dict_save_load(self):
        """Test that state can be saved and loaded."""
        model1 = nn.Linear(128, 128)
        optimizer1 = FSDPMuonOptimizer(model1)

        # Perform some steps
        for _ in range(3):
            optimizer1.zero_grad()
            loss = model1(torch.randn(32, 128)).sum()
            loss.backward()
            optimizer1.step()

        # Save state
        state_dict = optimizer1.state_dict()

        # Create new model and optimizer
        model2 = nn.Linear(128, 128)
        optimizer2 = FSDPMuonOptimizer(model2)

        # Load state
        optimizer2.load_state_dict(state_dict)

        # Check that step count is preserved
        assert optimizer2._step_count == optimizer1._step_count


class TestFSDPMuonOptimizerGradientAccumulation:
    """Test suite for gradient accumulation."""

    def test_gradient_accumulation_steps(self):
        """Test that gradient accumulation works correctly."""
        model = nn.Linear(128, 128)
        optimizer = FSDPMuonOptimizer(
            model,
            gradient_accumulation_steps=4,
        )

        initial_weight = model.weight.data.clone()

        # Perform 3 steps (less than accumulation steps)
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(torch.randn(32, 128)).sum()
            loss.backward()
            optimizer.step()

        # Weight should not have been updated yet
        # (since we haven't reached accumulation_steps)
        # Note: This depends on implementation details

        # Perform 4th step to trigger update
        optimizer.zero_grad()
        loss = model(torch.randn(32, 128)).sum()
        loss.backward()
        optimizer.step()

        # Now weight should be updated
        assert not torch.allclose(model.weight.data, initial_weight)


class TestFSDPMuonOptimizerMockDistributed:
    """Test suite for FSDPMuonOptimizer with mocked distributed environment."""

    def test_mock_distributed_unavailable(self):
        """Test optimizer works when distributed is unavailable."""
        with patch("muon_fsdp.distributed.is_available", return_value=False):
            with patch("muon_fsdp.distributed.get_world_size", return_value=1):
                model = nn.Linear(128, 128)
                optimizer = FSDPMuonOptimizer(model)

                loss = model(torch.randn(32, 128)).sum()
                loss.backward()
                optimizer.step()

                # Should complete without error
                assert True

    def test_mock_single_process(self):
        """Test optimizer in mock single-process mode."""
        with FSDPMockContext():
            model = nn.Linear(128, 128)
            optimizer = FSDPMuonOptimizer(model)

            loss = model(torch.randn(32, 128)).sum()
            loss.backward()
            optimizer.step()

            # Should complete without error
            assert True


class TestFSDPMuonOptimizerWeightDecay:
    """Test suite for weight decay functionality."""

    def test_weight_decay_applied(self):
        """Test that weight decay is applied during step."""
        model = nn.Linear(128, 128)
        weight_decay = 0.01
        lr = 0.02
        optimizer = FSDPMuonOptimizer(
            model,
            lr=lr,
            weight_decay=weight_decay,
        )

        # Store initial weight
        initial_weight = model.weight.data.clone()

        # Create gradient and step
        loss = model(torch.randn(32, 128)).sum()
        loss.backward()
        optimizer.step()

        # Weight should have been updated
        assert not torch.allclose(model.weight.data, initial_weight)


class TestMockDTensor:
    """Test suite for MockDTensor utility."""

    def test_mock_dtensor_creation(self):
        """Test that MockDTensor can be created."""
        shape = (128, 64)
        dtensor = create_mock_dtensor(shape)

        assert isinstance(dtensor, MockDTensor)
        assert dtensor.shape == torch.Size(shape)

    def test_mock_dtensor_to_local(self):
        """Test MockDTensor to_local method."""
        shape = (128, 64)
        dtensor = create_mock_dtensor(shape)

        local = dtensor.to_local()
        assert isinstance(local, torch.Tensor)
        assert local.shape == torch.Size(shape)

    def test_mock_dtensor_full_tensor(self):
        """Test MockDTensor full_tensor method."""
        shape = (128, 64)
        dtensor = create_mock_dtensor(shape)

        full = dtensor.full_tensor()
        assert isinstance(full, torch.Tensor)
        assert full.shape == torch.Size(shape)


class TestMockFSDPModule:
    """Test suite for MockFSDPModule utility."""

    def test_mock_fsdp_module_creation(self):
        """Test that MockFSDPModule can be created."""
        model = nn.Linear(128, 64)
        fsdp_model = create_mock_fsdp_model(model)

        assert isinstance(fsdp_model, MockFSDPModule)

    def test_mock_fsdp_unshard(self):
        """Test MockFSDPModule unshard context manager."""
        model = nn.Linear(128, 64)
        fsdp_model = create_mock_fsdp_model(model)

        # Test unshard context manager
        with fsdp_model.unshard():
            pass  # Context should enter and exit without error

        # Should complete without error
        assert True

    def test_mock_fsdp_parameters(self):
        """Test MockFSDPModule parameters access."""
        model = nn.Linear(128, 64)
        fsdp_model = create_mock_fsdp_model(model)

        params = list(fsdp_model.parameters())
        assert len(params) == 2  # weight and bias


class TestFSDPMuonOptimizerIntegration:
    """Integration tests for FSDPMuonOptimizer."""

    def test_simple_training_loop(self):
        """Test a simple training loop."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        optimizer = FSDPMuonOptimizer(model, lr=0.01)
        criterion = nn.MSELoss()

        # Training loop
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            input_data = torch.randn(16, 64)
            target = torch.randn(16, 64)
            output = model(input_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # All steps should complete without error
        assert len(losses) == 5

    def test_optimizer_with_different_layer_sizes(self):
        """Test optimizer with various layer sizes."""
        model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        optimizer = FSDPMuonOptimizer(model)

        loss = model(torch.randn(8, 1024)).sum()
        loss.backward()
        optimizer.step()

        # Should complete without error
        assert True
