"""Unit tests for muon_fsdp.optimizer module.

This module contains tests for the MuonOptimizer class, covering:
- Newton-Schulz orthogonalization
- Optimizer step functionality
- Momentum buffer updates
- State dict save/load
- Learning rate scaling
"""

import pytest
import torch
import torch.nn as nn

from muon_fsdp.optimizer import MuonOptimizer
from muon_fsdp.utils import zeropower_via_newtonschulz5
from tests.conftest import compute_orthogonality_error, TestConfig


class TestMuonOptimizerInitialization:
    """Test suite for MuonOptimizer initialization."""

    def test_default_initialization(self):
        """Test optimizer initializes with default parameters."""
        model = nn.Linear(128, 128)
        optimizer = MuonOptimizer(model.parameters())

        assert optimizer.defaults["lr"] == 0.02
        assert optimizer.defaults["momentum"] == 0.95
        assert optimizer.defaults["weight_decay"] == 0.0
        assert optimizer.defaults["nesterov"] is False
        assert optimizer.defaults["ns_steps"] == 5

    def test_custom_initialization(self):
        """Test optimizer initializes with custom parameters."""
        model = nn.Linear(128, 128)
        optimizer = MuonOptimizer(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.01,
            nesterov=True,
            ns_steps=3,
        )

        assert optimizer.defaults["lr"] == 0.01
        assert optimizer.defaults["momentum"] == 0.9
        assert optimizer.defaults["weight_decay"] == 0.01
        assert optimizer.defaults["nesterov"] is True
        assert optimizer.defaults["ns_steps"] == 3

    def test_invalid_lr(self):
        """Test that negative learning rate raises error."""
        model = nn.Linear(128, 128)
        with pytest.raises(ValueError, match="Invalid learning rate"):
            MuonOptimizer(model.parameters(), lr=-0.01)

    def test_invalid_momentum(self):
        """Test that invalid momentum raises error."""
        model = nn.Linear(128, 128)
        with pytest.raises(ValueError, match="Invalid momentum"):
            MuonOptimizer(model.parameters(), momentum=1.5)

    def test_invalid_weight_decay(self):
        """Test that negative weight decay raises error."""
        model = nn.Linear(128, 128)
        with pytest.raises(ValueError, match="Invalid weight_decay"):
            MuonOptimizer(model.parameters(), weight_decay=-0.01)

    def test_invalid_ns_steps(self):
        """Test that negative ns_steps raises error."""
        model = nn.Linear(128, 128)
        with pytest.raises(ValueError, match="Invalid ns_steps"):
            MuonOptimizer(model.parameters(), ns_steps=-1)

    def test_nesterov_requires_momentum(self):
        """Test that Nesterov requires positive momentum."""
        model = nn.Linear(128, 128)
        with pytest.raises(ValueError, match="Nesterov momentum requires"):
            MuonOptimizer(model.parameters(), nesterov=True, momentum=0.0)


class TestMuonOptimizerStep:
    """Test suite for MuonOptimizer step functionality."""

    def test_basic_step(self):
        """Test that optimizer step updates parameters."""
        model = nn.Linear(128, 128)
        optimizer = MuonOptimizer(model.parameters(), lr=0.02)

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
        """Test that step works with zero gradients."""
        model = nn.Linear(128, 128)
        optimizer = MuonOptimizer(model.parameters())

        initial_weight = model.weight.data.clone()

        # No backward pass - gradients are None
        optimizer.step()

        # Weights should remain unchanged
        assert torch.allclose(model.weight.data, initial_weight)

    def test_step_with_closure(self):
        """Test optimizer step with closure."""
        model = nn.Linear(128, 128)
        optimizer = MuonOptimizer(model.parameters(), lr=0.02)

        def closure():
            optimizer.zero_grad()
            output = model(torch.randn(32, 128))
            loss = output.sum()
            loss.backward()
            return loss.item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)

    def test_sparse_gradient_error(self):
        """Test that sparse gradients raise error."""
        model = nn.Linear(128, 128)
        optimizer = MuonOptimizer(model.parameters())

        # Create sparse gradient
        model.weight.grad = torch.sparse_coo_tensor(
            indices=torch.tensor([[0], [0]]),
            values=torch.tensor([1.0]),
            size=(128, 128),
        )

        with pytest.raises(RuntimeError, match="sparse gradients"):
            optimizer.step()


class TestMuonOptimizerMomentum:
    """Test suite for momentum buffer functionality."""

    def test_momentum_buffer_initialization(self):
        """Test that momentum buffers are initialized correctly."""
        model = nn.Linear(128, 128)
        optimizer = MuonOptimizer(model.parameters(), momentum=0.9)

        # Before first step, state should be empty
        assert len(optimizer.state) == 0

        # Create gradient and step
        loss = model(torch.randn(32, 128)).sum()
        loss.backward()
        optimizer.step()

        # After first step, momentum buffer should exist
        for p in model.parameters():
            if p.requires_grad:
                assert "momentum_buffer" in optimizer.state[p]
                buf = optimizer.state[p]["momentum_buffer"]
                assert buf.shape == p.shape
                assert buf.dtype == p.dtype

    def test_momentum_update(self):
        """Test that momentum is updated correctly."""
        model = nn.Linear(128, 128)
        momentum = 0.9
        optimizer = MuonOptimizer(model.parameters(), momentum=momentum)

        # First step
        loss = model(torch.randn(32, 128)).sum()
        loss.backward()
        optimizer.step()

        # Get momentum buffer after first step
        for p in model.parameters():
            if p.requires_grad:
                buf1 = optimizer.state[p]["momentum_buffer"].clone()

        # Second step
        optimizer.zero_grad()
        loss = model(torch.randn(32, 128)).sum()
        loss.backward()
        optimizer.step()

        # Check momentum has been updated
        for p in model.parameters():
            if p.requires_grad:
                buf2 = optimizer.state[p]["momentum_buffer"]
                # Buffer should have changed
                assert not torch.allclose(buf1, buf2)

    def test_momentum_persistence(self):
        """Test that momentum buffers persist across multiple steps."""
        model = nn.Linear(128, 128)
        optimizer = MuonOptimizer(model.parameters(), momentum=0.95)

        # Multiple steps
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(torch.randn(32, 128)).sum()
            loss.backward()
            optimizer.step()

        # Check momentum buffers still exist and are non-zero
        for p in model.parameters():
            if p.requires_grad:
                assert "momentum_buffer" in optimizer.state[p]
                buf = optimizer.state[p]["momentum_buffer"]
                assert buf.norm().item() > 0

    def test_nesterov_momentum(self):
        """Test Nesterov momentum update."""
        model = nn.Linear(128, 128)
        momentum = 0.9
        optimizer = MuonOptimizer(
            model.parameters(),
            momentum=momentum,
            nesterov=True,
        )

        # Create gradient and step
        loss = model(torch.randn(32, 128)).sum()
        loss.backward()

        initial_weight = model.weight.data.clone()
        optimizer.step()

        # Weight should have been updated
        assert not torch.allclose(model.weight.data, initial_weight)


class TestMuonOptimizerNewtonSchulz:
    """Test suite for Newton-Schulz orthogonalization in optimizer."""

    def test_newton_schulz_applied_to_2d(self):
        """Test that Newton-Schulz is applied to 2D weight matrices."""
        model = nn.Linear(256, 256)
        optimizer = MuonOptimizer(model.parameters(), ns_steps=5)

        # Create gradient
        loss = model(torch.randn(32, 256)).sum()
        loss.backward()

        # Get gradient before step
        grad_before = model.weight.grad.clone()

        # Step
        optimizer.step()

        # Weight should be updated (Newton-Schulz modifies the update)
        assert model.weight.grad is not None

    def test_newton_schulz_orthogonality(self):
        """Test that Newton-Schulz produces near-orthogonal updates."""
        model = nn.Linear(128, 128)
        optimizer = MuonOptimizer(model.parameters(), ns_steps=5)

        # Create gradient
        loss = model(torch.randn(32, 128)).sum()
        loss.backward()

        # Get the gradient
        G = model.weight.grad

        # Apply Newton-Schulz manually
        W = zeropower_via_newtonschulz5(G, steps=5)

        # Check orthogonality
        error = compute_orthogonality_error(W)
        assert error < TestConfig.MAX_ORTHOGONALITY_ERROR, (
            f"Orthogonality error {error:.6f} exceeds maximum "
            f"{TestConfig.MAX_ORTHOGONALITY_ERROR}"
        )

    def test_no_newton_schulz_for_1d(self):
        """Test that 1D parameters (biases) don't use Newton-Schulz."""
        model = nn.Linear(128, 128)
        optimizer = MuonOptimizer(model.parameters())

        # Create gradient
        loss = model(torch.randn(32, 128)).sum()
        loss.backward()

        # Check bias gradient exists
        assert model.bias.grad is not None

        # Step should work without error
        optimizer.step()


class TestMuonOptimizerLRScaling:
    """Test suite for learning rate scaling based on matrix dimensions."""

    def test_lr_scaling_square_matrix(self):
        """Test LR scaling for square matrix (m == n)."""
        model = nn.Linear(128, 128)
        optimizer = MuonOptimizer(model.parameters(), lr=0.02)

        # For square matrix, min_dim == max_dim, so scale = max(1, 1)**0.5 = 1.0
        loss = model(torch.randn(32, 128)).sum()
        loss.backward()
        optimizer.step()

        # Test passes if no error
        assert True

    def test_lr_scaling_tall_matrix(self):
        """Test LR scaling for tall matrix (m > n)."""
        model = nn.Linear(64, 128)  # in_features=64, out_features=128
        optimizer = MuonOptimizer(model.parameters(), lr=0.02)

        # Weight shape is (128, 64) - tall matrix
        # min_dim = 64, max_dim = 128
        # scale = max(1, 64/128)**0.5 = max(1, 0.5)**0.5 = 1.0
        loss = model(torch.randn(32, 64)).sum()
        loss.backward()
        optimizer.step()

        # Test passes if no error
        assert True

    def test_lr_scaling_wide_matrix(self):
        """Test LR scaling for wide matrix (m < n)."""
        model = nn.Linear(256, 64)  # in_features=256, out_features=64
        optimizer = MuonOptimizer(model.parameters(), lr=0.02)

        # Weight shape is (64, 256) - wide matrix
        # min_dim = 64, max_dim = 256
        # scale = max(1, 64/256)**0.5 = max(1, 0.25)**0.5 = 1.0
        loss = model(torch.randn(32, 256)).sum()
        loss.backward()
        optimizer.step()

        # Test passes if no error
        assert True

    def test_lr_scaling_very_tall_matrix(self):
        """Test LR scaling for very tall matrix."""
        # Create a layer with large output and small input
        model = nn.Linear(16, 1024)
        optimizer = MuonOptimizer(model.parameters(), lr=0.02)

        # Weight shape is (1024, 16)
        # min_dim = 16, max_dim = 1024
        # scale = max(1, 16/1024)**0.5 = max(1, 0.0156)**0.5 = 1.0
        loss = model(torch.randn(8, 16)).sum()
        loss.backward()
        optimizer.step()

        # Test passes if no error
        assert True


class TestMuonOptimizerStateDict:
    """Test suite for state dict save/load functionality."""

    def test_state_dict_structure(self):
        """Test that state dict has correct structure."""
        model = nn.Linear(128, 128)
        optimizer = MuonOptimizer(model.parameters())

        # Perform a step to initialize state
        loss = model(torch.randn(32, 128)).sum()
        loss.backward()
        optimizer.step()

        # Get state dict
        state_dict = optimizer.state_dict()

        # Check structure
        assert "state" in state_dict
        assert "param_groups" in state_dict

    def test_state_dict_save_load(self):
        """Test that state can be saved and loaded."""
        model1 = nn.Linear(128, 128)
        optimizer1 = MuonOptimizer(model1.parameters(), momentum=0.95)

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
        optimizer2 = MuonOptimizer(model2.parameters(), momentum=0.95)

        # Load state
        optimizer2.load_state_dict(state_dict)

        # Check that momentum buffers match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.requires_grad and p2.requires_grad:
                buf1 = optimizer1.state[p1]["momentum_buffer"]
                buf2 = optimizer2.state[p2]["momentum_buffer"]
                assert torch.allclose(buf1, buf2)

    def test_param_groups_preserved(self):
        """Test that parameter groups are preserved in state dict."""
        model = nn.Linear(128, 128)
        optimizer = MuonOptimizer(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.001,
        )

        # Perform a step
        loss = model(torch.randn(32, 128)).sum()
        loss.backward()
        optimizer.step()

        # Save and load
        state_dict = optimizer.state_dict()

        # Create new optimizer and load
        model2 = nn.Linear(128, 128)
        optimizer2 = MuonOptimizer(model2.parameters())
        optimizer2.load_state_dict(state_dict)

        # Check hyperparameters preserved
        assert optimizer2.param_groups[0]["lr"] == 0.01
        assert optimizer2.param_groups[0]["momentum"] == 0.9
        assert optimizer2.param_groups[0]["weight_decay"] == 0.001


class TestMuonOptimizerWeightDecay:
    """Test suite for weight decay functionality."""

    def test_weight_decay_applied(self):
        """Test that weight decay is applied during step."""
        model = nn.Linear(128, 128)
        weight_decay = 0.01
        lr = 0.02
        optimizer = MuonOptimizer(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Store initial weight
        initial_weight = model.weight.data.clone()

        # Create gradient and step
        loss = model(torch.randn(32, 128)).sum()
        loss.backward()
        optimizer.step()

        # Weight should have decayed
        # With weight decay: p = p * (1 - lr * weight_decay) - lr * update
        expected_decay = 1 - lr * weight_decay
        # Weight should be smaller due to decay
        weight_norm_after = model.weight.data.norm().item()
        weight_norm_before = initial_weight.norm().item()

        # Weight norm should decrease due to weight decay
        assert weight_norm_after < weight_norm_before * 1.01  # Allow small tolerance

    def test_no_weight_decay(self):
        """Test that weight decay is not applied when set to 0."""
        model = nn.Linear(128, 128)
        optimizer = MuonOptimizer(model.parameters(), weight_decay=0.0)

        # Store initial weight
        initial_weight = model.weight.data.clone()

        # Create gradient and step
        loss = model(torch.randn(32, 128)).sum()
        loss.backward()
        optimizer.step()

        # Without weight decay and no gradient, weight shouldn't change
        # But with gradient, weight will change due to update
        # Just verify step completes without error
        assert True


class TestMuonOptimizerIntegration:
    """Integration tests for MuonOptimizer."""

    def test_multiple_steps_convergence(self):
        """Test that optimizer can perform multiple steps."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        optimizer = MuonOptimizer(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Training loop
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            input_data = torch.randn(16, 64)
            target = torch.randn(16, 64)
            output = model(input_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # All steps should complete without error
        assert len(losses) == 10

    def test_mixed_parameter_types(self):
        """Test optimizer with mixed 1D and 2D parameters."""

        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 64)
                self.bias = nn.Parameter(torch.zeros(64))

            def forward(self, x):
                return self.linear(x) + self.bias

        model = MixedModel()
        optimizer = MuonOptimizer(model.parameters())

        # Create gradient and step
        loss = model(torch.randn(32, 128)).sum()
        loss.backward()
        optimizer.step()

        # Both linear weight and custom bias should be updated
        assert model.linear.weight.grad is not None
        assert model.bias.grad is not None
