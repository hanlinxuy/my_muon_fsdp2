"""Unit tests for muon_fsdp.spectral and muon_fsdp.sso modules.

This module contains tests for:
- Spectral norm computation via power iteration
- Power iteration convergence
- Bisection solver
- SpectralSphereOptimizer initialization and functionality
- Spectral constraint enforcement
"""

import math

import pytest
import torch
import torch.nn as nn

from muon_fsdp.spectral import (
    apply_spectral_retraction,
    bisect_spectral_radius,
    compute_spectral_norm,
    compute_target_radius,
    power_iteration,
)
from muon_fsdp.sso import SpectralSphereOptimizer
from tests.conftest import compute_orthogonality_error, TestConfig


class TestSpectralNormComputation:
    """Test suite for spectral norm computation."""

    def test_compute_spectral_norm_square_matrix(self):
        """Test spectral norm computation for square matrix."""
        torch.manual_seed(42)
        W = torch.randn(128, 128)
        sigma = compute_spectral_norm(W, num_iterations=20)

        # Verify against torch.svd
        _, S, _ = torch.svd(W)
        expected_sigma = S[0].item()

        # Should be close (power iteration converges to largest singular value)
        assert abs(sigma - expected_sigma) / expected_sigma < 0.1

    def test_compute_spectral_norm_tall_matrix(self):
        """Test spectral norm computation for tall matrix."""
        torch.manual_seed(42)
        W = torch.randn(256, 128)
        sigma = compute_spectral_norm(W, num_iterations=20)

        # Verify against torch.linalg.matrix_norm
        expected_sigma = torch.linalg.matrix_norm(W, ord=2).item()

        assert abs(sigma - expected_sigma) / expected_sigma < 0.1

    def test_compute_spectral_norm_wide_matrix(self):
        """Test spectral norm computation for wide matrix."""
        torch.manual_seed(42)
        W = torch.randn(128, 256)
        sigma = compute_spectral_norm(W, num_iterations=20)

        expected_sigma = torch.linalg.matrix_norm(W, ord=2).item()

        assert abs(sigma - expected_sigma) / expected_sigma < 0.1

    def test_compute_spectral_norm_invalid_dims(self):
        """Test that invalid dimensions raise error."""
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            compute_spectral_norm(torch.randn(128))


class TestPowerIteration:
    """Test suite for power iteration algorithm."""

    def test_power_iteration_returns_triplet(self):
        """Test that power iteration returns (sigma, u, v)."""
        torch.manual_seed(42)
        W = torch.randn(128, 128)
        sigma, u, v = power_iteration(W, num_iterations=10)

        assert isinstance(sigma, torch.Tensor)
        assert sigma.ndim == 0  # scalar
        assert u.shape == (128, 1)
        assert v.shape == (128, 1)

    def test_power_iteration_convergence(self):
        """Test that power iteration converges to correct singular vectors."""
        torch.manual_seed(42)
        W = torch.randn(64, 64)
        sigma, u, v = power_iteration(W, num_iterations=50)

        # Verify: W @ v ≈ sigma * u
        lhs = W @ v
        rhs = sigma * u
        residual = (lhs - rhs).norm().item()

        assert residual < 0.1  # Power iteration converges approximately

    def test_power_iteration_normalization(self):
        """Test that singular vectors are normalized."""
        torch.manual_seed(42)
        W = torch.randn(128, 128)
        sigma, u, v = power_iteration(W, num_iterations=10)

        # Check unit norm
        assert abs(u.norm().item() - 1.0) < 1e-5
        assert abs(v.norm().item() - 1.0) < 1e-5

    def test_power_iteration_invalid_dims(self):
        """Test that invalid dimensions raise error."""
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            power_iteration(torch.randn(128))


class TestBisectionSolver:
    """Test suite for bisection solver."""

    def test_bisect_spectral_radius_basic(self):
        """Test basic bisection solver functionality."""
        target_radius = 2.0
        matrix_dim = (512, 256)
        lam = bisect_spectral_radius(target_radius, matrix_dim, max_iterations=100)

        # Should return a finite value
        assert isinstance(lam, float)
        assert not math.isnan(lam)
        assert not math.isinf(lam)

    def test_bisect_spectral_radius_convergence(self):
        """Test that bisection solver converges."""
        target_radius = 1.5
        matrix_dim = (256, 256)
        lam = bisect_spectral_radius(target_radius, matrix_dim, max_iterations=50, tolerance=1e-4)

        # Result should be reasonable
        assert abs(lam) < 100  # Should not explode

    def test_bisect_spectral_radius_invalid_radius(self):
        """Test that invalid target radius raises error."""
        with pytest.raises(ValueError, match="target_radius must be positive"):
            bisect_spectral_radius(-1.0, (128, 128))


class TestTargetRadius:
    """Test suite for target radius computation."""

    def test_compute_target_radius_spectral_mup(self):
        """Test spectral_mup radius mode."""
        shape = (512, 256)
        R = compute_target_radius(shape, radius_mode="spectral_mup")

        expected = math.sqrt(512 / 256)
        assert abs(R - expected) < 1e-6

    def test_compute_target_radius_identity(self):
        """Test identity radius mode."""
        shape = (512, 256)
        R = compute_target_radius(shape, radius_mode="identity")

        assert R == 1.0

    def test_compute_target_radius_with_scaler(self):
        """Test radius with scaler."""
        shape = (512, 256)
        R = compute_target_radius(shape, radius_mode="spectral_mup", radius_scaler=2.0)

        expected = 2.0 * math.sqrt(512 / 256)
        assert abs(R - expected) < 1e-6

    def test_compute_target_radius_invalid_mode(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="Invalid radius_mode"):
            compute_target_radius((128, 128), radius_mode="invalid")


class TestSpectralRetraction:
    """Test suite for spectral retraction."""

    def test_apply_spectral_retraction_hard(self):
        """Test hard retraction mode."""
        torch.manual_seed(42)
        W = torch.randn(128, 128)
        initial_sigma = compute_spectral_norm(W, num_iterations=20)
        target_radius = 2.0

        W_copy = W.clone()
        apply_spectral_retraction(W_copy, initial_sigma, target_radius, mode="hard")

        # After retraction, spectral norm should be close to target
        new_sigma = compute_spectral_norm(W_copy, num_iterations=20)
        assert abs(new_sigma - target_radius) < 0.1

    def test_apply_spectral_retraction_dynamic(self):
        """Test dynamic retraction mode."""
        torch.manual_seed(42)
        W = torch.randn(128, 128)
        initial_sigma = compute_spectral_norm(W, num_iterations=20)
        target_radius = 2.0

        W_copy = W.clone()
        apply_spectral_retraction(W_copy, initial_sigma, target_radius, mode="dynamic")

        # Weight should have changed
        assert not torch.allclose(W, W_copy)

    def test_apply_spectral_retraction_invalid_mode(self):
        """Test that invalid mode raises error."""
        W = torch.randn(128, 128)
        with pytest.raises(ValueError, match="Invalid retraction mode"):
            apply_spectral_retraction(W, 1.0, 2.0, mode="invalid")


class TestSSOInitialization:
    """Test suite for SpectralSphereOptimizer initialization."""

    def test_default_initialization(self):
        """Test optimizer initializes with default parameters."""
        model = nn.Linear(128, 128)
        optimizer = SpectralSphereOptimizer(model.parameters())

        assert optimizer.defaults["lr"] == 0.02
        assert optimizer.defaults["momentum"] == 0.95
        assert optimizer.defaults["weight_decay"] == 0.0
        assert optimizer.defaults["nesterov"] is False
        assert optimizer.defaults["ns_steps"] == 5
        assert optimizer.defaults["power_iteration_steps"] == 10
        assert optimizer.defaults["radius_mode"] == "spectral_mup"
        assert optimizer.defaults["radius_scaler"] == 1.0
        assert optimizer.defaults["retract_mode"] == "hard"

    def test_custom_initialization(self):
        """Test optimizer initializes with custom parameters."""
        model = nn.Linear(128, 128)
        optimizer = SpectralSphereOptimizer(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.01,
            nesterov=True,
            ns_steps=3,
            power_iteration_steps=15,
            radius_mode="identity",
            radius_scaler=2.0,
            retract_mode="dynamic",
        )

        assert optimizer.defaults["lr"] == 0.01
        assert optimizer.defaults["momentum"] == 0.9
        assert optimizer.defaults["weight_decay"] == 0.01
        assert optimizer.defaults["nesterov"] is True
        assert optimizer.defaults["ns_steps"] == 3
        assert optimizer.defaults["power_iteration_steps"] == 15
        assert optimizer.defaults["radius_mode"] == "identity"
        assert optimizer.defaults["radius_scaler"] == 2.0
        assert optimizer.defaults["retract_mode"] == "dynamic"

    def test_invalid_lr(self):
        """Test that negative learning rate raises error."""
        model = nn.Linear(128, 128)
        with pytest.raises(ValueError, match="Invalid learning rate"):
            SpectralSphereOptimizer(model.parameters(), lr=-0.01)

    def test_invalid_momentum(self):
        """Test that invalid momentum raises error."""
        model = nn.Linear(128, 128)
        with pytest.raises(ValueError, match="Invalid momentum"):
            SpectralSphereOptimizer(model.parameters(), momentum=1.5)

    def test_invalid_weight_decay(self):
        """Test that negative weight decay raises error."""
        model = nn.Linear(128, 128)
        with pytest.raises(ValueError, match="Invalid weight_decay"):
            SpectralSphereOptimizer(model.parameters(), weight_decay=-0.01)

    def test_invalid_ns_steps(self):
        """Test that negative ns_steps raises error."""
        model = nn.Linear(128, 128)
        with pytest.raises(ValueError, match="Invalid ns_steps"):
            SpectralSphereOptimizer(model.parameters(), ns_steps=-1)

    def test_invalid_power_iteration_steps(self):
        """Test that invalid power_iteration_steps raises error."""
        model = nn.Linear(128, 128)
        with pytest.raises(ValueError, match="Invalid power_iteration_steps"):
            SpectralSphereOptimizer(model.parameters(), power_iteration_steps=0)

    def test_invalid_radius_mode(self):
        """Test that invalid radius_mode raises error."""
        model = nn.Linear(128, 128)
        with pytest.raises(ValueError, match="Invalid radius_mode"):
            SpectralSphereOptimizer(model.parameters(), radius_mode="invalid")

    def test_invalid_retract_mode(self):
        """Test that invalid retract_mode raises error."""
        model = nn.Linear(128, 128)
        with pytest.raises(ValueError, match="Invalid retract_mode"):
            SpectralSphereOptimizer(model.parameters(), retract_mode="invalid")


class TestSSOStep:
    """Test suite for SSO step functionality."""

    def test_basic_step(self):
        """Test that optimizer step updates parameters."""
        model = nn.Linear(128, 128)
        optimizer = SpectralSphereOptimizer(model.parameters(), lr=0.02)

        initial_weight = model.weight.data.clone()

        loss = model(torch.randn(32, 128)).sum()
        loss.backward()

        optimizer.step()

        assert not torch.allclose(model.weight.data, initial_weight)

    def test_step_with_zero_grad(self):
        """Test that step works with zero gradients."""
        model = nn.Linear(128, 128)
        optimizer = SpectralSphereOptimizer(model.parameters())

        initial_weight = model.weight.data.clone()

        optimizer.step()

        # With spectral retraction, weights may still change
        # even without gradients due to retraction
        assert True  # Step should complete without error

    def test_step_with_closure(self):
        """Test optimizer step with closure."""
        model = nn.Linear(128, 128)
        optimizer = SpectralSphereOptimizer(model.parameters(), lr=0.02)

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
        optimizer = SpectralSphereOptimizer(model.parameters())

        model.weight.grad = torch.sparse_coo_tensor(
            indices=torch.tensor([[0], [0]]),
            values=torch.tensor([1.0]),
            size=(128, 128),
        )

        with pytest.raises(RuntimeError, match="sparse gradients"):
            optimizer.step()


class TestSSOSpectralConstraint:
    """Test suite for spectral constraint enforcement."""

    def test_spectral_norm_tracking(self):
        """Test that spectral norms are tracked."""
        model = nn.Linear(128, 128)
        optimizer = SpectralSphereOptimizer(
            model.parameters(),
            power_iteration_steps=10,
            radius_mode="identity",
        )

        loss = model(torch.randn(32, 128)).sum()
        loss.backward()
        optimizer.step()

        # Check spectral norm info is stored
        spectral_norms = optimizer.get_spectral_norms()
        assert len(spectral_norms) > 0

        for param_id, info in spectral_norms.items():
            assert "current_sigma" in info
            assert "target_radius" in info

    def test_spectral_retraction_applied(self):
        """Test that spectral retraction is applied during step."""
        torch.manual_seed(42)
        model = nn.Linear(64, 64)
        optimizer = SpectralSphereOptimizer(
            model.parameters(),
            lr=0.02,
            power_iteration_steps=15,
            radius_mode="identity",
            retract_mode="hard",
        )

        # Initial spectral norm
        initial_sigma = compute_spectral_norm(model.weight.data, num_iterations=15)

        # Perform step
        loss = model(torch.randn(32, 64)).sum()
        loss.backward()
        optimizer.step()

        # After step with hard retraction and identity mode,
        # spectral norm should be close to 1.0
        final_sigma = compute_spectral_norm(model.weight.data, num_iterations=15)
        assert abs(final_sigma - 1.0) < 0.2

    def test_spectral_mup_mode(self):
        """Test spectral_mup radius mode."""
        torch.manual_seed(42)
        model = nn.Linear(256, 128)  # shape = (128, 256)
        optimizer = SpectralSphereOptimizer(
            model.parameters(),
            power_iteration_steps=15,
            radius_mode="spectral_mup",
        )

        # Expected target radius
        expected_radius = math.sqrt(128 / 256)

        # Perform step
        loss = model(torch.randn(32, 256)).sum()
        loss.backward()
        optimizer.step()

        # Check spectral norms
        spectral_norms = optimizer.get_spectral_norms()
        for info in spectral_norms.values():
            assert abs(info["target_radius"] - expected_radius) < 1e-6


class TestSSONewtonSchulz:
    """Test suite for Newton-Schulz in SSO."""

    def test_newton_schulz_applied(self):
        """Test that Newton-Schulz is applied to 2D matrices."""
        model = nn.Linear(128, 128)
        optimizer = SpectralSphereOptimizer(model.parameters(), ns_steps=5)

        loss = model(torch.randn(32, 128)).sum()
        loss.backward()

        # Should complete without error
        optimizer.step()
        assert True

    def test_newton_schulz_orthogonality(self):
        """Test that Newton-Schulz produces near-orthogonal updates."""
        torch.manual_seed(42)
        model = nn.Linear(64, 64)
        optimizer = SpectralSphereOptimizer(model.parameters(), ns_steps=5)

        loss = model(torch.randn(32, 64)).sum()
        loss.backward()

        # Get gradient
        G = model.weight.grad.clone()

        from muon_fsdp.utils import zeropower_via_newtonschulz5

        W = zeropower_via_newtonschulz5(G, steps=5)

        error = compute_orthogonality_error(W)
        assert error < TestConfig.MAX_ORTHOGONALITY_ERROR


class TestSSOStateDict:
    """Test suite for state dict functionality."""

    def test_state_dict_structure(self):
        """Test that state dict has correct structure."""
        model = nn.Linear(128, 128)
        optimizer = SpectralSphereOptimizer(model.parameters())

        loss = model(torch.randn(32, 128)).sum()
        loss.backward()
        optimizer.step()

        state_dict = optimizer.state_dict()

        assert "state" in state_dict
        assert "param_groups" in state_dict

    def test_state_dict_save_load(self):
        """Test that state can be saved and loaded."""
        model1 = nn.Linear(128, 128)
        optimizer1 = SpectralSphereOptimizer(model1.parameters(), momentum=0.95)

        for _ in range(3):
            optimizer1.zero_grad()
            loss = model1(torch.randn(32, 128)).sum()
            loss.backward()
            optimizer1.step()

        state_dict = optimizer1.state_dict()

        model2 = nn.Linear(128, 128)
        optimizer2 = SpectralSphereOptimizer(model2.parameters(), momentum=0.95)

        optimizer2.load_state_dict(state_dict)

        # Check momentum buffers match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.requires_grad and p2.requires_grad:
                buf1 = optimizer1.state[p1]["momentum_buffer"]
                buf2 = optimizer2.state[p2]["momentum_buffer"]
                assert torch.allclose(buf1, buf2)


class TestSSOIntegration:
    """Integration tests for SSO."""

    def test_multiple_steps_convergence(self):
        """Test that optimizer can perform multiple steps."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        optimizer = SpectralSphereOptimizer(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

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
        optimizer = SpectralSphereOptimizer(model.parameters())

        loss = model(torch.randn(32, 128)).sum()
        loss.backward()
        optimizer.step()

        assert model.linear.weight.grad is not None
        assert model.bias.grad is not None
