"""Pytest configuration and fixtures for Muon FSDP2 tests.

This module provides pytest configuration, fixtures, and utility functions
for testing the Muon FSDP2 package components.
"""

import pytest
import torch


@pytest.fixture
def random_matrix_256():
    """Provide a random 256x256 float32 matrix."""
    return torch.randn(256, 256, dtype=torch.float32)


@pytest.fixture
def random_matrix_512():
    """Provide a random 512x512 float32 matrix."""
    return torch.randn(512, 512, dtype=torch.float32)


@pytest.fixture
def random_matrix_tall():
    """Provide a tall random matrix (512x256)."""
    return torch.randn(512, 256, dtype=torch.float32)


@pytest.fixture
def random_matrix_wide():
    """Provide a wide random matrix (256x512)."""
    return torch.randn(256, 512, dtype=torch.float32)


@pytest.fixture
def random_matrix_small():
    """Provide a small random matrix (16x16) for quick tests."""
    return torch.randn(16, 16, dtype=torch.float32)


@pytest.fixture
def random_matrix_large():
    """Provide a large random matrix (1024x1024) for stress tests."""
    return torch.randn(1024, 1024, dtype=torch.float32)


@pytest.fixture
def bfloat16_matrix():
    """Provide a random bfloat16 matrix."""
    return torch.randn(256, 256, dtype=torch.bfloat16)


@pytest.fixture
def seed():
    """Provide a fixed random seed for reproducibility."""
    return 42


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)


def compute_orthogonality_error(W: torch.Tensor) -> float:
    """Compute orthogonality error: ||W @ W.T - I|| or ||W.T @ W - I||.

    For tall matrices (m > n), we check W.T @ W ≈ I (columns are orthogonal)
    For wide matrices (m < n), we check W @ W.T ≈ I (rows are orthogonal)
    For square matrices (m == n), we check W @ W.T ≈ I

    Args:
        W: Orthogonal matrix to verify.

    Returns:
        Relative orthogonality error.
    """
    device = W.device
    dtype = W.dtype

    if W.size(0) > W.size(1):
        # Tall matrix: check columns (W.T @ W)
        n = W.size(1)
        identity = torch.eye(n, device=device, dtype=dtype)
        product = W.T @ W
    else:
        # Wide or square: check rows (W @ W.T)
        n = W.size(0)
        identity = torch.eye(n, device=device, dtype=dtype)
        product = W @ W.T

    error = (product - identity).norm().item()
    return error


def compute_orthogonality_error_normalized(W: torch.Tensor) -> float:
    """Compute normalized orthogonality error: ||W @ W.T - I|| / ||W||.

    Args:
        W: Orthogonal matrix to verify.

    Returns:
        Normalized orthogonality error.
    """
    device = W.device
    dtype = W.dtype
    n = W.size(0)

    identity = torch.eye(n, device=device, dtype=dtype)
    WWT = W @ W.T

    error_norm = (WWT - identity).norm().item()
    W_norm = W.norm().item()

    return error_norm / (W_norm + 1e-8)


class TestConfig:
    """Test configuration constants."""

    # Maximum allowed orthogonality error for tests
    # Note: Quintic Newton-Schulz (Muon) produces error ~2-10 depending on matrix size
    # This is by design - the quintic iteration maximizes convergence speed
    # for the Muon optimizer rather than achieving exact orthogonality
    MAX_ORTHOGONALITY_ERROR = 15.0

    # Maximum allowed normalized orthogonality error
    MAX_NORMALIZED_ERROR = 0.01

    # Default number of Newton-Schulz iterations
    DEFAULT_STEPS = 5


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "distributed: marks tests requiring distributed setup"
    )
    config.addinivalue_line(
        "markers", "cuda: marks tests requiring CUDA (skipped on CPU)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # Skip CUDA tests if not available
    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if item.get_closest_marker("cuda"):
                item.add_marker(skip_cuda)
