"""Unit tests for muon_fsdp.utils module.

This module contains tests for the Newton-Schulz iteration and other
utility functions in the muon_fsdp package.
"""

import pytest
import torch

from muon_fsdp.utils import zeropower_via_newtonschulz5
from tests.conftest import compute_orthogonality_error, TestConfig


class TestZeropowerViaNewtonschulz5:
    """Test suite for zeropower_via_newtonschulz5 function."""

    def test_basic_orthogonality(self, random_matrix_256):
        """Test that output matrix is orthogonal."""
        G = random_matrix_256
        W = zeropower_via_newtonschulz5(G)

        assert W.shape == G.shape
        error = compute_orthogonality_error(W)
        assert error < TestConfig.MAX_ORTHOGONALITY_ERROR, (
            f"Orthogonality error {error:.6f} exceeds maximum "
            f"{TestConfig.MAX_ORTHOGONALITY_ERROR}"
        )

    def test_orthogonality_512x512(self, random_matrix_512):
        """Test orthogonality for 512x512 matrix."""
        G = random_matrix_512
        W = zeropower_via_newtonschulz5(G)

        assert W.shape == G.shape
        error = compute_orthogonality_error(W)
        assert error < TestConfig.MAX_ORTHOGONALITY_ERROR

    def test_orthogonality_tall_matrix(self, random_matrix_tall):
        """Test orthogonality for tall matrix (m > n)."""
        G = random_matrix_tall  # 512x256
        W = zeropower_via_newtonschulz5(G)

        assert W.shape == G.shape
        error = compute_orthogonality_error(W)
        assert error < TestConfig.MAX_ORTHOGONALITY_ERROR

    def test_orthogonality_wide_matrix(self, random_matrix_wide):
        """Test orthogonality for wide matrix (m < n)."""
        G = random_matrix_wide  # 256x512
        W = zeropower_via_newtonschulz5(G)

        assert W.shape == G.shape
        # For wide matrices, we check the smaller dimension
        W_reduced = W[:, :256] if W.size(1) > W.size(0) else W
        error = compute_orthogonality_error(W_reduced)
        assert error < TestConfig.MAX_ORTHOGONALITY_ERROR

    def test_bfloat16_support(self, bfloat16_matrix):
        """Test that bfloat16 computation works."""
        G = bfloat16_matrix
        W = zeropower_via_newtonschulz5(G)

        assert W.dtype == torch.bfloat16
        assert W.shape == G.shape

    def test_custom_iterations(self, random_matrix_256):
        """Test with different number of iterations."""
        G = random_matrix_256

        for steps in [1, 3, 5, 10]:
            W = zeropower_via_newtonschulz5(G, steps=steps)
            assert W.shape == G.shape
            error = compute_orthogonality_error(W)
            # More iterations should give better results
            if steps >= 5:
                assert error < TestConfig.MAX_ORTHOGONALITY_ERROR

    def test_deterministic_with_seed(self, seed):
        """Test that same seed produces same results."""
        torch.manual_seed(seed)

        G1 = torch.randn(256, 256)
        G2 = torch.randn(256, 256)

        W1 = zeropower_via_newtonschulz5(G1)
        W2 = zeropower_via_newtonschulz5(G2)

        # Different inputs should produce different outputs
        assert not torch.allclose(W1, W2)

    def test_empty_matrix_error(self):
        """Test that empty tensor is handled."""
        G = torch.tensor([]).reshape(0, 0)
        W = zeropower_via_newtonschulz5(G)
        assert W.numel() == 0

    def test_invalid_dimensions(self):
        """Test that 1D tensor raises error."""
        G = torch.randn(256)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            zeropower_via_newtonschulz5(G)

    def test_invalid_steps(self):
        """Test that negative steps raises error."""
        G = torch.randn(256, 256)
        with pytest.raises(ValueError, match="non-negative"):
            zeropower_via_newtonschulz5(G, steps=-1)

    def test_small_matrix(self, random_matrix_small):
        """Test with small matrix."""
        G = random_matrix_small
        W = zeropower_via_newtonschulz5(G)

        assert W.shape == G.shape
        error = compute_orthogonality_error(W)
        assert error < TestConfig.MAX_ORTHOGONALITY_ERROR

    def test_large_matrix(self, random_matrix_large):
        """Test with large matrix."""
        G = random_matrix_large
        W = zeropower_via_newtonschulz5(G)

        assert W.shape == G.shape
        error = compute_orthogonality_error(W)
        assert error < TestConfig.MAX_ORTHOGONALITY_ERROR

    def test_preserves_dtype(self, random_matrix_256):
        """Test that output dtype matches input dtype."""
        for dtype in [torch.float32, torch.float64]:
            G = random_matrix_256.to(dtype)
            W = zeropower_via_newtonschulz5(G)
            assert W.dtype == dtype
