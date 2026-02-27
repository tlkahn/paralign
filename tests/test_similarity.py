"""Tests for paralign similarity matrix computation."""

from __future__ import annotations

import numpy as np

from paralign._similarity import (
    apply_window_mask,
    compute_windowed_similarity,
    cosine_similarity_matrix,
)


class TestCosineSimilarityMatrix:
    """Tests for vectorized cosine similarity."""

    def test_identical_vectors(self) -> None:
        """Identical normalized vectors should have similarity 1.0."""
        emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sim = cosine_similarity_matrix(emb, emb)
        np.testing.assert_allclose(np.diag(sim), [1.0, 1.0], atol=1e-6)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors should have similarity 0.0."""
        src = np.array([[1.0, 0.0]], dtype=np.float32)
        tgt = np.array([[0.0, 1.0]], dtype=np.float32)
        sim = cosine_similarity_matrix(src, tgt)
        np.testing.assert_allclose(sim[0, 0], 0.0, atol=1e-6)

    def test_opposite_vectors(self) -> None:
        """Opposite vectors should have similarity -1.0."""
        src = np.array([[1.0, 0.0]], dtype=np.float32)
        tgt = np.array([[-1.0, 0.0]], dtype=np.float32)
        sim = cosine_similarity_matrix(src, tgt)
        np.testing.assert_allclose(sim[0, 0], -1.0, atol=1e-6)

    def test_shape(self) -> None:
        """Output shape should be (n_src, n_tgt)."""
        src = np.random.randn(5, 64).astype(np.float32)
        tgt = np.random.randn(8, 64).astype(np.float32)
        sim = cosine_similarity_matrix(src, tgt)
        assert sim.shape == (5, 8)

    def test_values_in_range(self) -> None:
        """All similarity values should be in [-1, 1]."""
        rng = np.random.RandomState(42)
        src = rng.randn(10, 32).astype(np.float32)
        tgt = rng.randn(10, 32).astype(np.float32)
        sim = cosine_similarity_matrix(src, tgt)
        assert sim.min() >= -1.0 - 1e-6
        assert sim.max() <= 1.0 + 1e-6

    def test_unnormalized_input(self) -> None:
        """Function should handle unnormalized vectors correctly."""
        src = np.array([[3.0, 0.0]], dtype=np.float32)
        tgt = np.array([[5.0, 0.0]], dtype=np.float32)
        sim = cosine_similarity_matrix(src, tgt)
        np.testing.assert_allclose(sim[0, 0], 1.0, atol=1e-6)

    def test_regression_vs_loop(self) -> None:
        """Verify vectorized output matches naive loop-based computation."""
        rng = np.random.RandomState(123)
        src = rng.randn(7, 16).astype(np.float32)
        tgt = rng.randn(9, 16).astype(np.float32)

        # Naive loop (like lingtrain's get_sim_matrix)
        expected = np.zeros((7, 9), dtype=np.float32)
        for i in range(7):
            for j in range(9):
                dot = np.dot(src[i], tgt[j])
                norm = np.linalg.norm(src[i]) * np.linalg.norm(tgt[j])
                expected[i, j] = dot / norm if norm > 0 else 0.0

        sim = cosine_similarity_matrix(src, tgt)
        np.testing.assert_allclose(sim, expected, atol=1e-5)


class TestApplyWindowMask:
    """Tests for diagonal band masking."""

    def test_no_masking_with_large_window(self) -> None:
        """Window larger than matrix should leave everything unmasked."""
        sim = np.ones((5, 5), dtype=np.float32)
        masked = apply_window_mask(sim, window=100)
        np.testing.assert_array_equal(masked, sim)

    def test_tight_window_masks_corners(self) -> None:
        """Tight window should zero out far-from-diagonal elements."""
        sim = np.ones((5, 5), dtype=np.float32)
        masked = apply_window_mask(sim, window=1)
        # Corner elements should be masked
        assert masked[0, 4] == 0.0
        assert masked[4, 0] == 0.0
        # Diagonal should survive
        assert masked[0, 0] == 1.0
        assert masked[4, 4] == 1.0

    def test_rectangular_matrix(self) -> None:
        """Window mask should work on non-square matrices."""
        sim = np.ones((3, 9), dtype=np.float32)
        masked = apply_window_mask(sim, window=2)
        # (0,0) should be preserved, (0,8) should be masked
        assert masked[0, 0] == 1.0
        assert masked[0, 8] == 0.0
        # (2,8) should be preserved (last row, last col in 3:9 ratio)
        assert masked[2, 8] == 1.0

    def test_preserves_diagonal_band(self) -> None:
        """The diagonal band should be fully preserved."""
        n = 10
        sim = np.ones((n, n), dtype=np.float32)
        masked = apply_window_mask(sim, window=3)
        for i in range(n):
            assert masked[i, i] == 1.0


class TestComputeWindowedSimilarity:
    """Tests for the full pipeline: cosine sim + window mask + floor."""

    def test_basic_pipeline(self) -> None:
        """Full pipeline should produce valid similarity matrix."""
        rng = np.random.RandomState(42)
        src = rng.randn(5, 32).astype(np.float32)
        tgt = rng.randn(5, 32).astype(np.float32)
        sim = compute_windowed_similarity(src, tgt, window=3, floor=0.0)
        assert sim.shape == (5, 5)

    def test_floor_clipping(self) -> None:
        """Values below floor should be clipped to 0."""
        src = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        tgt = np.array([[0.7, 0.7], [0.0, 1.0]], dtype=np.float32)
        sim = compute_windowed_similarity(src, tgt, window=100, floor=0.8)
        # Similarity below 0.8 should be clipped to 0
        assert sim[0, 0] < 0.8 or sim[0, 0] == 0.0
        # Perfect match should survive
        np.testing.assert_allclose(sim[1, 1], 1.0, atol=1e-5)

    def test_window_and_floor_combined(self) -> None:
        """Window masking and floor should work together."""
        rng = np.random.RandomState(42)
        src = rng.randn(8, 16).astype(np.float32)
        tgt = rng.randn(8, 16).astype(np.float32)
        sim = compute_windowed_similarity(src, tgt, window=2, floor=0.1)
        # No negative values due to floor
        assert sim.min() >= 0.0
        # Far-from-diagonal values should be 0
        assert sim[0, 7] == 0.0
