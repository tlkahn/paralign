"""Edge case tests for paralign."""

from __future__ import annotations

import numpy as np
import pytest

from paralign import AlignConfig, AlignmentResult, AlignmentType, align
from paralign._dp import DPConfig, dp_align
from paralign._similarity import compute_windowed_similarity, cosine_similarity_matrix


class TestEmptyInputs:
    """Tests for empty input handling."""

    def test_both_empty(self, deterministic_embedder: object) -> None:
        """Empty source and target should return empty result."""
        result = align([], [], model=deterministic_embedder)
        assert isinstance(result, AlignmentResult)
        assert len(result.pairs) == 0
        assert result.source_count == 0
        assert result.target_count == 0

    def test_empty_source(self, deterministic_embedder: object) -> None:
        """Empty source should produce 0:1 pairs for all targets."""
        result = align([], ["A", "B"], model=deterministic_embedder)
        assert result.source_count == 0
        assert result.target_count == 2
        for pair in result.pairs:
            assert pair.alignment_type == AlignmentType.ZERO_TO_ONE

    def test_empty_target(self, deterministic_embedder: object) -> None:
        """Empty target should produce 1:0 pairs for all sources."""
        result = align(["A", "B"], [], model=deterministic_embedder)
        assert result.source_count == 2
        assert result.target_count == 0
        for pair in result.pairs:
            assert pair.alignment_type == AlignmentType.ONE_TO_ZERO


class TestSingleSentence:
    """Tests for single-sentence inputs."""

    def test_one_to_one(self, deterministic_embedder: object) -> None:
        """Single source and target should produce exactly one 1:1 pair."""
        result = align(["Hello"], ["Hola"], model=deterministic_embedder)
        assert len(result.pairs) == 1
        assert result.pairs[0].alignment_type == AlignmentType.ONE_TO_ONE
        assert result.pairs[0].source_indices == (0,)
        assert result.pairs[0].target_indices == (0,)

    def test_one_source_two_targets(self, deterministic_embedder: object) -> None:
        """One source, two targets should still produce valid alignment."""
        result = align(["Hello"], ["Hola", "Mundo"], model=deterministic_embedder)
        src_indices: set[int] = set()
        tgt_indices: set[int] = set()
        for pair in result.pairs:
            src_indices.update(pair.source_indices)
            tgt_indices.update(pair.target_indices)
        assert src_indices == {0}
        assert tgt_indices == {0, 1}


class TestUnequalLengths:
    """Tests for very unequal input lengths."""

    def test_ten_to_two(self, deterministic_embedder: object) -> None:
        """10 source sentences vs 2 target should still align fully."""
        source = [f"Source sentence {i}" for i in range(10)]
        target = ["Target A", "Target B"]
        result = align(source, target, model=deterministic_embedder)

        # Coverage check
        src_indices: set[int] = set()
        tgt_indices: set[int] = set()
        for pair in result.pairs:
            src_indices.update(pair.source_indices)
            tgt_indices.update(pair.target_indices)
        assert src_indices == set(range(10))
        assert tgt_indices == {0, 1}

    def test_two_to_ten(self, deterministic_embedder: object) -> None:
        """2 source vs 10 target should still align fully."""
        source = ["Source A", "Source B"]
        target = [f"Target sentence {i}" for i in range(10)]
        result = align(source, target, model=deterministic_embedder)

        src_indices: set[int] = set()
        tgt_indices: set[int] = set()
        for pair in result.pairs:
            src_indices.update(pair.source_indices)
            tgt_indices.update(pair.target_indices)
        assert src_indices == {0, 1}
        assert tgt_indices == set(range(10))

    def test_monotonicity_unequal(self, deterministic_embedder: object) -> None:
        """Monotonicity must hold even with very unequal lengths."""
        source = [f"S{i}" for i in range(15)]
        target = [f"T{i}" for i in range(3)]
        result = align(source, target, model=deterministic_embedder)

        prev_src = -1
        prev_tgt = -1
        for pair in result.pairs:
            if pair.source_indices:
                assert min(pair.source_indices) > prev_src
                prev_src = max(pair.source_indices)
            if pair.target_indices:
                assert min(pair.target_indices) > prev_tgt
                prev_tgt = max(pair.target_indices)


class TestZeroVectors:
    """Tests for zero-vector embeddings (no division by zero)."""

    def test_zero_source_embeddings(self) -> None:
        """Zero embeddings should not cause division by zero."""
        src = np.zeros((3, 8), dtype=np.float32)
        tgt = np.random.randn(3, 8).astype(np.float32)
        sim = cosine_similarity_matrix(src, tgt)
        assert not np.any(np.isnan(sim))
        assert not np.any(np.isinf(sim))

    def test_zero_target_embeddings(self) -> None:
        """Zero target embeddings should not cause division by zero."""
        src = np.random.randn(3, 8).astype(np.float32)
        tgt = np.zeros((3, 8), dtype=np.float32)
        sim = cosine_similarity_matrix(src, tgt)
        assert not np.any(np.isnan(sim))
        assert not np.any(np.isinf(sim))

    def test_both_zero_embeddings(self) -> None:
        """All-zero embeddings should produce finite similarity values."""
        src = np.zeros((2, 8), dtype=np.float32)
        tgt = np.zeros((2, 8), dtype=np.float32)
        sim = cosine_similarity_matrix(src, tgt)
        assert not np.any(np.isnan(sim))
        assert not np.any(np.isinf(sim))

    def test_dp_with_zero_similarity(self) -> None:
        """DP should handle all-zero similarity matrix gracefully."""
        sim = np.zeros((3, 3), dtype=np.float32)
        emb = np.zeros((3, 8), dtype=np.float32)
        pairs = dp_align(sim, emb, emb, DPConfig())
        # Should still produce valid alignment
        src_seen = set()
        tgt_seen = set()
        for p in pairs:
            src_seen.update(p.source_indices)
            tgt_seen.update(p.target_indices)
        assert src_seen == {0, 1, 2}
        assert tgt_seen == {0, 1, 2}


class TestLargeWindow:
    """Tests for window larger than matrix dimensions."""

    def test_window_larger_than_matrix(self) -> None:
        """Window > matrix size should not mask anything."""
        rng = np.random.RandomState(42)
        src = rng.randn(5, 16).astype(np.float32)
        tgt = rng.randn(5, 16).astype(np.float32)
        sim_windowed = compute_windowed_similarity(src, tgt, window=1000, floor=0.0)
        sim_raw = cosine_similarity_matrix(src, tgt)
        np.testing.assert_allclose(sim_windowed, sim_raw, atol=1e-6)

    def test_zero_window(self) -> None:
        """Window of 0 should only keep exact diagonal for square matrices."""
        src = np.eye(5, dtype=np.float32)
        tgt = np.eye(5, dtype=np.float32)
        sim = compute_windowed_similarity(src, tgt, window=0, floor=0.0)
        # Only diagonal should survive
        for i in range(5):
            assert sim[i, i] != 0.0
            for j in range(5):
                if j != i:
                    assert sim[i, j] == 0.0


class TestNoModelProvided:
    """Tests for error handling when no model is provided."""

    def test_no_model_and_no_config_model_raises(self) -> None:
        """Should raise ValueError if no model instance and no config.model."""
        with pytest.raises(ValueError, match="Either pass a model"):
            align(["A"], ["B"])

    def test_no_model_with_config_model_raises_import(self) -> None:
        """If config.model is set but sentence-transformers not installed, should fail."""
        cfg = AlignConfig(model="labse")
        # This will try to create a SentenceTransformerModel and encode,
        # which will fail if sentence-transformers is not installed
        with pytest.raises((ImportError, Exception)):
            align(["A"], ["B"], config=cfg)
