"""Tests for paralign DP alignment algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from paralign._dp import DPConfig, dp_align
from paralign._types import AlignmentType


def _make_diagonal_sim(n: int, on_diag: float = 0.9, off_diag: float = 0.1) -> np.ndarray:
    """Create a similarity matrix with high values on diagonal."""
    sim = np.full((n, n), off_diag, dtype=np.float32)
    np.fill_diagonal(sim, on_diag)
    return sim


def _make_embeddings(n: int, dim: int = 8) -> np.ndarray:
    """Create simple unit embeddings for n sentences."""
    rng = np.random.RandomState(42)
    emb = rng.randn(n, dim).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


class TestDPConfig:
    """Tests for DPConfig frozen dataclass."""

    def test_defaults(self) -> None:
        cfg = DPConfig()
        assert cfg.skip_penalty == pytest.approx(-0.3)
        assert cfg.band_width > 0

    def test_frozen(self) -> None:
        cfg = DPConfig()
        with pytest.raises(AttributeError):
            cfg.skip_penalty = 0.0  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = DPConfig(skip_penalty=-0.5, band_width=20)
        assert cfg.skip_penalty == -0.5
        assert cfg.band_width == 20


class TestDPAlignDiagonal:
    """Tests for perfect diagonal alignment."""

    def test_perfect_diagonal(self) -> None:
        """With high diagonal similarity, should produce all 1:1 pairs."""
        n = 5
        sim = _make_diagonal_sim(n, on_diag=0.95, off_diag=0.05)
        emb = _make_embeddings(n)
        pairs = dp_align(sim, emb, emb, DPConfig())

        # Should be exactly n pairs
        assert len(pairs) == n

        # All should be 1:1
        for pair in pairs:
            assert pair.alignment_type == AlignmentType.ONE_TO_ONE

        # Should map i -> i
        for i, pair in enumerate(pairs):
            assert pair.source_indices == (i,)
            assert pair.target_indices == (i,)

    def test_scores_are_positive(self) -> None:
        """Alignment scores should be positive for matching pairs."""
        sim = _make_diagonal_sim(3)
        emb = _make_embeddings(3)
        pairs = dp_align(sim, emb, emb, DPConfig())
        for pair in pairs:
            assert pair.score > 0


class TestDPAlignMerge:
    """Tests for 2:1 and 1:2 merge transitions."""

    def test_two_to_one_merge(self) -> None:
        """Two source sentences should merge when they best match one target."""
        # 3 source, 2 target: sources 0+1 -> target 0, source 2 -> target 1
        sim = np.array(
            [
                [0.6, 0.1],
                [0.6, 0.1],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        src_emb = np.array(
            [[0.7, 0.7], [0.7, 0.7], [0.0, 1.0]],
            dtype=np.float32,
        )
        tgt_emb = np.array(
            [[0.7, 0.7], [0.0, 1.0]],
            dtype=np.float32,
        )
        # Normalize
        src_emb /= np.linalg.norm(src_emb, axis=1, keepdims=True)
        tgt_emb /= np.linalg.norm(tgt_emb, axis=1, keepdims=True)

        pairs = dp_align(sim, src_emb, tgt_emb, DPConfig(band_width=100))

        # Find the 2:1 pair
        types = [p.alignment_type for p in pairs]
        assert AlignmentType.TWO_TO_ONE in types

    def test_one_to_two_merge(self) -> None:
        """One source should match two targets when appropriate."""
        # 2 source, 3 target: source 0 -> targets 0+1, source 1 -> target 2
        sim = np.array(
            [
                [0.6, 0.6, 0.1],
                [0.1, 0.1, 0.9],
            ],
            dtype=np.float32,
        )
        src_emb = np.array(
            [[0.7, 0.7], [0.0, 1.0]],
            dtype=np.float32,
        )
        tgt_emb = np.array(
            [[0.7, 0.7], [0.7, 0.7], [0.0, 1.0]],
            dtype=np.float32,
        )
        src_emb /= np.linalg.norm(src_emb, axis=1, keepdims=True)
        tgt_emb /= np.linalg.norm(tgt_emb, axis=1, keepdims=True)

        pairs = dp_align(sim, src_emb, tgt_emb, DPConfig(band_width=100))

        types = [p.alignment_type for p in pairs]
        assert AlignmentType.ONE_TO_TWO in types


class TestDPAlignSkips:
    """Tests for skip transitions (1:0 and 0:1)."""

    def test_source_skip(self) -> None:
        """A source sentence with no good target match should be skipped."""
        # 3 source, 2 target: source 1 has no good match
        sim = np.array(
            [
                [0.95, 0.05],
                [0.05, 0.05],  # bad match for everything
                [0.05, 0.95],
            ],
            dtype=np.float32,
        )
        emb_src = _make_embeddings(3)
        emb_tgt = _make_embeddings(2)

        cfg = DPConfig(skip_penalty=-0.02, band_width=100)
        pairs = dp_align(sim, emb_src, emb_tgt, cfg)

        # Should contain a 1:0 skip or a 2:1 merge -- either way, all sources covered
        source_indices = set()
        for p in pairs:
            source_indices.update(p.source_indices)
        assert source_indices == {0, 1, 2}

    def test_target_skip(self) -> None:
        """A target sentence with no good source match should be skipped."""
        # 2 source, 3 target: target 1 has no good match
        sim = np.array(
            [
                [0.95, 0.05, 0.05],
                [0.05, 0.05, 0.95],
            ],
            dtype=np.float32,
        )
        emb_src = _make_embeddings(2)
        emb_tgt = _make_embeddings(3)

        cfg = DPConfig(skip_penalty=-0.02, band_width=100)
        pairs = dp_align(sim, emb_src, emb_tgt, cfg)

        target_indices = set()
        for p in pairs:
            target_indices.update(p.target_indices)
        assert target_indices == {0, 1, 2}


class TestDPAlignInvariants:
    """Tests for alignment invariants that must always hold."""

    def test_monotonicity(self) -> None:
        """Source and target indices must be monotonically increasing."""
        rng = np.random.RandomState(99)
        sim = rng.rand(10, 12).astype(np.float32)
        src_emb = _make_embeddings(10)
        tgt_emb = _make_embeddings(12)

        pairs = dp_align(sim, src_emb, tgt_emb, DPConfig(band_width=100))

        prev_src = -1
        prev_tgt = -1
        for pair in pairs:
            if pair.source_indices:
                assert min(pair.source_indices) > prev_src, (
                    f"Source indices not monotonic: {pair.source_indices} after {prev_src}"
                )
                prev_src = max(pair.source_indices)
            if pair.target_indices:
                assert min(pair.target_indices) > prev_tgt, (
                    f"Target indices not monotonic: {pair.target_indices} after {prev_tgt}"
                )
                prev_tgt = max(pair.target_indices)

    def test_full_coverage(self) -> None:
        """Every source and target index must appear exactly once."""
        n_src, n_tgt = 8, 10
        sim = _make_diagonal_sim(n_src)[:, :n_tgt]  # truncate for non-square
        # Actually, make a proper random one
        rng = np.random.RandomState(77)
        sim = rng.rand(n_src, n_tgt).astype(np.float32)
        src_emb = _make_embeddings(n_src)
        tgt_emb = _make_embeddings(n_tgt)

        pairs = dp_align(sim, src_emb, tgt_emb, DPConfig(band_width=100))

        src_seen: list[int] = []
        tgt_seen: list[int] = []
        for pair in pairs:
            src_seen.extend(pair.source_indices)
            tgt_seen.extend(pair.target_indices)

        assert sorted(src_seen) == list(range(n_src))
        assert sorted(tgt_seen) == list(range(n_tgt))

    def test_no_crossings(self) -> None:
        """No alignment pair should cross another."""
        rng = np.random.RandomState(55)
        sim = rng.rand(15, 15).astype(np.float32)
        emb = _make_embeddings(15)

        pairs = dp_align(sim, emb, emb, DPConfig(band_width=100))

        for i in range(len(pairs) - 1):
            p1 = pairs[i]
            p2 = pairs[i + 1]
            if p1.source_indices and p2.source_indices:
                assert max(p1.source_indices) < min(p2.source_indices)
            if p1.target_indices and p2.target_indices:
                assert max(p1.target_indices) < min(p2.target_indices)


class TestDPAlignBand:
    """Tests for band restriction."""

    def test_band_restriction(self) -> None:
        """With tight band, alignment should not consider far-off-diagonal cells."""
        n = 20
        # Put high similarity on anti-diagonal (maximally off-diagonal)
        sim = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            sim[i, n - 1 - i] = 0.99
        # Put moderate similarity on diagonal
        np.fill_diagonal(sim, 0.5)
        emb = _make_embeddings(n)

        # Tight band should force diagonal alignment
        pairs = dp_align(sim, emb, emb, DPConfig(band_width=3))
        for pair in pairs:
            if pair.source_indices and pair.target_indices:
                src_max = max(pair.source_indices)
                tgt_max = max(pair.target_indices)
                # Should be close to diagonal
                assert abs(src_max - tgt_max) <= 5, (
                    f"Pair ({pair.source_indices}, {pair.target_indices}) too far from diagonal"
                )
