"""Tests for paralign top-level align() function and AlignConfig."""

from __future__ import annotations

import numpy as np
import pytest

from paralign import AlignConfig, AlignmentResult, AlignmentType, align
from paralign._dp import DPConfig


class TestAlignConfig:
    """Tests for AlignConfig frozen dataclass."""

    def test_defaults(self) -> None:
        cfg = AlignConfig()
        assert cfg.window == 50
        assert cfg.floor == 0.0
        assert cfg.embed_batch_size == 64
        assert cfg.normalize is True
        assert cfg.include_similarity_matrix is False

    def test_frozen(self) -> None:
        cfg = AlignConfig()
        with pytest.raises(AttributeError):
            cfg.window = 10  # type: ignore[misc]

    def test_custom_dp_config(self) -> None:
        dp = DPConfig(skip_penalty=-0.5)
        cfg = AlignConfig(dp=dp)
        assert cfg.dp.skip_penalty == -0.5


class TestAlignFunction:
    """Integration tests for align() with mock embedder."""

    def test_basic_alignment(self, deterministic_embedder: object) -> None:
        """align() should return an AlignmentResult."""
        source = ["Hello world", "How are you", "Goodbye"]
        target = ["Hola mundo", "Como estas", "Adios"]
        result = align(source, target, model=deterministic_embedder)
        assert isinstance(result, AlignmentResult)
        assert result.source_count == 3
        assert result.target_count == 3

    def test_returns_correct_sentences(self, deterministic_embedder: object) -> None:
        """Result should contain the original sentences."""
        source = ["A", "B"]
        target = ["X", "Y"]
        result = align(source, target, model=deterministic_embedder)
        assert result.source_sentences == ("A", "B")
        assert result.target_sentences == ("X", "Y")

    def test_full_coverage(self, deterministic_embedder: object) -> None:
        """All source and target indices should appear in pairs."""
        source = ["S1", "S2", "S3", "S4"]
        target = ["T1", "T2", "T3", "T4"]
        result = align(source, target, model=deterministic_embedder)

        # Note: using sets — we only check that every index is covered,
        # not which source maps to which target (the deterministic embedder
        # has no semantic similarity, so pairing order is arbitrary).
        src_indices: set[int] = set()
        tgt_indices: set[int] = set()
        for pair in result.pairs:
            src_indices.update(pair.source_indices)
            tgt_indices.update(pair.target_indices)

        assert src_indices == {0, 1, 2, 3}
        assert tgt_indices == {0, 1, 2, 3}

    def test_monotonicity(self, deterministic_embedder: object) -> None:
        """Alignment must be monotonic.

        The DP algorithm must consume ALL source and target sentences — it
        partitions both sequences into contiguous groups, never leaving any
        sentence out.  Monotonicity is a structural guarantee of the DP
        (it walks left-to-right through both sequences), not a property of
        the embeddings.  With equal-length inputs and random embeddings the
        DP defaults to 1:1 across the board because merges/skips incur
        penalties with no offsetting similarity benefit.
        """
        source = [f"Source {i}" for i in range(10)]
        target = [f"Target {i}" for i in range(10)]

        print(f"\nsrc: {source}")
        print(f"tgt: {target}")

        result = align(source, target, model=deterministic_embedder)

        src_flat = [idx for p in result.pairs for idx in p.source_indices]
        tgt_flat = [idx for p in result.pairs for idx in p.target_indices]

        print(f"\nsrc_flat: {src_flat}")
        print(f"tgt_flat: {tgt_flat}")

        assert all(a < b for a, b in zip(src_flat, src_flat[1:]))
        assert all(a < b for a, b in zip(tgt_flat, tgt_flat[1:]))

    def test_similarity_matrix_excluded_by_default(
        self, deterministic_embedder: object
    ) -> None:
        """By default, similarity_matrix should be None."""
        result = align(["A"], ["B"], model=deterministic_embedder)
        assert result.similarity_matrix is None

    def test_similarity_matrix_included_when_requested(
        self, deterministic_embedder: object
    ) -> None:
        """When include_similarity_matrix=True, matrix should be present."""
        cfg = AlignConfig(include_similarity_matrix=True)
        result = align(["A", "B"], ["X", "Y"], config=cfg, model=deterministic_embedder)
        assert result.similarity_matrix is not None
        assert result.similarity_matrix.shape == (2, 2)

    def test_config_propagation(self, deterministic_embedder: object) -> None:
        """Config settings should be respected."""
        cfg = AlignConfig(window=5, floor=0.1)
        result = align(
            ["A", "B", "C"],
            ["X", "Y", "Z"],
            config=cfg,
            model=deterministic_embedder,
        )
        # Should still produce a valid result
        assert isinstance(result, AlignmentResult)
        assert len(result.pairs) > 0

    def test_merged_pairs_integration(self, deterministic_embedder: object) -> None:
        """merged_pairs should produce text for every pair."""
        source = ["Hello", "World"]
        target = ["Hola", "Mundo"]
        result = align(source, target, model=deterministic_embedder)
        merged = result.merged_pairs()
        assert len(merged) == len(result.pairs)
        for src_text, tgt_text in merged:
            assert isinstance(src_text, str)
            assert isinstance(tgt_text, str)


class TestUnequalLengthAlignment:
    """Tests for unequal source/target lengths using IdentityEmbedder.

    IdentityEmbedder returns pre-set embeddings, giving us full control
    over the similarity matrix so we can verify the DP produces expected
    merges, skips, and groupings.
    """

    def test_two_sources_merge_into_one_target(self, identity_embedder_factory) -> None:
        """Two similar source sentences should merge into one target (2:1).

        Embeddings:
            src[0] = [1, 0]   src[1] = [1, 0]
            tgt[0] = [1, 0]

        Both sources are identical to the single target, so the DP should
        produce a 2:1 alignment.
        """
        embeddings = np.array([
            [1.0, 0.0],  # src 0
            [1.0, 0.0],  # src 1
            [1.0, 0.0],  # tgt 0
        ], dtype=np.float32)
        embedder = identity_embedder_factory(embeddings)

        result = align(["A", "B"], ["X"], model=embedder)

        assert len(result.pairs) == 1
        pair = result.pairs[0]
        assert pair.source_indices == (0, 1)
        assert pair.target_indices == (0,)
        assert pair.alignment_type == AlignmentType.TWO_TO_ONE

    def test_one_source_splits_into_two_targets(self, identity_embedder_factory) -> None:
        """One source sentence should align to two targets (1:2).

        Embeddings:
            src[0] = [1, 0]
            tgt[0] = [1, 0]   tgt[1] = [1, 0]

        The single source is identical to both targets, so the DP should
        produce a 1:2 alignment.
        """
        embeddings = np.array([
            [1.0, 0.0],  # src 0
            [1.0, 0.0],  # tgt 0
            [1.0, 0.0],  # tgt 1
        ], dtype=np.float32)
        embedder = identity_embedder_factory(embeddings)

        result = align(["A"], ["X", "Y"], model=embedder)

        assert len(result.pairs) == 1
        pair = result.pairs[0]
        assert pair.source_indices == (0,)
        assert pair.target_indices == (0, 1)
        assert pair.alignment_type == AlignmentType.ONE_TO_TWO

    def test_three_sources_two_targets_forces_merge(
        self, identity_embedder_factory
    ) -> None:
        """3 sources vs 2 targets: the DP must merge somewhere.

        Embeddings (dim=2):
            src[0] = [1, 0]   src[1] = [1, 0]   src[2] = [0, 1]
            tgt[0] = [1, 0]   tgt[1] = [0, 1]

        src[0] and src[1] are both similar to tgt[0], while src[2] matches
        tgt[1].  Expected: (src 0,1 -> tgt 0) + (src 2 -> tgt 1).
        """
        embeddings = np.array([
            [1.0, 0.0],  # src 0
            [1.0, 0.0],  # src 1
            [0.0, 1.0],  # src 2
            [1.0, 0.0],  # tgt 0
            [0.0, 1.0],  # tgt 1
        ], dtype=np.float32)
        embedder = identity_embedder_factory(embeddings)

        result = align(["A", "B", "C"], ["X", "Y"], model=embedder)

        # All indices must be covered
        src_flat = [idx for p in result.pairs for idx in p.source_indices]
        tgt_flat = [idx for p in result.pairs for idx in p.target_indices]
        assert set(src_flat) == {0, 1, 2}
        assert set(tgt_flat) == {0, 1}

        # Monotonicity
        assert all(a < b for a, b in zip(src_flat, src_flat[1:]))
        assert all(a < b for a, b in zip(tgt_flat, tgt_flat[1:]))

        # Expect a 2:1 merge for the first pair
        assert result.pairs[0].source_indices == (0, 1)
        assert result.pairs[0].target_indices == (0,)
        assert result.pairs[0].alignment_type == AlignmentType.TWO_TO_ONE

        # And a 1:1 for the second
        assert result.pairs[1].source_indices == (2,)
        assert result.pairs[1].target_indices == (1,)
        assert result.pairs[1].alignment_type == AlignmentType.ONE_TO_ONE

    def test_two_sources_three_targets_forces_split(
        self, identity_embedder_factory
    ) -> None:
        """2 sources vs 3 targets: the DP must split somewhere.

        Embeddings (dim=2):
            src[0] = [1, 0]   src[1] = [0, 1]
            tgt[0] = [1, 0]   tgt[1] = [0, 1]   tgt[2] = [0, 1]

        src[0] matches tgt[0], src[1] matches both tgt[1] and tgt[2].
        Expected: (src 0 -> tgt 0) + (src 1 -> tgt 1,2).
        """
        embeddings = np.array([
            [1.0, 0.0],  # src 0
            [0.0, 1.0],  # src 1
            [1.0, 0.0],  # tgt 0
            [0.0, 1.0],  # tgt 1
            [0.0, 1.0],  # tgt 2
        ], dtype=np.float32)
        embedder = identity_embedder_factory(embeddings)

        result = align(["A", "B"], ["X", "Y", "Z"], model=embedder)

        src_flat = [idx for p in result.pairs for idx in p.source_indices]
        tgt_flat = [idx for p in result.pairs for idx in p.target_indices]
        assert set(src_flat) == {0, 1}
        assert set(tgt_flat) == {0, 1, 2}

        assert all(a < b for a, b in zip(src_flat, src_flat[1:]))
        assert all(a < b for a, b in zip(tgt_flat, tgt_flat[1:]))

        assert result.pairs[0].source_indices == (0,)
        assert result.pairs[0].target_indices == (0,)
        assert result.pairs[0].alignment_type == AlignmentType.ONE_TO_ONE

        assert result.pairs[1].source_indices == (1,)
        assert result.pairs[1].target_indices == (1, 2)
        assert result.pairs[1].alignment_type == AlignmentType.ONE_TO_TWO

    def test_dissimilar_source_gets_skipped(self, identity_embedder_factory) -> None:
        """A source with no good target match may produce a 1:0 skip.

        Embeddings (dim=2):
            src[0] = [1, 0]   src[1] = [-1, 0]   src[2] = [0, 1]
            tgt[0] = [1, 0]   tgt[1] = [0, 1]

        src[1] is orthogonal/opposite to both targets.  The DP may skip it
        (1:0) rather than force a bad merge.  We verify all indices are
        still covered and monotonicity holds.
        """
        embeddings = np.array([
            [1.0, 0.0],   # src 0 — matches tgt 0
            [-1.0, 0.0],  # src 1 — matches nothing
            [0.0, 1.0],   # src 2 — matches tgt 1
            [1.0, 0.0],   # tgt 0
            [0.0, 1.0],   # tgt 1
        ], dtype=np.float32)
        embedder = identity_embedder_factory(embeddings)

        result = align(["A", "B", "C"], ["X", "Y"], model=embedder)

        # All indices must still be covered (DP consumes everything)
        src_flat = [idx for p in result.pairs for idx in p.source_indices]
        tgt_flat = [idx for p in result.pairs for idx in p.target_indices]
        assert set(src_flat) == {0, 1, 2}
        assert set(tgt_flat) == {0, 1}

        # Monotonicity
        assert all(a < b for a, b in zip(src_flat, src_flat[1:]))
        assert all(a < b for a, b in zip(tgt_flat, tgt_flat[1:]))


class TestTransitionLimits:
    """Tests verifying the DP caps transitions at 2 on each side.

    The DP only supports transitions up to (2,2).  There is no (1,3),
    (3,1), etc.  When the real data would ideally be a 1:3 or 3:1
    grouping, the DP approximates it by chaining smaller transitions
    (e.g. 1:2 + 0:1, or 2:1 + 1:0).  These tests verify that behaviour
    and confirm all sentences are still covered.
    """

    def test_one_source_four_targets_no_single_pair(
        self, identity_embedder_factory
    ) -> None:
        """1 source vs 4 targets: cannot produce a single 1:4 pair.

        Embeddings (dim=2):
            src[0] = [1, 0]
            tgt[0..3] = [1, 0]  (all identical to src)

        Ideally this would be 1:4, but the DP has no such transition.
        It must break it into smaller steps (e.g. 1:2 + 0:1 + 0:1).
        We verify no single pair contains more than 2 target indices.
        """
        embeddings = np.array([
            [1.0, 0.0],  # src 0
            [1.0, 0.0],  # tgt 0
            [1.0, 0.0],  # tgt 1
            [1.0, 0.0],  # tgt 2
            [1.0, 0.0],  # tgt 3
        ], dtype=np.float32)
        embedder = identity_embedder_factory(embeddings)

        result = align(["A"], ["W", "X", "Y", "Z"], model=embedder)

        # No single pair can hold more than 2 on either side
        for pair in result.pairs:
            assert len(pair.source_indices) <= 2
            assert len(pair.target_indices) <= 2

        # All indices must still be covered
        src_flat = [idx for p in result.pairs for idx in p.source_indices]
        tgt_flat = [idx for p in result.pairs for idx in p.target_indices]
        assert set(src_flat) == {0}
        assert set(tgt_flat) == {0, 1, 2, 3}

    def test_four_sources_one_target_no_single_pair(
        self, identity_embedder_factory
    ) -> None:
        """4 sources vs 1 target: cannot produce a single 4:1 pair.

        Embeddings (dim=2):
            src[0..3] = [1, 0]  (all identical)
            tgt[0] = [1, 0]

        The DP must decompose this into steps of at most 2 source
        sentences each (e.g. 2:1 + 2:0, or 2:1 + 1:0 + 1:0).
        """
        embeddings = np.array([
            [1.0, 0.0],  # src 0
            [1.0, 0.0],  # src 1
            [1.0, 0.0],  # src 2
            [1.0, 0.0],  # src 3
            [1.0, 0.0],  # tgt 0
        ], dtype=np.float32)
        embedder = identity_embedder_factory(embeddings)

        result = align(["A", "B", "C", "D"], ["X"], model=embedder)

        for pair in result.pairs:
            assert len(pair.source_indices) <= 2
            assert len(pair.target_indices) <= 2

        src_flat = [idx for p in result.pairs for idx in p.source_indices]
        tgt_flat = [idx for p in result.pairs for idx in p.target_indices]
        assert set(src_flat) == {0, 1, 2, 3}
        assert set(tgt_flat) == {0}

    def test_three_sources_one_target_decomposes(
        self, identity_embedder_factory
    ) -> None:
        """3 sources vs 1 target: the ideal 3:1 is approximated.

        Embeddings (dim=2):
            src[0..2] = [1, 0]
            tgt[0] = [1, 0]

        The DP must use at least 2 pairs (e.g. 2:1 + 1:0, or 1:0 + 2:1)
        since no single transition can consume 3 sources at once.
        """
        embeddings = np.array([
            [1.0, 0.0],  # src 0
            [1.0, 0.0],  # src 1
            [1.0, 0.0],  # src 2
            [1.0, 0.0],  # tgt 0
        ], dtype=np.float32)
        embedder = identity_embedder_factory(embeddings)

        result = align(["A", "B", "C"], ["X"], model=embedder)

        # Must use more than 1 pair — no single transition can do 3:1
        assert len(result.pairs) >= 2

        for pair in result.pairs:
            assert len(pair.source_indices) <= 2
            assert len(pair.target_indices) <= 2

        src_flat = [idx for p in result.pairs for idx in p.source_indices]
        tgt_flat = [idx for p in result.pairs for idx in p.target_indices]
        assert set(src_flat) == {0, 1, 2}
        assert set(tgt_flat) == {0}

        # Monotonicity
        assert all(a < b for a, b in zip(src_flat, src_flat[1:]))

    def test_one_source_three_targets_decomposes(
        self, identity_embedder_factory
    ) -> None:
        """1 source vs 3 targets: the ideal 1:3 is approximated.

        Embeddings (dim=2):
            src[0] = [1, 0]
            tgt[0..2] = [1, 0]

        The DP must use at least 2 pairs (e.g. 1:2 + 0:1) since no
        single transition can consume 3 targets at once.
        """
        embeddings = np.array([
            [1.0, 0.0],  # src 0
            [1.0, 0.0],  # tgt 0
            [1.0, 0.0],  # tgt 1
            [1.0, 0.0],  # tgt 2
        ], dtype=np.float32)
        embedder = identity_embedder_factory(embeddings)

        result = align(["A"], ["X", "Y", "Z"], model=embedder)

        assert len(result.pairs) >= 2

        for pair in result.pairs:
            assert len(pair.source_indices) <= 2
            assert len(pair.target_indices) <= 2

        src_flat = [idx for p in result.pairs for idx in p.source_indices]
        tgt_flat = [idx for p in result.pairs for idx in p.target_indices]
        assert set(src_flat) == {0}
        assert set(tgt_flat) == {0, 1, 2}

        # Monotonicity
        assert all(a < b for a, b in zip(tgt_flat, tgt_flat[1:]))
