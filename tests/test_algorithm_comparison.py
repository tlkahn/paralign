"""Comparison tests: paralign vs bertalign vs vecalign DP algorithms.

Each scenario constructs a desired similarity matrix, builds embeddings via
ConstantEmbedder, and runs all three algorithms to observe agreement/disagreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import ConstantEmbedder
from tests.reference_algorithms import (
    AlignmentLink,
    bertalign_second_pass_dp,
    paralign_dp,
    prepare_bertalign_inputs,
    prepare_vecalign_inputs,
    run_all_algorithms,
    vecalign_dense_dp,
    vecalign_sparse_dp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _links_structure(links: list[AlignmentLink]) -> list[tuple[list[int], list[int]]]:
    """Extract bare (src, tgt) structure for comparison."""
    return [(link.src, link.tgt) for link in links]


def _assert_covers_all(links: list[AlignmentLink], n_src: int, n_tgt: int) -> None:
    """Assert all source and target indices appear exactly once."""
    src_seen: list[int] = []
    tgt_seen: list[int] = []
    for link in links:
        src_seen.extend(link.src)
        tgt_seen.extend(link.tgt)
    assert sorted(src_seen) == list(range(n_src)), f"src coverage: {sorted(src_seen)}"
    assert sorted(tgt_seen) == list(range(n_tgt)), f"tgt coverage: {sorted(tgt_seen)}"


def _assert_monotonic(links: list[AlignmentLink]) -> None:
    """Assert source and target indices are monotonically increasing."""
    prev_s = -1
    prev_t = -1
    for link in links:
        if link.src:
            assert min(link.src) > prev_s, f"src not monotonic: {link.src} after {prev_s}"
            prev_s = max(link.src)
        if link.tgt:
            assert min(link.tgt) > prev_t, f"tgt not monotonic: {link.tgt} after {prev_t}"
            prev_t = max(link.tgt)


# ---------------------------------------------------------------------------
# ConstantEmbedder unit tests
# ---------------------------------------------------------------------------


class TestConstantEmbedder:
    """Verify that ConstantEmbedder produces the requested similarity matrix."""

    def test_identity_sim(self) -> None:
        """Diagonal similarity matrix should produce exact cosine similarities."""
        n = 4
        sim = np.eye(n, dtype=np.float32) * 0.8
        np.fill_diagonal(sim, 0.8)
        # Off-diagonal already 0
        ce = ConstantEmbedder(sim)

        actual = ce.src_emb @ ce.tgt_emb.T
        np.testing.assert_allclose(actual, sim, atol=1e-5)

    def test_arbitrary_sim(self) -> None:
        """An arbitrary similarity matrix with column norms < 1."""
        sim = np.array(
            [[0.9, 0.1, 0.2], [0.1, 0.85, 0.15], [0.05, 0.1, 0.8]],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)

        actual = ce.src_emb @ ce.tgt_emb.T
        np.testing.assert_allclose(actual, sim, atol=1e-5)

    def test_unit_norm(self) -> None:
        """All embeddings should have unit norm."""
        sim = np.array([[0.5, 0.3], [0.2, 0.6]], dtype=np.float32)
        ce = ConstantEmbedder(sim)

        src_norms = np.linalg.norm(ce.src_emb, axis=1)
        tgt_norms = np.linalg.norm(ce.tgt_emb, axis=1)
        np.testing.assert_allclose(src_norms, 1.0, atol=1e-5)
        np.testing.assert_allclose(tgt_norms, 1.0, atol=1e-5)

    def test_column_norm_too_large(self) -> None:
        """Should raise ValueError when column norm >= 1."""
        sim = np.array([[0.8], [0.7]], dtype=np.float32)
        # Column norm = sqrt(0.64 + 0.49) = sqrt(1.13) > 1
        with pytest.raises(ValueError, match="L2 norm >= 1"):
            ConstantEmbedder(sim)

    def test_similarity_property(self) -> None:
        """The similarity_matrix property should return the original matrix."""
        sim = np.array([[0.5, 0.3], [0.2, 0.6]], dtype=np.float32)
        ce = ConstantEmbedder(sim)
        np.testing.assert_array_equal(ce.similarity_matrix, sim)


# ---------------------------------------------------------------------------
# Scenario 1: Perfect diagonal
# ---------------------------------------------------------------------------


class TestPerfectDiagonal:
    """5x5 identity-like similarity: all algorithms should produce 1:1 alignment."""

    @pytest.fixture()
    def setup(self):
        n = 5
        sim = np.full((n, n), 0.05, dtype=np.float32)
        np.fill_diagonal(sim, 0.9)
        ce = ConstantEmbedder(sim)
        return ce, n

    def test_paralign_diagonal(self, setup) -> None:
        ce, n = setup
        links = paralign_dp(ce.similarity_matrix, ce.src_emb, ce.tgt_emb)
        expected = [([i], [i]) for i in range(n)]
        assert _links_structure(links) == expected

    def test_bertalign_diagonal(self, setup) -> None:
        ce, n = setup
        src_vecs, tgt_vecs = prepare_bertalign_inputs(ce.src_emb, ce.tgt_emb, max_align=5)
        links = bertalign_second_pass_dp(src_vecs, tgt_vecs, n, n, max_align=5, skip=-0.1)
        expected = [([i], [i]) for i in range(n)]
        assert _links_structure(links) == expected

    def test_vecalign_dense_diagonal(self, setup) -> None:
        ce, n = setup
        v0, v1, n0, n1 = prepare_vecalign_inputs(ce.src_emb, ce.tgt_emb, num_overlaps=1)
        links = vecalign_dense_dp(v0, v1, n0, n1, del_penalty=0.5)
        expected = [([i], [i]) for i in range(n)]
        assert _links_structure(links) == expected

    def test_vecalign_sparse_diagonal(self, setup) -> None:
        ce, n = setup
        v0, v1, n0, n1 = prepare_vecalign_inputs(ce.src_emb, ce.tgt_emb, num_overlaps=3)
        links = vecalign_sparse_dp(v0, v1, n0, n1, del_penalty=0.5, max_align=4)
        expected = [([i], [i]) for i in range(n)]
        assert _links_structure(links) == expected

    def test_all_agree(self, setup) -> None:
        ce, n = setup
        results = run_all_algorithms(ce.src_emb, ce.tgt_emb)
        pa = _links_structure(results["paralign"])
        ba = _links_structure(results["bertalign"])
        va = _links_structure(results["vecalign"])
        expected = [([i], [i]) for i in range(n)]
        assert pa == expected
        assert ba == expected
        assert va == expected


# ---------------------------------------------------------------------------
# Scenario 2: Clear merges
# ---------------------------------------------------------------------------


class TestClearMerges:
    """Cases where merges are unambiguously optimal."""

    def test_2_to_1_merge(self) -> None:
        """3 src, 2 tgt: src[0,1]->tgt[0], src[2]->tgt[1]."""
        sim = np.array(
            [[0.7, 0.05], [0.7, 0.05], [0.05, 0.9]],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)

        pa = paralign_dp(ce.similarity_matrix, ce.src_emb, ce.tgt_emb, merge_penalty=-0.01)
        ba_src, ba_tgt = prepare_bertalign_inputs(ce.src_emb, ce.tgt_emb, max_align=5)
        ba = bertalign_second_pass_dp(ba_src, ba_tgt, 3, 2, max_align=5, skip=-0.1)

        # Both should produce 2:1 + 1:1
        pa_struct = _links_structure(pa)
        ba_struct = _links_structure(ba)

        assert pa_struct == [([0, 1], [0]), ([2], [1])]
        assert ba_struct == [([0, 1], [0]), ([2], [1])]

        _assert_covers_all(pa, 3, 2)
        _assert_covers_all(ba, 3, 2)

    def test_1_to_2_merge(self) -> None:
        """2 src, 3 tgt: src[0]->tgt[0,1], src[1]->tgt[2]."""
        sim = np.array(
            [[0.7, 0.7, 0.05], [0.05, 0.05, 0.9]],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)

        pa = paralign_dp(ce.similarity_matrix, ce.src_emb, ce.tgt_emb, merge_penalty=-0.01)
        ba_src, ba_tgt = prepare_bertalign_inputs(ce.src_emb, ce.tgt_emb, max_align=5)
        ba = bertalign_second_pass_dp(ba_src, ba_tgt, 2, 3, max_align=5, skip=-0.1)

        pa_struct = _links_structure(pa)
        ba_struct = _links_structure(ba)

        assert pa_struct == [([0], [0, 1]), ([1], [2])]
        assert ba_struct == [([0], [0, 1]), ([1], [2])]

        _assert_covers_all(pa, 2, 3)
        _assert_covers_all(ba, 2, 3)

    def test_2_to_2_merge(self) -> None:
        """4 src, 4 tgt: src[0,1]->tgt[0,1], src[2,3]->tgt[2,3].

        2:2 merge beats 1:1 when cross-matches exist (s0->t1, s1->t0 are high)
        but monotonic 1:1 pairing (s0->t0, s1->t1) gives low scores.
        """
        sim = np.array(
            [
                [0.1, 0.5, 0.05, 0.05],
                [0.5, 0.1, 0.05, 0.05],
                [0.05, 0.05, 0.1, 0.5],
                [0.05, 0.05, 0.5, 0.1],
            ],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)

        pa = paralign_dp(ce.similarity_matrix, ce.src_emb, ce.tgt_emb, merge_penalty=-0.01)
        pa_struct = _links_structure(pa)

        assert pa_struct == [([0, 1], [0, 1]), ([2, 3], [2, 3])]
        _assert_covers_all(pa, 4, 4)


# ---------------------------------------------------------------------------
# Scenario 3: Skip/deletion
# ---------------------------------------------------------------------------


class TestSkipDeletion:
    """Cases where some sentences must be skipped/deleted."""

    def test_source_skip(self) -> None:
        """3 src, 2 tgt: middle source has no good match."""
        sim = np.array(
            [[0.9, 0.05], [0.05, 0.05], [0.05, 0.9]],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)

        pa = paralign_dp(
            ce.similarity_matrix, ce.src_emb, ce.tgt_emb,
            skip_penalty=-0.02, merge_penalty=-0.5,
        )
        pa_struct = _links_structure(pa)
        assert pa_struct == [([0], [0]), ([1], []), ([2], [1])]

    def test_target_skip(self) -> None:
        """2 src, 3 tgt: middle target has no good match."""
        sim = np.array(
            [[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)

        pa = paralign_dp(
            ce.similarity_matrix, ce.src_emb, ce.tgt_emb,
            skip_penalty=-0.02, merge_penalty=-0.5,
        )
        pa_struct = _links_structure(pa)
        assert pa_struct == [([0], [0]), ([], [1]), ([1], [2])]

    def test_bertalign_source_skip(self) -> None:
        """Bertalign should also skip the unmatched source."""
        sim = np.array(
            [[0.9, 0.05], [0.05, 0.05], [0.05, 0.9]],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)

        src_vecs, tgt_vecs = prepare_bertalign_inputs(ce.src_emb, ce.tgt_emb, max_align=3)
        links = bertalign_second_pass_dp(
            src_vecs, tgt_vecs, 3, 2, max_align=3, skip=-0.02,
        )
        ba_struct = _links_structure(links)

        # Bertalign should skip src[1]
        assert ba_struct == [([0], [0]), ([1], []), ([2], [1])]

    def test_consecutive_source_skips(self) -> None:
        """4 src, 2 tgt: two consecutive source sentences are unmatched."""
        sim = np.array(
            [[0.9, 0.05], [0.05, 0.05], [0.05, 0.05], [0.05, 0.9]],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)

        pa = paralign_dp(
            ce.similarity_matrix, ce.src_emb, ce.tgt_emb,
            skip_penalty=-0.02, merge_penalty=-0.5,
        )
        pa_struct = _links_structure(pa)
        assert pa_struct == [([0], [0]), ([1], []), ([2], []), ([3], [1])]

    def test_vecalign_dense_deletion(self) -> None:
        """Vecalign dense (1:1 only) should delete the unmatched source."""
        sim = np.array(
            [[0.9, 0.05], [0.05, 0.05], [0.05, 0.9]],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)

        v0, v1, n0, n1 = prepare_vecalign_inputs(ce.src_emb, ce.tgt_emb, num_overlaps=1)
        links = vecalign_dense_dp(v0, v1, n0, n1, del_penalty=0.5)
        va_struct = _links_structure(links)

        assert va_struct == [([0], [0]), ([1], []), ([2], [1])]


# ---------------------------------------------------------------------------
# Scenario 4: Mixed (1:1 + merge + skip)
# ---------------------------------------------------------------------------


class TestMixed:
    """6 src, 5 tgt: 1:1 + 2:1 merge + skip in one alignment."""

    @pytest.fixture()
    def setup(self):
        # Design:
        # src[0] -> tgt[0]    (1:1, high sim)
        # src[1,2] -> tgt[1]  (2:1 merge)
        # src[3] -> skip       (1:0)
        # src[4] -> tgt[2]    (1:1)
        # src[5] -> tgt[3,4] is too greedy; instead:
        # src[5] -> tgt[3]    (1:1)
        # ??? -> tgt[4]        this doesn't work with 6 src, 5 tgt cleanly
        # Let me redesign:
        # src[0] -> tgt[0]    (1:1)
        # src[1,2] -> tgt[1]  (2:1)
        # src[3] -> skip       (1:0)
        # src[4] -> tgt[2]    (1:1)
        # src[5] -> tgt[3]    (1:1)
        #         -> tgt[4] is a 0:1 skip
        # Actually 5 tgt need to be covered. Let me make it:
        # src[0] -> tgt[0]    (1:1)
        # src[1,2] -> tgt[1]  (2:1)
        # src[3] -> skip       (1:0)
        # src[4] -> tgt[2]    (1:1)
        # src[5] -> tgt[3,4]  (1:2)
        sim = np.array(
            [
                # t0    t1    t2    t3    t4
                [0.9, 0.05, 0.05, 0.05, 0.05],  # s0 -> t0
                [0.05, 0.65, 0.05, 0.05, 0.05],  # s1 -> merge with s2 for t1
                [0.05, 0.65, 0.05, 0.05, 0.05],  # s2 -> merge with s1 for t1
                [0.05, 0.05, 0.05, 0.05, 0.05],  # s3 -> skip (no match)
                [0.05, 0.05, 0.9, 0.05, 0.05],  # s4 -> t2
                [0.05, 0.05, 0.05, 0.65, 0.65],  # s5 -> t3,t4
            ],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)
        return ce

    def test_paralign_mixed(self, setup) -> None:
        ce = setup
        pa = paralign_dp(
            ce.similarity_matrix, ce.src_emb, ce.tgt_emb,
            skip_penalty=-0.02, merge_penalty=-0.01,
        )
        pa_struct = _links_structure(pa)

        expected = [
            ([0], [0]),       # 1:1
            ([1, 2], [1]),    # 2:1
            ([3], []),        # skip
            ([4], [2]),       # 1:1
            ([5], [3, 4]),    # 1:2
        ]
        assert pa_struct == expected
        _assert_covers_all(pa, 6, 5)
        _assert_monotonic(pa)

    def test_bertalign_mixed(self, setup) -> None:
        ce = setup
        src_vecs, tgt_vecs = prepare_bertalign_inputs(ce.src_emb, ce.tgt_emb, max_align=5)
        ba = bertalign_second_pass_dp(
            src_vecs, tgt_vecs, 6, 5, max_align=5, skip=-0.02,
        )
        ba_struct = _links_structure(ba)

        expected = [
            ([0], [0]),
            ([1, 2], [1]),
            ([3], []),
            ([4], [2]),
            ([5], [3, 4]),
        ]
        assert ba_struct == expected
        _assert_covers_all(ba, 6, 5)
        _assert_monotonic(ba)


# ---------------------------------------------------------------------------
# Scenario 5: Algorithm disagreements
# ---------------------------------------------------------------------------


class TestAlgorithmDisagreements:
    """Cases where penalty settings cause different algorithms to make different choices."""

    def test_merge_vs_skip_tradeoff(self) -> None:
        """When similarity is moderate, paralign's higher skip penalty (-0.3) may
        prefer a merge while bertalign's lower skip penalty (-0.1) prefers skip+1:1.

        3 src, 2 tgt:
        src[0] has moderate match to tgt[0]
        src[1] has moderate match to tgt[0]
        src[2] has high match to tgt[1]

        With low skip cost: skip src[0], src[1]->tgt[0], src[2]->tgt[1]
        With high skip cost: src[0,1]->tgt[0] (merge), src[2]->tgt[1]
        """
        sim = np.array(
            [[0.4, 0.05], [0.5, 0.05], [0.05, 0.9]],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)

        # Paralign with heavy skip penalty: should merge
        pa_merge = paralign_dp(
            ce.similarity_matrix, ce.src_emb, ce.tgt_emb,
            skip_penalty=-0.5, merge_penalty=-0.01,
        )
        pa_struct = _links_structure(pa_merge)
        assert pa_struct == [([0, 1], [0]), ([2], [1])], f"Expected merge, got: {pa_struct}"

        # Paralign with light skip penalty: should skip
        pa_skip = paralign_dp(
            ce.similarity_matrix, ce.src_emb, ce.tgt_emb,
            skip_penalty=-0.01, merge_penalty=-0.5,
        )
        pa_struct_skip = _links_structure(pa_skip)
        assert pa_struct_skip == [([0], []), ([1], [0]), ([2], [1])], (
            f"Expected skip, got: {pa_struct_skip}"
        )

        # Both are valid alignments, just different tradeoffs
        _assert_covers_all(pa_merge, 3, 2)
        _assert_covers_all(pa_skip, 3, 2)

    def test_large_merge_vs_multiple_1to1(self) -> None:
        """When all similarities are moderate, high merge penalty favors 1:1."""
        sim = np.array(
            [
                [0.5, 0.05, 0.05, 0.05],
                [0.05, 0.5, 0.05, 0.05],
                [0.05, 0.05, 0.5, 0.05],
                [0.05, 0.05, 0.05, 0.5],
            ],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)

        # Low merge penalty: might merge
        pa_low = paralign_dp(
            ce.similarity_matrix, ce.src_emb, ce.tgt_emb,
            skip_penalty=-0.3, merge_penalty=-0.001,
        )
        # High merge penalty: should do 1:1
        pa_high = paralign_dp(
            ce.similarity_matrix, ce.src_emb, ce.tgt_emb,
            skip_penalty=-0.3, merge_penalty=-1.0,
        )
        expected_1to1 = [([i], [i]) for i in range(4)]
        assert _links_structure(pa_high) == expected_1to1

        _assert_covers_all(pa_low, 4, 4)
        _assert_covers_all(pa_high, 4, 4)


# ---------------------------------------------------------------------------
# Scenario 6: Parameter sensitivity
# ---------------------------------------------------------------------------


class TestParameterSensitivity:
    """Sweep parameters and observe decision flip points."""

    def test_paralign_skip_penalty_sweep(self) -> None:
        """Increasing skip penalty should eventually prevent skips."""
        sim = np.array(
            [[0.9, 0.05], [0.15, 0.15], [0.05, 0.9]],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)

        decisions: list[str] = []
        penalties = [-0.01, -0.05, -0.1, -0.3, -0.5, -1.0]

        for penalty in penalties:
            links = paralign_dp(
                ce.similarity_matrix, ce.src_emb, ce.tgt_emb,
                skip_penalty=penalty, merge_penalty=-0.01,
            )
            struct = _links_structure(links)
            # Classify: does src[1] get skipped, merged, or 1:1?
            for s, t in struct:
                if 1 in s and not t:
                    decisions.append("skip")
                    break
                if 1 in s and t and len(s) > 1:
                    decisions.append("merge")
                    break
                if s == [1] and t:
                    decisions.append("1:1")
                    break
            else:
                decisions.append("other")

        # With very low penalty, should skip; with very high, should merge or 1:1
        assert decisions[0] == "skip", f"Expected skip at penalty={penalties[0]}, got {decisions[0]}"
        # At some point it should stop skipping
        assert any(d != "skip" for d in decisions), "Should flip away from skip at high penalty"

    def test_bertalign_skip_sweep(self) -> None:
        """Bertalign skip cost sweep shows flip point."""
        sim = np.array(
            [[0.9, 0.05], [0.15, 0.15], [0.05, 0.9]],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)

        src_vecs, tgt_vecs = prepare_bertalign_inputs(ce.src_emb, ce.tgt_emb, max_align=5)
        decisions: list[str] = []
        skip_values = [-0.01, -0.05, -0.1, -0.3, -0.5]

        for skip_val in skip_values:
            links = bertalign_second_pass_dp(
                src_vecs, tgt_vecs, 3, 2, max_align=5, skip=skip_val,
            )
            struct = _links_structure(links)
            for s, t in struct:
                if 1 in s and not t:
                    decisions.append("skip")
                    break
                if 1 in s and t and len(s) > 1:
                    decisions.append("merge")
                    break
                if s == [1] and t:
                    decisions.append("1:1")
                    break
            else:
                decisions.append("other")

        assert decisions[0] == "skip"
        assert any(d != "skip" for d in decisions)

    def test_vecalign_del_penalty_sweep(self) -> None:
        """Vecalign deletion penalty sweep: low penalty deletes, high penalty forces match.

        Uses equal-sized inputs (3x3) so that deletion of src[1] also requires
        deleting a target, making high deletion penalties force matching.
        """
        sim = np.array(
            [[0.9, 0.05, 0.05], [0.05, 0.15, 0.05], [0.05, 0.05, 0.9]],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)

        v0, v1, n0, n1 = prepare_vecalign_inputs(ce.src_emb, ce.tgt_emb, num_overlaps=1)

        decisions: list[str] = []
        del_penalties = [0.01, 0.1, 0.3, 0.5, 1.0, 2.0]

        for dp_val in del_penalties:
            links = vecalign_dense_dp(v0, v1, n0, n1, del_penalty=dp_val)
            struct = _links_structure(links)
            for s, t in struct:
                if 1 in s and not t:
                    decisions.append("delete")
                    break
                if s == [1] and t:
                    decisions.append("1:1")
                    break
            else:
                decisions.append("other")

        # Low deletion penalty: easy to delete src[1] + tgt[1]
        assert decisions[0] == "delete", (
            f"Expected delete at low penalty={del_penalties[0]}, got {decisions[0]}"
        )
        # High deletion penalty: should force matching src[1]->tgt[1]
        assert decisions[-1] != "delete", (
            f"Expected non-delete at high penalty={del_penalties[-1]}, got {decisions[-1]}"
        )

    def test_merge_penalty_flip_point(self) -> None:
        """Find the paralign merge penalty value where it flips from merge to skip+1:1.

        With skip_penalty=-0.1, the skip path scores about 1.3 total.
        The merge path scores ~0.707 + merge_penalty + 0.9.
        Flip point is around merge_penalty ~ -0.3.
        """
        sim = np.array(
            [[0.5, 0.05], [0.5, 0.05], [0.05, 0.9]],
            dtype=np.float32,
        )
        ce = ConstantEmbedder(sim)

        merge_penalties = np.linspace(-0.001, -0.8, 30)
        merge_seen = False
        non_merge_seen = False

        for mp in merge_penalties:
            links = paralign_dp(
                ce.similarity_matrix, ce.src_emb, ce.tgt_emb,
                skip_penalty=-0.1, merge_penalty=float(mp),
            )
            struct = _links_structure(links)
            has_merge = any(len(s) > 1 or len(t) > 1 for s, t in struct if s and t)
            if has_merge:
                merge_seen = True
            else:
                non_merge_seen = True

        # Should see both merge and non-merge across the sweep
        assert merge_seen, "Expected at least one merge result across sweep"
        assert non_merge_seen, "Expected at least one non-merge result across sweep"


# ---------------------------------------------------------------------------
# Cross-algorithm invariant tests
# ---------------------------------------------------------------------------


class TestCrossAlgorithmInvariants:
    """Properties that should hold for all algorithms on any input."""

    @pytest.fixture(params=[
        "diagonal_5x5",
        "random_4x6",
        "random_8x8",
    ])
    def embeddings(self, request):
        rng = np.random.RandomState(42)
        if request.param == "diagonal_5x5":
            n = 5
            sim = np.full((n, n), 0.05, dtype=np.float32)
            np.fill_diagonal(sim, 0.9)
            ce = ConstantEmbedder(sim)
            return ce.src_emb, ce.tgt_emb, n, n
        elif request.param == "random_4x6":
            n_src, n_tgt = 4, 6
            # Random but with column norms < 1
            sim = rng.uniform(0.02, 0.3, (n_src, n_tgt)).astype(np.float32)
            np.fill_diagonal(sim[:min(n_src, n_tgt), :min(n_src, n_tgt)], 0.6)
            ce = ConstantEmbedder(sim)
            return ce.src_emb, ce.tgt_emb, n_src, n_tgt
        else:  # random_8x8
            n = 8
            sim = rng.uniform(0.02, 0.2, (n, n)).astype(np.float32)
            np.fill_diagonal(sim, 0.5)
            ce = ConstantEmbedder(sim)
            return ce.src_emb, ce.tgt_emb, n, n

    def test_full_coverage(self, embeddings) -> None:
        src_emb, tgt_emb, n_src, n_tgt = embeddings
        results = run_all_algorithms(src_emb, tgt_emb)
        for name, links in results.items():
            _assert_covers_all(links, n_src, n_tgt)

    def test_monotonicity(self, embeddings) -> None:
        src_emb, tgt_emb, n_src, n_tgt = embeddings
        results = run_all_algorithms(src_emb, tgt_emb)
        for name, links in results.items():
            _assert_monotonic(links)
