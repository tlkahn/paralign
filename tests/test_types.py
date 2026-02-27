"""Tests for paralign types: AlignmentType, AlignmentPair, AlignmentResult."""

from __future__ import annotations

import numpy as np
import pytest

from paralign._types import AlignmentPair, AlignmentResult, AlignmentType


class TestAlignmentType:
    """Tests for AlignmentType enum."""

    def test_enum_members_exist(self) -> None:
        assert AlignmentType.ONE_TO_ONE is not None
        assert AlignmentType.ONE_TO_TWO is not None
        assert AlignmentType.TWO_TO_ONE is not None
        assert AlignmentType.TWO_TO_TWO is not None
        assert AlignmentType.ONE_TO_ZERO is not None
        assert AlignmentType.ZERO_TO_ONE is not None

    def test_all_members_count(self) -> None:
        assert len(AlignmentType) == 6

    def test_from_counts_one_to_one(self) -> None:
        assert AlignmentType.from_counts(1, 1) == AlignmentType.ONE_TO_ONE

    def test_from_counts_one_to_two(self) -> None:
        assert AlignmentType.from_counts(1, 2) == AlignmentType.ONE_TO_TWO

    def test_from_counts_two_to_one(self) -> None:
        assert AlignmentType.from_counts(2, 1) == AlignmentType.TWO_TO_ONE

    def test_from_counts_two_to_two(self) -> None:
        assert AlignmentType.from_counts(2, 2) == AlignmentType.TWO_TO_TWO

    def test_from_counts_one_to_zero(self) -> None:
        assert AlignmentType.from_counts(1, 0) == AlignmentType.ONE_TO_ZERO

    def test_from_counts_zero_to_one(self) -> None:
        assert AlignmentType.from_counts(0, 1) == AlignmentType.ZERO_TO_ONE

    def test_from_counts_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            AlignmentType.from_counts(3, 1)

    def test_from_counts_zero_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            AlignmentType.from_counts(0, 0)


class TestAlignmentPair:
    """Tests for AlignmentPair frozen dataclass."""

    def test_creation(self) -> None:
        pair = AlignmentPair(
            source_indices=(0,),
            target_indices=(0,),
            alignment_type=AlignmentType.ONE_TO_ONE,
            score=0.95,
        )
        assert pair.source_indices == (0,)
        assert pair.target_indices == (0,)
        assert pair.alignment_type == AlignmentType.ONE_TO_ONE
        assert pair.score == 0.95

    def test_frozen(self) -> None:
        pair = AlignmentPair(
            source_indices=(0,),
            target_indices=(0,),
            alignment_type=AlignmentType.ONE_TO_ONE,
            score=0.9,
        )
        with pytest.raises(AttributeError):
            pair.score = 0.5  # type: ignore[misc]

    def test_source_count(self) -> None:
        pair = AlignmentPair(
            source_indices=(0, 1),
            target_indices=(2,),
            alignment_type=AlignmentType.TWO_TO_ONE,
            score=0.8,
        )
        assert pair.source_count == 2

    def test_target_count(self) -> None:
        pair = AlignmentPair(
            source_indices=(0,),
            target_indices=(1, 2),
            alignment_type=AlignmentType.ONE_TO_TWO,
            score=0.7,
        )
        assert pair.target_count == 2

    def test_skip_pair(self) -> None:
        pair = AlignmentPair(
            source_indices=(3,),
            target_indices=(),
            alignment_type=AlignmentType.ONE_TO_ZERO,
            score=0.0,
        )
        assert pair.source_count == 1
        assert pair.target_count == 0


class TestAlignmentResult:
    """Tests for AlignmentResult frozen dataclass."""

    def test_creation(self) -> None:
        pairs = [
            AlignmentPair((0,), (0,), AlignmentType.ONE_TO_ONE, 0.9),
            AlignmentPair((1,), (1,), AlignmentType.ONE_TO_ONE, 0.8),
        ]
        sim = np.eye(2, dtype=np.float32)
        result = AlignmentResult(
            pairs=tuple(pairs),
            source_sentences=("Hello", "World"),
            target_sentences=("Hola", "Mundo"),
            similarity_matrix=sim,
        )
        assert len(result.pairs) == 2
        assert result.source_sentences == ("Hello", "World")

    def test_frozen(self) -> None:
        result = AlignmentResult(
            pairs=(),
            source_sentences=(),
            target_sentences=(),
            similarity_matrix=None,
        )
        with pytest.raises(AttributeError):
            result.pairs = ()  # type: ignore[misc]

    def test_total_score(self) -> None:
        pairs = (
            AlignmentPair((0,), (0,), AlignmentType.ONE_TO_ONE, 0.9),
            AlignmentPair((1,), (1,), AlignmentType.ONE_TO_ONE, 0.8),
            AlignmentPair((2,), (2,), AlignmentType.ONE_TO_ONE, 0.7),
        )
        result = AlignmentResult(
            pairs=pairs,
            source_sentences=("a", "b", "c"),
            target_sentences=("x", "y", "z"),
            similarity_matrix=None,
        )
        assert abs(result.total_score - 2.4) < 1e-6

    def test_total_score_empty(self) -> None:
        result = AlignmentResult(
            pairs=(),
            source_sentences=(),
            target_sentences=(),
            similarity_matrix=None,
        )
        assert result.total_score == 0.0

    def test_source_count(self) -> None:
        result = AlignmentResult(
            pairs=(),
            source_sentences=("a", "b", "c"),
            target_sentences=("x",),
            similarity_matrix=None,
        )
        assert result.source_count == 3

    def test_target_count(self) -> None:
        result = AlignmentResult(
            pairs=(),
            source_sentences=("a",),
            target_sentences=("x", "y"),
            similarity_matrix=None,
        )
        assert result.target_count == 2

    def test_merged_pairs(self) -> None:
        """merged_pairs returns (source_text, target_text) with merged sentences."""
        pairs = (
            AlignmentPair((0,), (0,), AlignmentType.ONE_TO_ONE, 0.9),
            AlignmentPair((1, 2), (1,), AlignmentType.TWO_TO_ONE, 0.8),
            AlignmentPair((3,), (2, 3), AlignmentType.ONE_TO_TWO, 0.7),
        )
        result = AlignmentResult(
            pairs=pairs,
            source_sentences=("A", "B", "C", "D"),
            target_sentences=("X", "Y", "Z", "W"),
            similarity_matrix=None,
        )
        merged = result.merged_pairs()
        assert merged == [
            ("A", "X"),
            ("B C", "Y"),
            ("D", "Z W"),
        ]

    def test_merged_pairs_with_skip(self) -> None:
        """Skip alignments produce empty string on skipped side."""
        pairs = (
            AlignmentPair((0,), (), AlignmentType.ONE_TO_ZERO, 0.0),
            AlignmentPair((), (0,), AlignmentType.ZERO_TO_ONE, 0.0),
        )
        result = AlignmentResult(
            pairs=pairs,
            source_sentences=("A",),
            target_sentences=("X",),
            similarity_matrix=None,
        )
        merged = result.merged_pairs()
        assert merged == [("A", ""), ("", "X")]
