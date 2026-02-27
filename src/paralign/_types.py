"""Core types for paralign: AlignmentType, AlignmentPair, AlignmentResult."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np


class AlignmentType(enum.Enum):
    """Type of alignment between source and target sentence groups."""

    ONE_TO_ONE = "1:1"
    ONE_TO_TWO = "1:2"
    TWO_TO_ONE = "2:1"
    TWO_TO_TWO = "2:2"
    ONE_TO_ZERO = "1:0"
    ZERO_TO_ONE = "0:1"

    @staticmethod
    def from_counts(src: int, tgt: int) -> AlignmentType:
        """Create AlignmentType from source and target counts."""
        _map = {
            (1, 1): AlignmentType.ONE_TO_ONE,
            (1, 2): AlignmentType.ONE_TO_TWO,
            (2, 1): AlignmentType.TWO_TO_ONE,
            (2, 2): AlignmentType.TWO_TO_TWO,
            (1, 0): AlignmentType.ONE_TO_ZERO,
            (0, 1): AlignmentType.ZERO_TO_ONE,
        }
        result = _map.get((src, tgt))
        if result is None:
            raise ValueError(f"Unsupported alignment counts: ({src}, {tgt})")
        return result


@dataclass(frozen=True)
class AlignmentPair:
    """A single aligned pair of sentence groups."""

    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    alignment_type: AlignmentType
    score: float

    @property
    def source_count(self) -> int:
        return len(self.source_indices)

    @property
    def target_count(self) -> int:
        return len(self.target_indices)


@dataclass(frozen=True)
class AlignmentResult:
    """Complete alignment result with pairs, sentences, and optional similarity matrix."""

    pairs: tuple[AlignmentPair, ...]
    source_sentences: tuple[str, ...]
    target_sentences: tuple[str, ...]
    similarity_matrix: np.ndarray | None

    @property
    def total_score(self) -> float:
        return sum(p.score for p in self.pairs)

    @property
    def source_count(self) -> int:
        return len(self.source_sentences)

    @property
    def target_count(self) -> int:
        return len(self.target_sentences)

    def merged_pairs(self, separator: str = " ") -> list[tuple[str, str]]:
        """Return aligned pairs with merged sentence text.

        For N:M alignments, sentences are joined with the separator.
        For skip alignments, the skipped side is an empty string.
        """
        result: list[tuple[str, str]] = []
        for pair in self.pairs:
            src_text = separator.join(
                self.source_sentences[i] for i in pair.source_indices
            )
            tgt_text = separator.join(
                self.target_sentences[i] for i in pair.target_indices
            )
            result.append((src_text, tgt_text))
        return result
