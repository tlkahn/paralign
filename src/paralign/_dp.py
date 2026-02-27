"""DP-based sentence alignment algorithm (Bertalign-style single-pass)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paralign._types import AlignmentPair, AlignmentType

# Transition types: (source_consumed, target_consumed)
TRANSITIONS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 0),
    (0, 1),
)


@dataclass(frozen=True)
class DPConfig:
    """Configuration for DP alignment."""

    skip_penalty: float = -0.3
    band_width: int = 50
    merge_penalty: float = -0.05


def _merged_similarity(
    sim: np.ndarray,
    src_emb: np.ndarray,
    tgt_emb: np.ndarray,
    src_start: int,
    src_end: int,
    tgt_start: int,
    tgt_end: int,
) -> float:
    """Compute similarity for a merged group of sentences.

    For 1:1, returns sim[src_start, tgt_start] directly.
    For N:M, averages the source and target embeddings and computes cosine similarity.
    """
    src_count = src_end - src_start
    tgt_count = tgt_end - tgt_start

    if src_count == 1 and tgt_count == 1:
        return float(sim[src_start, tgt_start])

    # Average source embeddings
    src_avg = np.mean(src_emb[src_start:src_end], axis=0)
    # Average target embeddings
    tgt_avg = np.mean(tgt_emb[tgt_start:tgt_end], axis=0)

    # Cosine similarity
    src_norm = float(np.linalg.norm(src_avg))
    tgt_norm = float(np.linalg.norm(tgt_avg))
    if src_norm == 0 or tgt_norm == 0:
        return 0.0
    return float(np.dot(src_avg, tgt_avg) / (src_norm * tgt_norm))


def dp_align(
    sim: np.ndarray,
    src_emb: np.ndarray,
    tgt_emb: np.ndarray,
    config: DPConfig,
) -> list[AlignmentPair]:
    """Find optimal monotonic alignment using dynamic programming.

    States: dp[i][j] = max cumulative score to align source[0..i) with target[0..j).
    Transitions: (1,1), (1,2), (2,1), (2,2), (1,0), (0,1).

    Args:
        sim: Similarity matrix of shape (n_src, n_tgt).
        src_emb: Source embeddings of shape (n_src, dim).
        tgt_emb: Target embeddings of shape (n_tgt, dim).
        config: DP configuration.

    Returns:
        List of AlignmentPair objects in monotonic order.
    """
    n_src, n_tgt = sim.shape

    if n_src == 0 and n_tgt == 0:
        return []
    if n_src == 0:
        return [
            AlignmentPair(
                source_indices=(),
                target_indices=(j,),
                alignment_type=AlignmentType.ZERO_TO_ONE,
                score=0.0,
            )
            for j in range(n_tgt)
        ]
    if n_tgt == 0:
        return [
            AlignmentPair(
                source_indices=(i,),
                target_indices=(),
                alignment_type=AlignmentType.ONE_TO_ZERO,
                score=0.0,
            )
            for i in range(n_src)
        ]

    neg_inf = float("-inf")

    # dp[i][j] = best score to have consumed source[0..i) and target[0..j)
    dp = np.full((n_src + 1, n_tgt + 1), neg_inf, dtype=np.float64)
    dp[0, 0] = 0.0

    # Backtrack: store (src_consumed, tgt_consumed) at each cell
    bt: list[list[tuple[int, int]]] = [
        [(0, 0) for _ in range(n_tgt + 1)] for _ in range(n_src + 1)
    ]

    # Precompute band limits
    scale = n_tgt / n_src if n_src > 0 else 1.0

    for i in range(n_src + 1):
        for j in range(n_tgt + 1):
            if i == 0 and j == 0:
                continue

            # Band pruning: check if (i, j) is within band
            expected_j = i * scale
            if abs(j - expected_j) > config.band_width + 2:
                continue

            best_score = neg_inf
            best_trans = (0, 0)

            for ds, dt in TRANSITIONS:
                pi = i - ds
                pj = j - dt
                if pi < 0 or pj < 0:
                    continue
                if dp[pi, pj] == neg_inf:
                    continue

                # Compute transition score
                if ds == 0 and dt == 0:
                    continue

                if ds == 0:
                    # 0:1 skip
                    trans_score = config.skip_penalty
                elif dt == 0:
                    # 1:0 skip
                    trans_score = config.skip_penalty
                else:
                    # N:M match
                    trans_score = _merged_similarity(
                        sim, src_emb, tgt_emb, pi, i, pj, j
                    )
                    # Apply merge penalty for non-1:1 transitions
                    if ds > 1 or dt > 1:
                        trans_score += config.merge_penalty

                candidate = float(dp[pi, pj]) + trans_score
                if candidate > best_score:
                    best_score = candidate
                    best_trans = (ds, dt)

            if best_score > neg_inf:
                dp[i, j] = best_score
                bt[i][j] = best_trans

    # Backtrace
    pairs: list[AlignmentPair] = []
    i, j = n_src, n_tgt
    while i > 0 or j > 0:
        ds, dt = bt[i][j]
        if ds == 0 and dt == 0:
            # Shouldn't happen if DP is correct, but handle gracefully
            break

        pi = i - ds
        pj = j - dt

        src_indices = tuple(range(pi, i))
        tgt_indices = tuple(range(pj, j))

        alignment_type = AlignmentType.from_counts(ds, dt)

        if ds == 0 or dt == 0:
            score = 0.0
        else:
            score = _merged_similarity(sim, src_emb, tgt_emb, pi, i, pj, j)

        pairs.append(
            AlignmentPair(
                source_indices=src_indices,
                target_indices=tgt_indices,
                alignment_type=alignment_type,
                score=score,
            )
        )

        i, j = pi, pj

    pairs.reverse()
    return pairs
