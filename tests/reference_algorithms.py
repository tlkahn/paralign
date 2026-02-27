"""Pure-Python reference reimplementations of bertalign and vecalign DP algorithms.

These are simplified versions for comparison testing on small inputs.
No numba/faiss/Cython required.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from paralign._dp import DPConfig, dp_align
from paralign._similarity import cosine_similarity_matrix


# ---------------------------------------------------------------------------
# Common output format
# ---------------------------------------------------------------------------

class AlignmentLink(NamedTuple):
    """Normalized alignment output: lists of source and target indices."""

    src: list[int]
    tgt: list[int]


# ---------------------------------------------------------------------------
# Bertalign: second-pass DP
# ---------------------------------------------------------------------------


def _get_bertalign_alignment_types(max_align: int) -> list[tuple[int, int]]:
    """Return bertalign transition types.

    Order: (0,1), (1,0), then all (x,y) with 1<=x,y and x+y<=max_align.
    """
    types: list[tuple[int, int]] = [(0, 1), (1, 0)]
    for x in range(1, max_align):
        for y in range(1, max_align):
            if x + y <= max_align:
                types.append((x, y))
    return types


def prepare_bertalign_inputs(
    src_emb: np.ndarray,
    tgt_emb: np.ndarray,
    max_align: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build bertalign-style 3D overlap embedding arrays.

    Returns:
        src_vecs: shape (max_align-1, n_src, dim)
            src_vecs[k-1, i] = normalized mean of src_emb[i:i+k]
        tgt_vecs: shape (max_align-1, n_tgt, dim)
            tgt_vecs[k-1, j] = normalized mean of tgt_emb[j:j+k]

    Padding positions (where i+k > n) are zero vectors.
    """
    n_src, dim = src_emb.shape
    n_tgt = tgt_emb.shape[0]
    num_overlaps = max_align - 1

    src_vecs = np.zeros((num_overlaps, n_src, dim), dtype=np.float64)
    tgt_vecs = np.zeros((num_overlaps, n_tgt, dim), dtype=np.float64)

    for k in range(1, num_overlaps + 1):
        for i in range(n_src - k + 1):
            v = np.mean(src_emb[i : i + k], axis=0).astype(np.float64)
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v /= norm
            src_vecs[k - 1, i] = v

        for j in range(n_tgt - k + 1):
            v = np.mean(tgt_emb[j : j + k], axis=0).astype(np.float64)
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v /= norm
            tgt_vecs[k - 1, j] = v

    return src_vecs, tgt_vecs


def bertalign_second_pass_dp(
    src_vecs: np.ndarray,
    tgt_vecs: np.ndarray,
    n_src: int,
    n_tgt: int,
    max_align: int = 5,
    skip: float = -0.1,
) -> list[AlignmentLink]:
    """Bertalign second-pass DP (full table, no search-path windowing).

    Maximizes cumulative dot-product score. Transitions up to x+y <= max_align.
    Skip (0:1 or 1:0) costs ``skip``.

    Args:
        src_vecs: Overlap embeddings, shape (max_align-1, n_src, dim).
        tgt_vecs: Overlap embeddings, shape (max_align-1, n_tgt, dim).
        n_src: Number of source sentences.
        n_tgt: Number of target sentences.
        max_align: Maximum alignment group size (x + y).
        skip: Flat penalty for deletion/insertion transitions.

    Returns:
        List of AlignmentLink in forward order.
    """
    a_types = _get_bertalign_alignment_types(max_align)
    neg_inf = float("-inf")

    cost = np.full((n_src + 1, n_tgt + 1), neg_inf, dtype=np.float64)
    cost[0, 0] = 0.0
    pointers = np.full((n_src + 1, n_tgt + 1), -1, dtype=np.int32)

    for i in range(n_src + 1):
        for j in range(n_tgt + 1):
            if i == 0 and j == 0:
                continue

            best_score = neg_inf
            best_a = -1

            for a_idx, (a1, a2) in enumerate(a_types):
                pi = i - a1
                pj = j - a2
                if pi < 0 or pj < 0:
                    continue
                if cost[pi, pj] == neg_inf:
                    continue

                if a1 == 0 or a2 == 0:
                    cur_score = skip
                else:
                    # Dot product of overlap embeddings
                    # src_vecs[a1-1, pi] is the embedding for a1 sentences starting at pi
                    sv = src_vecs[a1 - 1, pi]
                    tv = tgt_vecs[a2 - 1, pj]
                    cur_score = float(np.dot(sv, tv))

                score = cost[pi, pj] + cur_score
                if score > best_score:
                    best_score = score
                    best_a = a_idx

            if best_score > neg_inf:
                cost[i, j] = best_score
                pointers[i, j] = best_a

    # Backtrace
    alignment: list[AlignmentLink] = []
    i, j = n_src, n_tgt
    while i > 0 or j > 0:
        a_idx = int(pointers[i, j])
        if a_idx < 0:
            break
        a1, a2 = a_types[a_idx]
        pi, pj = i - a1, j - a2

        src_range = list(range(pi, i))
        tgt_range = list(range(pj, j))
        alignment.append(AlignmentLink(src=src_range, tgt=tgt_range))

        i, j = pi, pj

    alignment.reverse()
    return alignment


# ---------------------------------------------------------------------------
# Vecalign: dense DP (1:1 + deletions, minimizes cost)
# ---------------------------------------------------------------------------


def prepare_vecalign_inputs(
    src_emb: np.ndarray,
    tgt_emb: np.ndarray,
    num_overlaps: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build vecalign-style overlap arrays and normalization factors.

    For the dense variant, only 1:1 overlaps are needed (num_overlaps=1).
    For sparse, set num_overlaps higher for merge support.

    Returns:
        vecs0: shape (num_overlaps, n_src, dim), normalized
        vecs1: shape (num_overlaps, n_tgt, dim), normalized
        norms0: shape (num_overlaps, n_src) — average (1-cosine) to random tgt samples
        norms1: shape (num_overlaps, n_tgt) — average (1-cosine) to random src samples
    """
    n_src, dim = src_emb.shape
    n_tgt = tgt_emb.shape[0]

    vecs0 = np.zeros((num_overlaps, n_src, dim), dtype=np.float64)
    vecs1 = np.zeros((num_overlaps, n_tgt, dim), dtype=np.float64)

    for k in range(1, num_overlaps + 1):
        for i in range(n_src - k + 1):
            v = np.mean(src_emb[i : i + k], axis=0).astype(np.float64)
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v /= norm
            vecs0[k - 1, i] = v

        for j in range(n_tgt - k + 1):
            v = np.mean(tgt_emb[j : j + k], axis=0).astype(np.float64)
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v /= norm
            vecs1[k - 1, j] = v

    # Compute norms: average (1 - cosine_similarity) to random samples from the other side
    norms0 = _compute_norms(vecs0, vecs1)
    norms1 = _compute_norms(vecs1, vecs0)

    return vecs0, vecs1, norms0, norms1


def _compute_norms(
    vecs_a: np.ndarray, vecs_b: np.ndarray, num_samples: int = 100
) -> np.ndarray:
    """Compute vecalign normalization factors.

    norms[k, i] = 1 - mean(dot(vecs_a[k, i], random vecs_b samples))
    """
    overlaps_a, size_a, dim = vecs_a.shape
    overlaps_b, size_b, _ = vecs_b.shape

    rng = np.random.RandomState(42)
    norms = np.zeros((overlaps_a, size_a), dtype=np.float64)

    # Pool all vecs_b entries
    pool: list[np.ndarray] = []
    for k in range(overlaps_b):
        for j in range(size_b):
            v = vecs_b[k, j]
            if float(np.linalg.norm(v)) > 0:
                pool.append(v)
    if not pool:
        return np.ones((overlaps_a, size_a), dtype=np.float64)

    pool_arr = np.array(pool)  # (pool_size, dim)
    sample_indices = rng.choice(len(pool_arr), size=min(num_samples, len(pool_arr)), replace=True)
    samples = pool_arr[sample_indices]  # (num_samples, dim)

    for k in range(overlaps_a):
        for i in range(size_a):
            v = vecs_a[k, i]
            if float(np.linalg.norm(v)) == 0:
                norms[k, i] = 1.0
                continue
            sims = samples @ v  # dot products
            norms[k, i] = 1.0 - float(np.mean(sims))

    return norms


def vecalign_dense_dp(
    vecs0: np.ndarray,
    vecs1: np.ndarray,
    norms0: np.ndarray,
    norms1: np.ndarray,
    del_penalty: float,
) -> list[AlignmentLink]:
    """Vecalign dense DP: 1:1 + deletions only, minimizes cost.

    Cost for 1:1 match of (i, j):
        2 * (1 - dot(vecs0[0,i], vecs1[0,j])) / (norms0[0,i] + norms1[0,j] + 1e-6)

    Args:
        vecs0: Source overlap embeddings, shape (num_overlaps, n_src, dim).
        vecs1: Target overlap embeddings, shape (num_overlaps, n_tgt, dim).
        norms0: Source normalization, shape (num_overlaps, n_src).
        norms1: Target normalization, shape (num_overlaps, n_tgt).
        del_penalty: Deletion penalty cost.

    Returns:
        List of AlignmentLink in forward order.
    """
    n_src = vecs0.shape[1]
    n_tgt = vecs1.shape[1]

    # Precompute alignment costs for 1:1
    alignment_cost = np.zeros((n_src, n_tgt), dtype=np.float64)
    for i in range(n_src):
        for j in range(n_tgt):
            dot = float(np.dot(vecs0[0, i], vecs1[0, j]))
            n0 = norms0[0, i]
            n1 = norms1[0, j]
            alignment_cost[i, j] = 2.0 * (1.0 - dot) / (n0 + n1 + 1e-6)

    # DP table (minimizing cost)
    inf = float("inf")
    csum = np.full((n_src + 1, n_tgt + 1), inf, dtype=np.float64)
    bp = np.full((n_src + 1, n_tgt + 1), -1, dtype=np.int32)

    csum[0, 0] = 0.0
    for c in range(1, n_tgt + 1):
        csum[0, c] = c * del_penalty
        bp[0, c] = 1  # horizontal (target deletion)
    for r in range(1, n_src + 1):
        csum[r, 0] = r * del_penalty
        bp[r, 0] = 2  # vertical (source deletion)

    for r in range(1, n_src + 1):
        for c in range(1, n_tgt + 1):
            # Diagonal: 1:1 match
            cost0 = csum[r - 1, c - 1] + alignment_cost[r - 1, c - 1]
            # Horizontal: target deletion
            cost1 = csum[r, c - 1] + del_penalty
            # Vertical: source deletion
            cost2 = csum[r - 1, c] + del_penalty

            if cost0 <= cost1 and cost0 <= cost2:
                csum[r, c] = cost0
                bp[r, c] = 0
            elif cost1 <= cost2:
                csum[r, c] = cost1
                bp[r, c] = 1
            else:
                csum[r, c] = cost2
                bp[r, c] = 2

    # Backtrace
    alignment: list[AlignmentLink] = []
    r, c = n_src, n_tgt
    while r > 0 or c > 0:
        b = int(bp[r, c])
        if b == 0:  # diagonal
            alignment.append(AlignmentLink(src=[r - 1], tgt=[c - 1]))
            r, c = r - 1, c - 1
        elif b == 1:  # horizontal (target deletion)
            alignment.append(AlignmentLink(src=[], tgt=[c - 1]))
            c -= 1
        elif b == 2:  # vertical (source deletion)
            alignment.append(AlignmentLink(src=[r - 1], tgt=[]))
            r -= 1
        else:
            break

    alignment.reverse()
    return alignment


def vecalign_sparse_dp(
    vecs0: np.ndarray,
    vecs1: np.ndarray,
    norms0: np.ndarray,
    norms1: np.ndarray,
    del_penalty: float,
    max_align: int = 4,
) -> list[AlignmentLink]:
    """Vecalign sparse DP: arbitrary alignment types, minimizes cost.

    Cost for x:y match: 2·x·y·(1-dot)/(norms0+norms1+1e-6), where dot uses
    overlap embeddings for x consecutive source and y consecutive target sentences.

    Args:
        vecs0: Source overlap embeddings, shape (num_overlaps, n_src, dim).
        vecs1: Target overlap embeddings, shape (num_overlaps, n_tgt, dim).
        norms0: Source normalization, shape (num_overlaps, n_src).
        norms1: Target normalization, shape (num_overlaps, n_tgt).
        del_penalty: Deletion penalty cost.
        max_align: Maximum alignment group size (x + y).

    Returns:
        List of AlignmentLink in forward order.
    """
    n_src = vecs0.shape[1]
    n_tgt = vecs1.shape[1]
    num_overlaps = vecs0.shape[0]

    # Build alignment types: all (x,y) where 1<=x,y and x+y<=max_align, plus deletions
    a_types: list[tuple[int, int]] = []
    for x in range(1, max_align):
        for y in range(1, max_align):
            if x + y <= max_align:
                if x <= num_overlaps and y <= num_overlaps:
                    a_types.append((x, y))
    # Add deletions
    a_types.append((1, 0))  # source deletion
    a_types.append((0, 1))  # target deletion

    inf = float("inf")
    csum = np.full((n_src + 1, n_tgt + 1), inf, dtype=np.float64)
    bp = np.full((n_src + 1, n_tgt + 1, 2), -1, dtype=np.int32)  # store (x, y)

    csum[0, 0] = 0.0
    for c in range(1, n_tgt + 1):
        csum[0, c] = c * del_penalty
        bp[0, c] = [0, 1]
    for r in range(1, n_src + 1):
        csum[r, 0] = r * del_penalty
        bp[r, 0] = [1, 0]

    for r in range(1, n_src + 1):
        for c in range(1, n_tgt + 1):
            best_cost = inf
            best_trans = (-1, -1)

            for x, y in a_types:
                pr = r - x
                pc = c - y
                if pr < 0 or pc < 0:
                    continue
                if csum[pr, pc] == inf:
                    continue

                if x == 0 or y == 0:
                    trans_cost = del_penalty
                else:
                    # Dot product using overlap embeddings
                    sv = vecs0[x - 1, pr]
                    tv = vecs1[y - 1, pc]
                    dot = float(np.dot(sv, tv))
                    n0 = norms0[x - 1, pr]
                    n1 = norms1[y - 1, pc]
                    trans_cost = 2.0 * x * y * (1.0 - dot) / (n0 + n1 + 1e-6)

                total = csum[pr, pc] + trans_cost
                if total < best_cost:
                    best_cost = total
                    best_trans = (x, y)

            if best_cost < inf:
                csum[r, c] = best_cost
                bp[r, c] = [best_trans[0], best_trans[1]]

    # Backtrace
    alignment: list[AlignmentLink] = []
    r, c = n_src, n_tgt
    while r > 0 or c > 0:
        x, y = int(bp[r, c, 0]), int(bp[r, c, 1])
        if x < 0 and y < 0:
            break
        pr, pc = r - x, c - y
        alignment.append(
            AlignmentLink(src=list(range(pr, r)), tgt=list(range(pc, c)))
        )
        r, c = pr, pc

    alignment.reverse()
    return alignment


# ---------------------------------------------------------------------------
# Paralign adapter
# ---------------------------------------------------------------------------


def paralign_dp(
    sim: np.ndarray,
    src_emb: np.ndarray,
    tgt_emb: np.ndarray,
    skip_penalty: float = -0.3,
    merge_penalty: float = -0.05,
) -> list[AlignmentLink]:
    """Run paralign DP and return results in normalized AlignmentLink format."""
    config = DPConfig(skip_penalty=skip_penalty, merge_penalty=merge_penalty, band_width=200)
    pairs = dp_align(sim, src_emb, tgt_emb, config)
    return [
        AlignmentLink(src=list(p.source_indices), tgt=list(p.target_indices))
        for p in pairs
    ]


# ---------------------------------------------------------------------------
# Convenience: run all algorithms
# ---------------------------------------------------------------------------


def run_all_algorithms(
    src_emb: np.ndarray,
    tgt_emb: np.ndarray,
    *,
    # paralign params
    paralign_skip: float = -0.3,
    paralign_merge: float = -0.05,
    # bertalign params
    bertalign_max_align: int = 5,
    bertalign_skip: float = -0.1,
    # vecalign params
    vecalign_del_penalty: float | None = None,
    vecalign_max_align: int = 4,
) -> dict[str, list[AlignmentLink]]:
    """Run paralign, bertalign, and vecalign on the same embeddings.

    Args:
        src_emb: Source embeddings, shape (n_src, dim). Expected to be normalized.
        tgt_emb: Target embeddings, shape (n_tgt, dim). Expected to be normalized.
        Various algorithm-specific parameters.

    Returns:
        Dict mapping algorithm name to list of AlignmentLink.
    """
    sim = cosine_similarity_matrix(src_emb, tgt_emb)

    # Paralign
    pa = paralign_dp(sim, src_emb, tgt_emb, paralign_skip, paralign_merge)

    # Bertalign
    max_overlap_b = bertalign_max_align - 1
    src_vecs, tgt_vecs = prepare_bertalign_inputs(src_emb, tgt_emb, bertalign_max_align)
    ba = bertalign_second_pass_dp(
        src_vecs, tgt_vecs, src_emb.shape[0], tgt_emb.shape[0],
        max_align=bertalign_max_align, skip=bertalign_skip,
    )

    # Vecalign
    max_overlap_v = max(vecalign_max_align - 1, 1)
    v0, v1, n0, n1 = prepare_vecalign_inputs(src_emb, tgt_emb, num_overlaps=max_overlap_v)
    if vecalign_del_penalty is None:
        vecalign_del_penalty = _estimate_vecalign_del_penalty(v0, v1, n0, n1)
    va = vecalign_sparse_dp(v0, v1, n0, n1, vecalign_del_penalty, vecalign_max_align)

    return {"paralign": pa, "bertalign": ba, "vecalign": va}


def _estimate_vecalign_del_penalty(
    vecs0: np.ndarray,
    vecs1: np.ndarray,
    norms0: np.ndarray,
    norms1: np.ndarray,
    percentile: float = 0.2,
    num_samples: int = 200,
) -> float:
    """Estimate vecalign deletion penalty from random alignment cost samples.

    Samples random (i, j) pairs, computes cost, and returns the given percentile.
    """
    n_src = vecs0.shape[1]
    n_tgt = vecs1.shape[1]
    rng = np.random.RandomState(42)

    actual_samples = min(num_samples, n_src * n_tgt)
    if n_src * n_tgt <= num_samples:
        # Enumerate all pairs
        costs = []
        for i in range(n_src):
            for j in range(n_tgt):
                dot = float(np.dot(vecs0[0, i], vecs1[0, j]))
                n0 = norms0[0, i]
                n1 = norms1[0, j]
                c = 2.0 * (1.0 - dot) / (n0 + n1 + 1e-6)
                costs.append(c)
    else:
        x_idxs = rng.choice(n_src, size=actual_samples, replace=True)
        y_idxs = rng.choice(n_tgt, size=actual_samples, replace=True)
        costs = []
        for xi, yi in zip(x_idxs, y_idxs):
            dot = float(np.dot(vecs0[0, xi], vecs1[0, yi]))
            n0 = norms0[0, xi]
            n1 = norms1[0, yi]
            c = 2.0 * (1.0 - dot) / (n0 + n1 + 1e-6)
            costs.append(c)

    return float(np.percentile(costs, percentile * 100))
