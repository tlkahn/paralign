"""Vectorized cosine similarity matrix computation with windowing."""

from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(src_emb: np.ndarray, tgt_emb: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between source and target embeddings.

    Args:
        src_emb: Source embeddings, shape (n_src, dim).
        tgt_emb: Target embeddings, shape (n_tgt, dim).

    Returns:
        Similarity matrix of shape (n_src, n_tgt) with values in [-1, 1].
    """
    # Normalize
    src_norm = np.linalg.norm(src_emb, axis=1, keepdims=True)
    tgt_norm = np.linalg.norm(tgt_emb, axis=1, keepdims=True)
    src_norm = np.where(src_norm == 0, 1.0, src_norm)
    tgt_norm = np.where(tgt_norm == 0, 1.0, tgt_norm)

    src_normed = src_emb / src_norm
    tgt_normed = tgt_emb / tgt_norm

    result: np.ndarray = np.dot(src_normed, tgt_normed.T).astype(np.float32)
    return result


def apply_window_mask(sim: np.ndarray, window: int) -> np.ndarray:
    """Apply diagonal band mask to similarity matrix.

    Zeros out cells where the column is more than `window` positions away from
    the expected diagonal position (scaled for non-square matrices).

    Args:
        sim: Similarity matrix of shape (n_src, n_tgt).
        window: Half-width of the diagonal band.

    Returns:
        Masked similarity matrix (copy).
    """
    n_src, n_tgt = sim.shape
    if n_src == 0 or n_tgt == 0:
        out: np.ndarray = sim.copy()
        return out

    rows = np.arange(n_src)[:, None]
    cols = np.arange(n_tgt)[None, :]

    # Scale factor: expected column for row i
    scale = n_tgt / n_src
    expected_col = rows * scale
    mask = np.abs(cols - expected_col) <= window

    result: np.ndarray = sim.copy()
    result[~mask] = 0.0
    return result


def compute_windowed_similarity(
    src_emb: np.ndarray,
    tgt_emb: np.ndarray,
    window: int,
    floor: float = 0.0,
) -> np.ndarray:
    """Full similarity pipeline: cosine similarity + window mask + floor clipping.

    Args:
        src_emb: Source embeddings, shape (n_src, dim).
        tgt_emb: Target embeddings, shape (n_tgt, dim).
        window: Half-width of diagonal band mask.
        floor: Minimum similarity threshold; values below are set to 0.

    Returns:
        Processed similarity matrix of shape (n_src, n_tgt).
    """
    sim = cosine_similarity_matrix(src_emb, tgt_emb)
    sim = apply_window_mask(sim, window)
    if floor > 0.0:
        sim[sim < floor] = 0.0
    return sim
