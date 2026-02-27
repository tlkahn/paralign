"""Shared test fixtures for paralign tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest


class DeterministicEmbedder:
    """Mock embedding model that produces deterministic embeddings from sentence hashes.

    Uses SHA-256 hashing to generate reproducible, high-dimensional embeddings.
    Similar sentences (by hash) will NOT be similar -- this is purely for testing
    the alignment pipeline mechanics, not semantic similarity.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode sentences into deterministic embeddings."""
        embeddings = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for i, sentence in enumerate(sentences):
            h = hashlib.sha256(sentence.encode("utf-8")).digest()
            # Use hash bytes to seed a local RNG for reproducibility
            seed = int.from_bytes(h[:4], "big")
            rng = np.random.RandomState(seed)
            embeddings[i] = rng.randn(self.dim).astype(np.float32)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms

        return embeddings


class IdentityEmbedder:
    """Mock embedder that returns pre-set embeddings.

    Useful for tests where you need exact control over similarity scores.
    Pass embeddings directly; encode() returns them in order.
    """

    def __init__(self, embeddings: np.ndarray) -> None:
        self._embeddings = embeddings
        self._call_count = 0

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Return pre-set embeddings for the requested number of sentences."""
        start = self._call_count
        end = start + len(sentences)
        self._call_count = end
        result = self._embeddings[start:end].copy()

        if normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            result = result / norms

        return result


class ConstantEmbedder:
    """Mock embedder that produces embeddings achieving an exact cosine-similarity matrix.

    Given a desired (n_src, n_tgt) similarity matrix S, constructs:
      - Source embeddings: standard basis vectors e_i in R^(n_src + n_tgt)
      - Target embeddings: first n_src components = S[:, j], plus an orthogonal
        component at index n_src+j with value sqrt(1 - ||S[:,j]||^2) to ensure
        unit norm.

    Precondition: each column of S must have L2 norm < 1.
    """

    def __init__(self, sim: np.ndarray) -> None:
        n_src, n_tgt = sim.shape
        dim = n_src + n_tgt

        # Source embeddings: standard basis vectors e_i
        src = np.zeros((n_src, dim), dtype=np.float64)
        for i in range(n_src):
            src[i, i] = 1.0

        # Target embeddings: S[:, j] in first n_src dims, orthogonal pad in dim n_src+j
        tgt = np.zeros((n_tgt, dim), dtype=np.float64)
        for j in range(n_tgt):
            col = sim[:, j].astype(np.float64)
            col_norm_sq = float(np.dot(col, col))
            if col_norm_sq >= 1.0 - 1e-12:
                raise ValueError(
                    f"Column {j} of similarity matrix has L2 norm >= 1 "
                    f"({np.sqrt(col_norm_sq):.6f}); cannot construct unit-norm target."
                )
            tgt[j, :n_src] = col
            tgt[j, n_src + j] = np.sqrt(1.0 - col_norm_sq)

        self._src_emb = src.astype(np.float32)
        self._tgt_emb = tgt.astype(np.float32)
        self._sim = sim.copy()

    @property
    def src_emb(self) -> np.ndarray:
        return self._src_emb

    @property
    def tgt_emb(self) -> np.ndarray:
        return self._tgt_emb

    @property
    def similarity_matrix(self) -> np.ndarray:
        """Return the original requested similarity matrix."""
        return self._sim


@pytest.fixture
def deterministic_embedder() -> DeterministicEmbedder:
    """Provide a deterministic embedder for testing."""
    return DeterministicEmbedder(dim=128)


@pytest.fixture
def identity_embedder_factory():
    """Factory fixture for creating identity embedders with custom embeddings."""

    def _factory(embeddings: np.ndarray) -> IdentityEmbedder:
        return IdentityEmbedder(embeddings)

    return _factory
