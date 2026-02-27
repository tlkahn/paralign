"""Smoke test: align 10 English sentences with 10 Spanish translations."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from paralign import AlignConfig, align


class SmokEmbedder:
    """Mock embedder that makes corresponding EN/ES sentences similar.

    Sentences at the same index in source/target get similar embeddings.
    This simulates what a real multilingual model would produce for
    correct translation pairs.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._call_count = 0

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        embeddings = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for i, sentence in enumerate(sentences):
            h = hashlib.sha256(sentence.encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "big")
            rng = np.random.RandomState(seed)
            embeddings[i] = rng.randn(self.dim).astype(np.float32)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms

        return embeddings


class PairedEmbedder:
    """Mock embedder where source[i] and target[i] get the same base vector.

    This simulates a perfect multilingual model: translation pairs are
    near-identical in embedding space, with a tiny amount of noise.

    align() calls encode() twice: first for source, then for target.
    Both calls draw from the same base vectors so that sentence i on
    each side shares a common base, plus a small noise offset.
    """

    def __init__(self, n_pairs: int, dim: int = 64, noise: float = 0.05) -> None:
        self.dim = dim
        self._bases = np.random.RandomState(42).randn(n_pairs, dim).astype(np.float32)
        self._noise = noise
        self._noise_rng = np.random.RandomState(123)
        self._call_count = 0

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        n = len(sentences)
        # Both source and target start from base index 0 so pairs share vectors
        embeddings = self._bases[:n].copy()
        embeddings += self._noise_rng.randn(n, self.dim).astype(np.float32) * self._noise
        self._call_count += 1

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms

        return embeddings


def main() -> None:
    en = [
        "The sun rises over the mountains.",
        "She opened the old wooden door.",
        "Children were playing in the park.",
        "The train arrived exactly on time.",
        "He poured a cup of hot coffee.",
        "Rain began to fall softly.",
        "The library was quiet and peaceful.",
        "They walked along the river bank.",
        "A bird sang from the highest branch.",
        "The night sky was full of stars.",
    ]

    es = [
        "El sol se eleva sobre las montañas.",
        "Ella abrió la vieja puerta de madera.",
        "Los niños jugaban en el parque.",
        "El tren llegó exactamente a tiempo.",
        "Él sirvió una taza de café caliente.",
        "La lluvia comenzó a caer suavemente.",
        "La biblioteca estaba tranquila y pacífica.",
        "Caminaron a lo largo de la orilla del río.",
        "Un pájaro cantó desde la rama más alta.",
        "El cielo nocturno estaba lleno de estrellas.",
    ]

    print("=" * 70)
    print("SMOKE TEST: paralign DP alignment engine")
    print("=" * 70)

    # --- Test 1: Perfect 1:1 with paired embedder ---
    print("\n--- Test 1: 10 EN → 10 ES (paired embedder, expect 1:1) ---\n")

    model = PairedEmbedder(n_pairs=len(en) + len(es))
    cfg = AlignConfig(window=50, include_similarity_matrix=True)
    result = align(en, es, config=cfg, model=model)

    for i, pair in enumerate(result.pairs):
        src_text = " | ".join(en[j] for j in pair.source_indices)
        tgt_text = " | ".join(es[j] for j in pair.target_indices)
        print(f"  [{pair.alignment_type.value}] (score={pair.score:.3f})")
        print(f"    EN: {src_text}")
        print(f"    ES: {tgt_text}")
        print()

    # Verify monotonicity
    prev_s, prev_t = -1, -1
    monotonic = True
    for pair in result.pairs:
        if pair.source_indices:
            if min(pair.source_indices) <= prev_s:
                monotonic = False
            prev_s = max(pair.source_indices)
        if pair.target_indices:
            if min(pair.target_indices) <= prev_t:
                monotonic = False
            prev_t = max(pair.target_indices)

    # Verify coverage
    src_covered = set()
    tgt_covered = set()
    for pair in result.pairs:
        src_covered.update(pair.source_indices)
        tgt_covered.update(pair.target_indices)

    print(f"  Pairs: {len(result.pairs)}")
    print(f"  Total score: {result.total_score:.3f}")
    print(f"  Monotonic: {monotonic}")
    print(f"  Source coverage: {src_covered == set(range(len(en)))}")
    print(f"  Target coverage: {tgt_covered == set(range(len(es)))}")
    print(f"  Similarity matrix shape: {result.similarity_matrix.shape}")  # type: ignore[union-attr]

    # --- Test 2: Unequal lengths (7 EN → 10 ES) ---
    print("\n--- Test 2: 7 EN → 10 ES (expect merges/skips) ---\n")

    en_short = en[:7]
    model2 = PairedEmbedder(n_pairs=len(en_short) + len(es), dim=64, noise=0.1)
    result2 = align(en_short, es, config=AlignConfig(window=50), model=model2)

    for pair in result2.pairs:
        src_text = " | ".join(en_short[j] for j in pair.source_indices) or "(skip)"
        tgt_text = " | ".join(es[j] for j in pair.target_indices) or "(skip)"
        print(f"  [{pair.alignment_type.value}] {src_text}  ↔  {tgt_text}")

    print(f"\n  Pairs: {len(result2.pairs)}")
    print(f"  Types: {[p.alignment_type.value for p in result2.pairs]}")

    # --- Test 3: merged_pairs() output ---
    print("\n--- Test 3: merged_pairs() convenience method ---\n")

    for src_text, tgt_text in result.merged_pairs():
        print(f"  {src_text}")
        print(f"  → {tgt_text}")
        print()

    print("=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
