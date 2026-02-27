"""paralign: DP-based parallel text alignment engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from paralign._config import AlignConfig
from paralign._dp import DPConfig, dp_align
from paralign._embedding import EmbeddingModel, create_model
from paralign._similarity import compute_windowed_similarity
from paralign._types import AlignmentPair, AlignmentResult, AlignmentType

__all__ = [
    "align",
    "create_model",
    "AlignConfig",
    "AlignmentPair",
    "AlignmentResult",
    "AlignmentType",
    "DPConfig",
    "EmbeddingModel",
]


def align(
    source_sentences: Sequence[str],
    target_sentences: Sequence[str],
    config: AlignConfig | None = None,
    model: Any = None,
) -> AlignmentResult:
    """Align source and target sentences using DP-based alignment.

    Args:
        source_sentences: Source language sentences.
        target_sentences: Target language sentences.
        config: Alignment configuration. Uses defaults if None.
        model: An EmbeddingModel instance. If None, creates one from config.model.

    Returns:
        AlignmentResult with aligned pairs and metadata.
    """
    if config is None:
        config = AlignConfig()

    # Get or create embedding model
    if model is None:
        if config.model is None:
            raise ValueError(
                "Either pass a model instance or set config.model to a model name/path."
            )
        model = create_model(config.model, device=config.device)

    # Encode sentences
    src_emb: np.ndarray = model.encode(
        list(source_sentences),
        batch_size=config.embed_batch_size,
        normalize=config.normalize,
    )
    tgt_emb: np.ndarray = model.encode(
        list(target_sentences),
        batch_size=config.embed_batch_size,
        normalize=config.normalize,
    )

    # Compute similarity matrix
    sim = compute_windowed_similarity(
        src_emb, tgt_emb, window=config.window, floor=config.floor
    )

    # Run DP alignment
    pairs = dp_align(sim, src_emb, tgt_emb, config.dp)

    return AlignmentResult(
        pairs=tuple(pairs),
        source_sentences=tuple(source_sentences),
        target_sentences=tuple(target_sentences),
        similarity_matrix=sim if config.include_similarity_matrix else None,
    )
