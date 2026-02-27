"""Embedding model abstraction with Protocol-based structural typing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, runtime_checkable

import numpy as np
from typing_extensions import Protocol


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for sentence embedding models.

    Any class with a matching encode() method satisfies this protocol
    via structural subtyping -- no inheritance required.
    """

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray: ...


MODEL_REGISTRY: dict[str, str] = {
    "e5-large": "intfloat/multilingual-e5-large-instruct",
    "bge-m3": "BAAI/bge-m3",
    "labse": "sentence-transformers/LaBSE",
}


class SentenceTransformerModel:
    """Concrete embedding model using sentence-transformers with lazy loading."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    @classmethod
    def from_pretrained(
        cls, model_name: str, device: str | None = None
    ) -> SentenceTransformerModel:
        """Create a model instance (lazy -- does not load weights until encode())."""
        return cls(model_name, device)

    def _load_model(self) -> Any:
        """Load the sentence-transformers model on first use."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerModel. "
                "Install with: pip install paralign[models]"
            ) from e

        kwargs: dict[str, Any] = {}
        if self._device is not None:
            kwargs["device"] = self._device
        return SentenceTransformer(self._model_name, **kwargs)

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode sentences into embeddings, loading the model on first call."""
        if self._model is None:
            self._model = self._load_model()
        result: np.ndarray = self._model.encode(
            list(sentences),
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress_bar,
        )
        return result


def create_model(
    name_or_path: str, device: str | None = None
) -> SentenceTransformerModel:
    """Factory function to create an embedding model.

    Args:
        name_or_path: Either a registry key (e.g. "e5-large") or a
            HuggingFace model id / local path.
        device: Device to load model on (e.g. "cpu", "cuda").

    Returns:
        A SentenceTransformerModel instance (lazy-loaded).
    """
    model_name = MODEL_REGISTRY.get(name_or_path, name_or_path)
    return SentenceTransformerModel.from_pretrained(model_name, device=device)
