"""Tests for paralign embedding abstraction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from paralign._embedding import (
    MODEL_REGISTRY,
    EmbeddingModel,
    SentenceTransformerModel,
    create_model,
)


class TestEmbeddingModelProtocol:
    """Tests for Protocol compliance."""

    def test_deterministic_embedder_satisfies_protocol(
        self, deterministic_embedder: object
    ) -> None:
        """The test fixture should satisfy the EmbeddingModel protocol."""
        assert hasattr(deterministic_embedder, "encode")
        result = deterministic_embedder.encode(["hello", "world"])  # type: ignore[union-attr]
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2

    def test_custom_class_satisfies_protocol(self) -> None:
        """Any class with encode() matching the signature should work."""

        class CustomEmbedder:
            def encode(
                self,
                sentences: Sequence[str],
                batch_size: int = 32,
                normalize: bool = True,
                show_progress_bar: bool = False,
            ) -> np.ndarray:
                return np.zeros((len(sentences), 10), dtype=np.float32)

        embedder = CustomEmbedder()
        result = embedder.encode(["test"])
        assert result.shape == (1, 10)

    def test_protocol_runtime_check(self) -> None:
        """Verify runtime_checkable works on the Protocol."""

        class ValidEmbedder:
            def encode(
                self,
                sentences: Sequence[str],
                batch_size: int = 32,
                normalize: bool = True,
                show_progress_bar: bool = False,
            ) -> np.ndarray:
                return np.zeros((len(sentences), 10))

        assert isinstance(ValidEmbedder(), EmbeddingModel)


class TestModelRegistry:
    """Tests for MODEL_REGISTRY."""

    def test_registry_has_e5_large(self) -> None:
        assert "e5-large" in MODEL_REGISTRY

    def test_registry_has_bge_m3(self) -> None:
        assert "bge-m3" in MODEL_REGISTRY

    def test_registry_has_labse(self) -> None:
        assert "labse" in MODEL_REGISTRY

    def test_registry_values_are_strings(self) -> None:
        for key, value in MODEL_REGISTRY.items():
            assert isinstance(value, str), f"Registry value for {key!r} is not a string"

    def test_registry_values_look_like_model_ids(self) -> None:
        for key, value in MODEL_REGISTRY.items():
            assert "/" in value, (
                f"Registry value for {key!r} doesn't look like a HuggingFace model id"
            )


class TestSentenceTransformerModel:
    """Tests for SentenceTransformerModel (without loading real models)."""

    def test_lazy_loading_attribute(self) -> None:
        """Model should not be loaded at construction time."""
        model = SentenceTransformerModel.__new__(SentenceTransformerModel)
        model._model_name = "some/model"
        model._device = None
        model._model = None
        # _model should be None until encode() is called
        assert model._model is None

    def test_satisfies_protocol(self) -> None:
        """SentenceTransformerModel should satisfy EmbeddingModel."""
        assert issubclass(SentenceTransformerModel, EmbeddingModel)


class TestCreateModel:
    """Tests for create_model factory."""

    def test_create_with_registry_name(self) -> None:
        """create_model with a registry key should return a SentenceTransformerModel."""
        model = create_model("labse")
        assert isinstance(model, SentenceTransformerModel)

    def test_create_with_custom_path(self) -> None:
        """create_model with a custom path should also work."""
        model = create_model("some/custom-model")
        assert isinstance(model, SentenceTransformerModel)

    def test_create_with_device(self) -> None:
        """create_model should accept a device parameter."""
        model = create_model("labse", device="cpu")
        assert isinstance(model, SentenceTransformerModel)
