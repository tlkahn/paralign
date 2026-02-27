# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

paralign is a DP-based parallel text alignment engine. It uses sentence embeddings and dynamic programming to find optimal monotonic alignments between source and target language sentences, supporting 1:1, 1:2, 2:1, 2:2, and skip (1:0, 0:1) alignments.

## Commands

```bash
# Install (uses uv)
uv sync --all-extras

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_dp.py::TestDPAlign::test_diagonal_alignment -v

# Type checking (strict mode)
uv run mypy src/paralign

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

## Architecture

The pipeline flows: **encode → similarity → DP alignment → result**.

The public API is the `align()` function in `src/paralign/__init__.py`. All internal modules use `_` prefix and are re-exported through `__init__.py.__all__`.

### Core Modules (`src/paralign/`)

- **`_embedding.py`** — `EmbeddingModel` Protocol (structural typing, runtime-checkable) + `SentenceTransformerModel` wrapper + `create_model()` factory with `MODEL_REGISTRY` (e5-large, bge-m3, labse)
- **`_similarity.py`** — Vectorized NumPy cosine similarity with diagonal band windowing (`apply_window_mask`)
- **`_dp.py`** — DP alignment over 6 transition types: `(1,1), (1,2), (2,1), (2,2), (1,0), (0,1)`. Uses `_merged_similarity()` to average embeddings for N:M groups. Key params: `skip_penalty`, `merge_penalty`, `band_width`
- **`_types.py`** — Frozen dataclasses: `AlignmentType` enum, `AlignmentPair`, `AlignmentResult`
- **`_config.py`** — `AlignConfig` frozen dataclass wrapping embedding, windowing, and `DPConfig` settings

### Testing

Tests use mock embedders (`DeterministicEmbedder`, `IdentityEmbedder` in `conftest.py`) to avoid requiring actual ML models. The `IdentityEmbedder` accepts pre-set embedding matrices for fully deterministic tests.

## Conventions

- Python 3.9+ with `from __future__ import annotations`
- Strict mypy, frozen dataclasses for immutability
- Ruff: line length 100, select `E, F, I, N, W, UP`
- numpy is the only required dependency; sentence-transformers/torch are optional (`models` extra)
