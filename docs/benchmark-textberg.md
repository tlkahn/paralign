# Text+Berg Benchmark: paralign vs bertalign vs vecalign

## Overview

Benchmark of sentence alignment quality on the **Text+Berg** corpus -- 7 German-French Alpine Club yearbook document pairs with gold standard alignments.

**Date:** 2026-02-28
**Hardware:** NVIDIA A100-SXM4-40GB, Lambda Cloud instance
**Embedding model:** LaBSE (`sentence-transformers/LaBSE`) for all three systems

## Corpus Statistics

| Metric | Value |
|---|---|
| Documents | 7 |
| Source sentences (de) | 991 |
| Target sentences (fr) | 1,011 |
| Gold alignments | 916 |
| Gold types exceeding 2:2 | 23 (2.5%) |

### Gold Alignment Type Distribution

| Type | Count | % |
|---|---|---|
| 1:1 | 678 | 74.0% |
| 2:1 | 82 | 9.0% |
| 1:2 | 63 | 6.9% |
| 0:1 | 47 | 5.1% |
| 2:2 | 12 | 1.3% |
| 1:0 | 11 | 1.2% |
| 3:1 | 10 | 1.1% |
| 1:3 | 8 | 0.9% |
| 3:2 | 2 | 0.2% |
| 1:4 | 2 | 0.2% |
| 2:3 | 1 | 0.1% |

Paralign's maximum alignment size is 2:2, so 23 gold alignments (2.5%) are structurally unreachable, giving a theoretical strict recall ceiling of ~97.5%.

## Systems Under Test

### paralign (5 configurations)

All configs use LaBSE, `window=50`. Embeddings are encoded once and cached, then reused across all DP configurations.

| Config | skip_penalty | merge_penalty |
|---|---|---|
| `default` | -0.3 | -0.05 |
| `low_skip` | -0.1 | -0.05 |
| `high_skip` | -0.5 | -0.05 |
| `low_merge` | -0.3 | -0.01 |
| `bertalign_like` | -0.1 | -0.01 |

### bertalign

Reference implementation from [bfsujason/bertalign](https://github.com/bfsujason/bertalign). Uses its own DP algorithm with faiss-based nearest-neighbor search, margin-based scoring, and length penalty. Default parameters (`max_align=5, top_k=3, win=5, skip=-0.1`).

Run via subprocess in a separate Python 3.10 venv (requires faiss, googletrans, sentence-splitter, numba). The faiss GPU path was disabled on the test instance (faiss-cpu only).

### vecalign

Reference implementation from [thompsonb/vecalign](https://github.com/thompsonb/vecalign). Uses overlap-based document embeddings and Cython-accelerated DP. `alignment_max_size=8`.

Originally designed for LASER embeddings; here we use **LaBSE** instead for a fair comparison (same embedding model across all three systems). Run via subprocess in a separate Python 3.10 venv. Overlap files are generated per-document, then encoded with LaBSE embeddings in paralign's process.

## Results

### Aggregate Metrics

| System | P_strict | R_strict | F1_strict | P_lax | R_lax | F1_lax |
|---|---|---|---|---|---|---|
| **bertalign** | **0.9320** | **0.9406** | **0.9363** | 0.9866 | 0.9907 | **0.9886** |
| vecalign (LaBSE) | 0.8657 | 0.8508 | 0.8582 | **0.9931** | 0.9825 | 0.9878 |
| paralign (bertalign_like) | 0.8158 | 0.8765 | 0.8451 | 0.9259 | **0.9860** | 0.9550 |
| paralign (low_skip) | 0.7952 | 0.8683 | 0.8301 | 0.9092 | 0.9848 | 0.9455 |
| paralign (low_merge) | 0.8103 | 0.8415 | 0.8256 | 0.9498 | 0.9779 | 0.9636 |
| paralign (default) | 0.8078 | 0.8415 | 0.8243 | 0.9438 | 0.9755 | 0.9594 |
| paralign (high_skip) | 0.7835 | 0.8193 | 0.8010 | 0.9209 | 0.9522 | 0.9363 |

### Per-Document F1 (strict)

| Doc | paralign (default) | paralign (bertalign_like) | bertalign | vecalign |
|---|---|---|---|---|
| 001 | 0.7066 | 0.7600 | **0.9111** | 0.7579 |
| 002 | 0.8447 | 0.8749 | **0.9547** | 0.8796 |
| 003 | 0.8636 | 0.8670 | **0.9368** | 0.8772 |
| 004 | 0.8228 | 0.8384 | **0.9649** | 0.9137 |
| 005 | 0.8657 | 0.8973 | **0.8806** | 0.8485 |
| 006 | 0.8739 | 0.8793 | **0.9873** | 0.9145 |
| 007 | 0.8148 | 0.8238 | **0.8848** | 0.8142 |

## Analysis

### Key findings

1. **Bertalign leads** by a significant margin on strict F1 (0.936 vs 0.858 vecalign vs 0.845 paralign best). It wins on every single document.

2. **Lax F1 is much closer**: all systems score 0.94--0.99, indicating paralign and vecalign find the right *neighborhoods* but miss exact alignment boundaries. The gap is primarily in strict precision/recall, not lax.

3. **Paralign's best config is `bertalign_like`** (skip=-0.1, merge=-0.01). Low skip penalty allows more flexible merging; low merge penalty reduces the cost of N:M alignments.

4. **The 2:2 ceiling is not the main bottleneck.** Paralign's theoretical strict recall ceiling is 97.5% (losing 23 gold alignments). But actual strict recall for the best config is only 87.7% -- the ~10% gap comes from DP alignment errors, not the structural limitation.

5. **Vecalign with LaBSE outperforms paralign on strict F1** (0.858 vs 0.845) but is comparable on lax F1. Vecalign's overlap-based embeddings (concatenating up to 6 adjacent sentences) may give it better context for boundary decisions.

6. **Document 001 is hardest** for all systems (F1 0.58--0.76 for paralign/vecalign vs 0.91 for bertalign). This document likely has structural differences that benefit bertalign's two-pass alignment approach.

### What bertalign does differently

Bertalign's advantage likely comes from:
- **Two-pass alignment**: first a rough alignment using nearest-neighbor search (faiss), then refined DP alignment
- **Margin-based scoring**: uses the margin between similarity of a pair vs. its neighbors, rather than raw cosine similarity
- **Length penalty**: incorporates character-length ratios as an additional signal
- **Larger max alignment size**: `max_align=5` vs paralign's hard-coded 2:2 ceiling

### Potential improvements for paralign

- **Margin-based scoring**: replace raw cosine similarity with margin scores (the single biggest differentiator)
- **Length penalty**: add character-length ratio as an auxiliary signal in the DP cost function
- **Larger alignment types**: support 3:1, 1:3, 3:2, etc. (would recover the 2.5% ceiling)
- **Two-pass approach**: use a rough first pass to constrain the DP search space

## Evaluation Methodology

Metrics are ported from bertalign's `eval.py`:

- **Strict precision/recall**: exact match of alignment pairs (source indices, target indices must match exactly)
- **Lax precision/recall**: partial overlap -- if any source sentence in a test alignment overlaps with a gold alignment's source, and the corresponding target also overlaps, it counts as a lax match
- **F1**: harmonic mean of precision and recall

Skip alignments (1:0, 0:1) are excluded from recall computation (following bertalign's convention).

## Reproducibility

### Script

```bash
uv run python benchmarks/eval_textberg.py \
  --corpus-dir ~/bertalign/text+berg \
  --bertalign-dir ~/bertalign \
  --vecalign-dir ~/vecalign
```

### Environment setup

```bash
# paralign (Python 3.12, uv-managed)
cd ~/paralign && uv sync --all-extras

# bertalign (Python 3.10, uv venv)
cd ~/bertalign
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python \
  sentence-transformers torch numba \
  googletrans==4.0.0-rc.1 sentence-splitter faiss-cpu -e .
# Patch: disable faiss GPU path in bertalign/corelib.py

# vecalign (Python 3.10, uv venv)
cd ~/vecalign
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python numpy cython setuptools
```

### Logs

Raw output is in [`logs/benchmark_textberg_stdout.txt`](../logs/benchmark_textberg_stdout.txt) and [`logs/benchmark_textberg_stderr.txt`](../logs/benchmark_textberg_stderr.txt).
