#!/usr/bin/env python3
"""Evaluate paralign on Text+Berg corpus (vs bertalign & vecalign).

Single self-contained script that:
1. Runs paralign natively (multiple configs) with LaBSE
2. Runs bertalign via subprocess in its own venv
3. Runs vecalign via subprocess in its own venv, with LaBSE embeddings
4. Scores all against gold, prints comparison tables

Usage:
    uv run python benchmarks/eval_textberg.py
    uv run python benchmarks/eval_textberg.py --paralign-only
    uv run python benchmarks/eval_textberg.py --skip-bertalign --skip-vecalign
    uv run python benchmarks/eval_textberg.py --corpus-dir ~/bertalign/text+berg
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import textwrap
from ast import literal_eval
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Section 1: Constants & Config
# ---------------------------------------------------------------------------

BERTALIGN_DIR = Path("~/Desktop/bertalign").expanduser()
VECALIGN_DIR = Path("~/Desktop/vecalign").expanduser()
CORPUS_DIR = BERTALIGN_DIR / "text+berg"
DOC_IDS = ["001", "002", "003", "004", "005", "006", "007"]
CACHE_DIR = Path(__file__).parent / ".cache"

Alignment = list[tuple[tuple[int, ...], tuple[int, ...]]]


@dataclass(frozen=True)
class ParalignConfig:
    name: str
    skip_penalty: float
    merge_penalty: float
    window: int = 50


PARALIGN_CONFIGS = [
    ParalignConfig("default", skip_penalty=-0.3, merge_penalty=-0.05),
    ParalignConfig("low_skip", skip_penalty=-0.1, merge_penalty=-0.05),
    ParalignConfig("high_skip", skip_penalty=-0.5, merge_penalty=-0.05),
    ParalignConfig("low_merge", skip_penalty=-0.3, merge_penalty=-0.01),
    ParalignConfig("bertalign_like", skip_penalty=-0.1, merge_penalty=-0.01),
]


# ---------------------------------------------------------------------------
# Section 2: Eval Metrics (ported from bertalign/eval.py)
# ---------------------------------------------------------------------------


def _precision(
    goldalign: set[tuple[tuple[int, ...], tuple[int, ...]]],
    testalign: set[tuple[tuple[int, ...], tuple[int, ...]]],
    src_id_to_gold_tgt_ids: dict[int, set[int]],
) -> np.ndarray:
    """Compute tpstrict, fpstrict, tplax, fplax for test against gold."""
    tpstrict = 0
    tplax = 0
    fpstrict = 0
    fplax = 0

    for test_src, test_tgt in testalign:
        if (test_src, test_tgt) == ((), ()):
            continue
        if (test_src, test_tgt) in goldalign:
            tpstrict += 1
            tplax += 1
        else:
            target_ids: set[int] = set()
            for src_id in test_src:
                target_ids.update(src_id_to_gold_tgt_ids[src_id])
            if set(test_tgt).intersection(target_ids):
                fpstrict += 1
                tplax += 1
            else:
                fpstrict += 1
                fplax += 1

    return np.array([tpstrict, fpstrict, tplax, fplax], dtype=np.int32)


def _build_src_to_tgt_map(
    goldalign: set[tuple[tuple[int, ...], tuple[int, ...]]],
) -> dict[int, set[int]]:
    mapping: dict[int, set[int]] = defaultdict(set)
    for gold_src, gold_tgt in goldalign:
        for src_id in gold_src:
            for tgt_id in gold_tgt:
                mapping[src_id].add(tgt_id)
    return mapping


def score_multiple(
    gold_list: list[Alignment],
    test_list: list[Alignment],
    value_for_div_by_0: float = 0.0,
) -> dict[str, float]:
    """Aggregate precision/recall/F1 (strict & lax) across multiple documents."""
    pcounts = np.array([0, 0, 0, 0], dtype=np.int32)
    rcounts = np.array([0, 0, 0, 0], dtype=np.int32)

    for goldalign, testalign in zip(gold_list, test_list):
        gold_set = {(tuple(x), tuple(y)) for x, y in goldalign if len(x) or len(y)}
        test_set = {(tuple(x), tuple(y)) for x, y in testalign if len(x) or len(y)}

        src_to_tgt = _build_src_to_tgt_map(gold_set)
        pcounts += _precision(gold_set, test_set, src_to_tgt)

        # Recall = precision with args swapped, skips removed
        test_no_del = {(x, y) for x, y in test_set if len(x) and len(y)}
        gold_no_del = {(x, y) for x, y in gold_set if len(x) and len(y)}
        # For recall: goldalign=test_no_del, testalign=gold_no_del
        # Mapping must be built from the "goldalign" arg (test_no_del)
        recall_map = _build_src_to_tgt_map(test_no_del)
        rcounts += _precision(test_no_del, gold_no_del, recall_map)

    def safe_div(a: int, b: int) -> float:
        return a / float(b) if b else value_for_div_by_0

    pstrict = safe_div(pcounts[0], pcounts[0] + pcounts[1])
    plax = safe_div(pcounts[2], pcounts[2] + pcounts[3])
    rstrict = safe_div(rcounts[0], rcounts[0] + rcounts[1])
    rlax = safe_div(rcounts[2], rcounts[2] + rcounts[3])

    fstrict = (
        2 * pstrict * rstrict / (pstrict + rstrict)
        if (pstrict + rstrict)
        else value_for_div_by_0
    )
    flax = (
        2 * plax * rlax / (plax + rlax) if (plax + rlax) else value_for_div_by_0
    )

    return {
        "precision_strict": pstrict,
        "recall_strict": rstrict,
        "f1_strict": fstrict,
        "precision_lax": plax,
        "recall_lax": rlax,
        "f1_lax": flax,
    }


def score_single(
    gold: Alignment, test: Alignment, value_for_div_by_0: float = 0.0
) -> dict[str, float]:
    return score_multiple([gold], [test], value_for_div_by_0)


# ---------------------------------------------------------------------------
# Section 3: Corpus Loading
# ---------------------------------------------------------------------------


def load_sentences(corpus_dir: Path, doc_id: str, lang: str) -> list[str]:
    """Read one-sentence-per-line file (no .txt extension)."""
    path = corpus_dir / lang / doc_id
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def load_gold(corpus_dir: Path, doc_id: str) -> Alignment:
    """Parse gold alignment file with [ids]:[ids] format."""
    path = corpus_dir / "gold" / doc_id
    alignments: Alignment = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        fields = [x.strip() for x in line.split(":") if x.strip()]
        if len(fields) < 2:
            raise ValueError(f"Bad gold line: {line!r}")
        src = literal_eval(fields[0])
        tgt = literal_eval(fields[1])
        if isinstance(src, int):
            src = (src,)
        if isinstance(tgt, int):
            tgt = (tgt,)
        alignments.append((tuple(src), tuple(tgt)))
    return alignments


@dataclass
class DocData:
    doc_id: str
    src_sents: list[str]
    tgt_sents: list[str]
    gold: Alignment


def load_corpus(corpus_dir: Path) -> list[DocData]:
    corpus = []
    for doc_id in DOC_IDS:
        src = load_sentences(corpus_dir, doc_id, "de")
        tgt = load_sentences(corpus_dir, doc_id, "fr")
        gold = load_gold(corpus_dir, doc_id)
        corpus.append(DocData(doc_id=doc_id, src_sents=src, tgt_sents=tgt, gold=gold))
    return corpus


def corpus_stats(corpus: list[DocData]) -> dict[str, Any]:
    total_src = sum(len(d.src_sents) for d in corpus)
    total_tgt = sum(len(d.tgt_sents) for d in corpus)
    total_gold = sum(len(d.gold) for d in corpus)

    type_counts: Counter[str] = Counter()
    exceeds_2x2 = 0
    for d in corpus:
        for src_ids, tgt_ids in d.gold:
            key = f"{len(src_ids)}:{len(tgt_ids)}"
            type_counts[key] += 1
            if len(src_ids) > 2 or len(tgt_ids) > 2:
                exceeds_2x2 += 1

    return {
        "total_src": total_src,
        "total_tgt": total_tgt,
        "total_gold": total_gold,
        "type_counts": type_counts,
        "exceeds_2x2": exceeds_2x2,
        "exceeds_pct": 100.0 * exceeds_2x2 / total_gold if total_gold else 0,
    }


# ---------------------------------------------------------------------------
# Section 4: Paralign Runner
# ---------------------------------------------------------------------------


def _get_or_compute_embeddings(
    corpus: list[DocData], cache_dir: Path
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Encode all documents with LaBSE, caching to disk."""
    from paralign import create_model

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "embeddings.npz"

    if cache_file.exists():
        print("  Loading cached embeddings...", file=sys.stderr)
        data = np.load(cache_file)
        src_embs = [data[f"src_{i}"] for i in range(len(corpus))]
        tgt_embs = [data[f"tgt_{i}"] for i in range(len(corpus))]
        return src_embs, tgt_embs

    print("  Encoding with LaBSE (first run, will cache)...", file=sys.stderr)
    model = create_model("labse")

    src_embs = []
    tgt_embs = []
    save_dict: dict[str, np.ndarray] = {}

    for i, doc in enumerate(corpus):
        print(f"    Encoding doc {doc.doc_id}...", file=sys.stderr)
        src_emb: np.ndarray = model.encode(doc.src_sents, batch_size=64, normalize=True)
        tgt_emb: np.ndarray = model.encode(doc.tgt_sents, batch_size=64, normalize=True)
        src_embs.append(src_emb)
        tgt_embs.append(tgt_emb)
        save_dict[f"src_{i}"] = src_emb
        save_dict[f"tgt_{i}"] = tgt_emb

    np.savez(cache_file, **save_dict)
    print("  Embeddings cached.", file=sys.stderr)
    return src_embs, tgt_embs


def run_paralign(
    corpus: list[DocData],
    configs: list[ParalignConfig],
    cache_dir: Path,
) -> dict[str, list[Alignment]]:
    """Run paralign with multiple configs. Returns {config_name: [alignments_per_doc]}."""
    from paralign import DPConfig
    from paralign._dp import dp_align
    from paralign._similarity import compute_windowed_similarity

    src_embs, tgt_embs = _get_or_compute_embeddings(corpus, cache_dir)

    results: dict[str, list[Alignment]] = {}

    for cfg in configs:
        print(f"  Running paralign ({cfg.name})...", file=sys.stderr)
        dp_cfg = DPConfig(
            skip_penalty=cfg.skip_penalty,
            merge_penalty=cfg.merge_penalty,
            band_width=cfg.window,
        )
        doc_alignments: list[Alignment] = []
        for i in range(len(corpus)):
            sim = compute_windowed_similarity(
                src_embs[i], tgt_embs[i], window=cfg.window, floor=0.0
            )
            pairs = dp_align(sim, src_embs[i], tgt_embs[i], dp_cfg)
            alignment: Alignment = [
                (pair.source_indices, pair.target_indices) for pair in pairs
            ]
            doc_alignments.append(alignment)
        results[f"paralign ({cfg.name})"] = doc_alignments

    return results


# ---------------------------------------------------------------------------
# Section 5: Bertalign Runner (subprocess)
# ---------------------------------------------------------------------------


def run_bertalign(
    corpus: list[DocData], bertalign_dir: Path
) -> list[Alignment] | None:
    """Run bertalign via subprocess. Returns None on failure."""
    import subprocess

    python = bertalign_dir / ".venv" / "bin" / "python"
    if not python.exists():
        print(
            f"  WARNING: bertalign venv not found at {python}, skipping.",
            file=sys.stderr,
        )
        return None

    doc_alignments: list[Alignment] = []

    for doc in corpus:
        print(f"  Running bertalign on doc {doc.doc_id}...", file=sys.stderr)
        src_text = "\n".join(doc.src_sents)
        tgt_text = "\n".join(doc.tgt_sents)

        # Use a temp file for JSON output to avoid bertalign's stdout noise.
        result_file = tempfile.mktemp(suffix=".json")
        script = textwrap.dedent(f"""\
            import json
            import sys
            sys.path.insert(0, {str(bertalign_dir)!r})
            from bertalign import Bertalign
            src_text = {src_text!r}
            tgt_text = {tgt_text!r}
            aligner = Bertalign(src_text, tgt_text, is_split=True)
            aligner.align_sents()
            result = []
            for src_ids, tgt_ids in aligner.result:
                result.append([list(src_ids), list(tgt_ids)])
            with open({result_file!r}, "w") as f:
                json.dump(result, f)
        """)

        script_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(script)
                script_path = f.name

            proc = subprocess.run(
                [str(python), script_path],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(bertalign_dir),
            )
            if proc.returncode != 0:
                print(
                    f"  WARNING: bertalign failed on doc {doc.doc_id}:\n"
                    f"    {proc.stderr[-500:]}",
                    file=sys.stderr,
                )
                return None

            result_path = Path(result_file)
            if not result_path.exists():
                print(
                    f"  WARNING: bertalign produced no output for doc {doc.doc_id}:\n"
                    f"    {proc.stderr[-500:]}",
                    file=sys.stderr,
                )
                return None

            parsed = json.loads(result_path.read_text())
            alignment: Alignment = [
                (tuple(src_ids), tuple(tgt_ids)) for src_ids, tgt_ids in parsed
            ]
            doc_alignments.append(alignment)
        except Exception as e:
            print(f"  WARNING: bertalign exception on doc {doc.doc_id}: {e}", file=sys.stderr)
            return None
        finally:
            if script_path is not None:
                Path(script_path).unlink(missing_ok=True)
            Path(result_file).unlink(missing_ok=True)

    return doc_alignments


# ---------------------------------------------------------------------------
# Section 6: Vecalign Runner (subprocess)
# ---------------------------------------------------------------------------


def _generate_overlaps(
    sents: list[str],
    label: str,
    tmpdir: Path,
    vecalign_dir: Path,
    num_overlaps: int = 6,
) -> Path:
    """Write sentences to file and generate overlap file via vecalign's overlap.py."""
    import subprocess

    sent_file = tmpdir / f"{label}.txt"
    sent_file.write_text("\n".join(sents) + "\n", encoding="utf-8")

    overlap_file = tmpdir / f"{label}_overlaps.txt"
    python = vecalign_dir / ".venv" / "bin" / "python"
    overlap_script = vecalign_dir / "overlap.py"

    proc = subprocess.run(
        [
            str(python),
            str(overlap_script),
            "-i", str(sent_file),
            "-o", str(overlap_file),
            "-n", str(num_overlaps),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"overlap.py failed for {label}: {proc.stderr[:500]}")
    return overlap_file


def _encode_for_vecalign(overlap_file: Path, emb_file: Path) -> None:
    """Encode overlap text file into binary .emb format using LaBSE."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    texts = [
        line.strip() for line in overlap_file.read_text(encoding="utf-8").splitlines()
    ]
    model = SentenceTransformer("sentence-transformers/LaBSE")
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    np.array(emb, dtype=np.float32).tofile(str(emb_file))


def run_vecalign(
    corpus: list[DocData], vecalign_dir: Path
) -> list[Alignment] | None:
    """Run vecalign via subprocess. Returns None on failure."""
    import subprocess

    python = vecalign_dir / ".venv" / "bin" / "python"
    vecalign_script = vecalign_dir / "vecalign.py"
    if not python.exists():
        print(
            f"  WARNING: vecalign venv not found at {python}, skipping.",
            file=sys.stderr,
        )
        return None

    doc_alignments: list[Alignment] = []

    for doc in corpus:
        print(f"  Running vecalign on doc {doc.doc_id}...", file=sys.stderr)

        try:
            with tempfile.TemporaryDirectory() as tmpdir_str:
                tmpdir = Path(tmpdir_str)

                # 6a. Generate overlaps
                src_overlap = _generate_overlaps(
                    doc.src_sents, "src", tmpdir, vecalign_dir
                )
                tgt_overlap = _generate_overlaps(
                    doc.tgt_sents, "tgt", tmpdir, vecalign_dir
                )

                # 6b. Generate embeddings (in current process with LaBSE)
                src_emb_file = tmpdir / "src_overlaps.emb"
                tgt_emb_file = tmpdir / "tgt_overlaps.emb"
                print(f"    Encoding overlaps for doc {doc.doc_id}...", file=sys.stderr)
                _encode_for_vecalign(src_overlap, src_emb_file)
                _encode_for_vecalign(tgt_overlap, tgt_emb_file)

                # Source/target sentence files (vecalign reads these too)
                src_sent_file = tmpdir / "src.txt"
                tgt_sent_file = tmpdir / "tgt.txt"
                src_sent_file.write_text(
                    "\n".join(doc.src_sents) + "\n", encoding="utf-8"
                )
                tgt_sent_file.write_text(
                    "\n".join(doc.tgt_sents) + "\n", encoding="utf-8"
                )

                # 6c. Run vecalign
                proc = subprocess.run(
                    [
                        str(python),
                        str(vecalign_script),
                        "--alignment_max_size", "8",
                        "--src", str(src_sent_file),
                        "--tgt", str(tgt_sent_file),
                        "--src_embed", str(src_overlap), str(src_emb_file),
                        "--tgt_embed", str(tgt_overlap), str(tgt_emb_file),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    cwd=str(vecalign_dir),
                )
                if proc.returncode != 0:
                    print(
                        f"  WARNING: vecalign failed on doc {doc.doc_id}:\n"
                        f"    {proc.stderr[:500]}",
                        file=sys.stderr,
                    )
                    return None

                # Parse output: "[src_ids]:[tgt_ids]:cost" per line
                alignment: Alignment = []
                for line in proc.stdout.strip().splitlines():
                    parts = line.split(":")
                    if len(parts) < 2:
                        continue
                    src_ids = literal_eval(parts[0].strip())
                    tgt_ids = literal_eval(parts[1].strip())
                    if isinstance(src_ids, int):
                        src_ids = (src_ids,)
                    if isinstance(tgt_ids, int):
                        tgt_ids = (tgt_ids,)
                    alignment.append((tuple(src_ids), tuple(tgt_ids)))
                doc_alignments.append(alignment)

        except Exception as e:
            print(f"  WARNING: vecalign exception on doc {doc.doc_id}: {e}", file=sys.stderr)
            return None

    return doc_alignments


# ---------------------------------------------------------------------------
# Section 7: Reporting
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "precision_strict",
    "recall_strict",
    "f1_strict",
    "precision_lax",
    "recall_lax",
    "f1_lax",
]
METRIC_HEADERS = ["P_strict", "R_strict", "F1_strict", "P_lax", "R_lax", "F1_lax"]


def print_aggregate_table(
    all_results: dict[str, dict[str, float]],
    system_order: list[str],
) -> None:
    name_w = max(len(s) for s in system_order)
    col_w = 9

    header = f"{'System':<{name_w}} | " + " | ".join(
        f"{h:>{col_w}}" for h in METRIC_HEADERS
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for sys_name in system_order:
        if sys_name not in all_results:
            continue
        res = all_results[sys_name]
        vals = " | ".join(f"{res[k]:>{col_w}.4f}" for k in METRIC_KEYS)
        print(f"{sys_name:<{name_w}} | {vals}")
    print(sep)


def print_per_document_table(
    per_doc: dict[str, list[dict[str, float]]],
    system_order: list[str],
) -> None:
    col_w = 12
    header = f"{'Doc':<5} | " + " | ".join(
        f"{s:>{col_w}}" for s in system_order if s in per_doc
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for i, doc_id in enumerate(DOC_IDS):
        vals = []
        for sys_name in system_order:
            if sys_name not in per_doc:
                continue
            f1 = per_doc[sys_name][i].get("f1_strict", 0.0)
            vals.append(f"{f1:>{col_w}.4f}")
        print(f"{doc_id:<5} | " + " | ".join(vals))
    print(sep)


def print_2x2_analysis(stats: dict[str, Any]) -> None:
    exceeds = stats["exceeds_2x2"]
    pct = stats["exceeds_pct"]
    ceiling = 100.0 - pct

    print(f"Gold types exceeding 2:2: {exceeds} ({pct:.1f}%)")
    print(f"Paralign strict recall ceiling: {ceiling:.1f}%")
    print()
    print("Gold alignment type distribution:")
    for atype, count in sorted(stats["type_counts"].items()):
        print(f"  {atype}: {count}")


# ---------------------------------------------------------------------------
# Section 8: Main / CLI
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    corpus_dir = Path(args.corpus_dir).expanduser()
    bertalign_dir = Path(args.bertalign_dir).expanduser()
    vecalign_dir = Path(args.vecalign_dir).expanduser()
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else CACHE_DIR

    # Load corpus
    print("Loading corpus...", file=sys.stderr)
    corpus = load_corpus(corpus_dir)
    stats = corpus_stats(corpus)

    # Header
    print("=" * 74)
    print("Text+Berg Benchmark: paralign vs bertalign vs vecalign (de->fr, 7 docs)")
    print("=" * 74)
    print()
    print(
        f"Corpus: {len(corpus)} documents, "
        f"~{stats['total_src']} src sents, "
        f"~{stats['total_tgt']} tgt sents, "
        f"~{stats['total_gold']} gold alignments"
    )
    print_2x2_analysis(stats)
    print()

    # Collect results
    all_results: dict[str, dict[str, float]] = {}
    per_doc: dict[str, list[dict[str, float]]] = {}
    system_order: list[str] = []

    gold_list = [doc.gold for doc in corpus]

    # --- Paralign ---
    print("Running paralign...", file=sys.stderr)
    paralign_results = run_paralign(corpus, PARALIGN_CONFIGS, cache_dir)

    for sys_name, doc_alignments in paralign_results.items():
        system_order.append(sys_name)
        all_results[sys_name] = score_multiple(gold_list, doc_alignments)
        per_doc[sys_name] = [
            score_single(gold, test)
            for gold, test in zip(gold_list, doc_alignments)
        ]

    if args.paralign_only:
        args.skip_bertalign = True
        args.skip_vecalign = True

    # --- Bertalign ---
    if not args.skip_bertalign:
        print("Running bertalign...", file=sys.stderr)
        bert_alignments = run_bertalign(corpus, bertalign_dir)
        if bert_alignments is not None:
            sys_name = "bertalign"
            system_order.append(sys_name)
            all_results[sys_name] = score_multiple(gold_list, bert_alignments)
            per_doc[sys_name] = [
                score_single(gold, test)
                for gold, test in zip(gold_list, bert_alignments)
            ]
        else:
            print("  Bertalign skipped due to errors.", file=sys.stderr)

    # --- Vecalign ---
    if not args.skip_vecalign:
        print("Running vecalign...", file=sys.stderr)
        vec_alignments = run_vecalign(corpus, vecalign_dir)
        if vec_alignments is not None:
            sys_name = "vecalign (LaBSE)"
            system_order.append(sys_name)
            all_results[sys_name] = score_multiple(gold_list, vec_alignments)
            per_doc[sys_name] = [
                score_single(gold, test)
                for gold, test in zip(gold_list, vec_alignments)
            ]
        else:
            print("  Vecalign skipped due to errors.", file=sys.stderr)

    # --- Report ---
    print()
    print("Aggregate Results:")
    print_aggregate_table(all_results, system_order)

    print()
    print("Per-Document F1 (strict):")
    print_per_document_table(per_doc, system_order)

    if not args.skip_vecalign and "vecalign (LaBSE)" in all_results:
        print()
        print("Note: vecalign uses LaBSE embeddings (not LASER as in the original paper)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate paralign on Text+Berg corpus"
    )
    parser.add_argument("--skip-bertalign", action="store_true")
    parser.add_argument("--skip-vecalign", action="store_true")
    parser.add_argument("--paralign-only", action="store_true")
    parser.add_argument(
        "--corpus-dir",
        default=str(CORPUS_DIR),
        help="Path to text+berg corpus directory",
    )
    parser.add_argument(
        "--bertalign-dir",
        default=str(BERTALIGN_DIR),
        help="Path to bertalign repo",
    )
    parser.add_argument(
        "--vecalign-dir",
        default=str(VECALIGN_DIR),
        help="Path to vecalign repo",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Path to cache directory for embeddings",
    )
    parsed = parser.parse_args()
    main(parsed)
