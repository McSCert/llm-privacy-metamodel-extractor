#!/usr/bin/env python3
"""Evaluate PIPEDA retrieval methods against a graded clause-reference CSV.

Run from the repository root:

    python evaluation/evaluate_retrieval.py --methods bm25
    python evaluation/evaluate_retrieval.py --methods bm25,semantic,hybrid
    python evaluation/evaluate_retrieval.py --methods bm25,semantic,hybrid \
        --bm25-weights 0.2,0.5,0.8 --k 5

The evaluator uses the same candidate pool for every method: all chunks whose
reference is the query's principle or one of its descendants. Use
--use-concept-tags to also apply the production concept-tag pre-filter.
SentenceTransformer methods require the optional sentence-transformers package
and may download the configured model on first use.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.embedder import (  # noqa: E402
    BM25Embedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
    hybrid_rank_by_similarity,
    rank_by_similarity,
)
from rag_pipeline.store import ChunkStore  # noqa: E402


LOG = logging.getLogger("evaluate_retrieval")
REFERENCE_RE = re.compile(
    r"^(?P<prefix>S1-?)?"
    r"(?P<clause>\d+(?:\.\d+)*(?:\([a-z]\))?)"
    r"(?::(?P<grade>[12])(?:\((?P<item>[a-z])\))?)?$",
    re.IGNORECASE,
)
CHUNK_REFERENCE_RE = re.compile(r"^(\d+(?:\.\d+)*(?:\([a-z]\))?)")


@dataclass(frozen=True)
class GoldQuery:
    query_id: str
    law: str
    principle: str
    concept: str
    query: str
    grades: dict[str, int]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    law: str
    article_ref: str
    reference: str
    text: str
    concept_tags: tuple[str, ...]
    vector: list[float] | None


def normalize_reference(value: str) -> str:
    """Extract the leading PIPEDA clause/item identifier from a chunk ref."""
    match = CHUNK_REFERENCE_RE.match(value.strip())
    return match.group(1).lower() if match else ""


def parse_gold_references(
    raw: str,
    query_id: str,
    warnings: list[str],
) -> dict[str, int]:
    """Parse semicolon-separated S1-clause:grade references.

    A listed reference without a grade is retained as grade 2, with a warning;
    this accommodates the existing CSV's ungraded entries without silently
    dropping them. A missing S1 prefix is also accepted with a warning.
    """
    grades: dict[str, int] = {}
    for token in raw.split(";"):
        token = token.strip()
        if not token:
            continue
        match = REFERENCE_RE.fullmatch(token)
        if match is None:
            raise ValueError(
                f"{query_id}: cannot parse gold reference {token!r}; "
                "expected S1-<clause>:<grade>"
            )
        if not match.group("prefix"):
            warnings.append(f"{query_id}: {token!r} has no S1 prefix")

        reference = match.group("clause").lower()
        if match.group("item"):
            reference += f"({match.group('item').lower()})"

        grade_text = match.group("grade")
        if grade_text is None:
            grade = 2
            warnings.append(
                f"{query_id}: {token!r} has no grade; treating it as grade 2"
            )
        else:
            grade = int(grade_text)

        grades[reference] = max(grade, grades.get(reference, 0))
    return grades


def load_gold(path: Path) -> tuple[list[GoldQuery], list[str]]:
    warnings: list[str] = []
    rows: list[GoldQuery] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "query_id",
            "law",
            "principle",
            "concept",
            "query",
            "relevant_pipeda_references",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            query_id = (row["query_id"] or "").strip()
            raw_references = row["relevant_pipeda_references"] or ""
            parsed_grades = parse_gold_references(
                raw_references,
                query_id,
                warnings,
            )
            if raw_references.strip() and not parsed_grades:
                warnings.append(
                    f"{query_id}: reference field contains no parseable clause references"
                )
            rows.append(
                GoldQuery(
                    query_id=query_id,
                    law=(row["law"] or "").strip().upper(),
                    principle=(row["principle"] or "").strip(),
                    concept=(row["concept"] or "").strip(),
                    query=(row["query"] or "").strip(),
                    grades=parsed_grades,
                )
            )
    return rows, warnings


def load_chunks(db_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with ChunkStore(db_path) as store:
        for law in store.laws():
            for row in store.get_chunks(law):
                chunks.append(
                    Chunk(
                        chunk_id=row["id"],
                        law=row["law"],
                        article_ref=row["article_ref"],
                        reference=normalize_reference(row["article_ref"]),
                        text=row["text"],
                        concept_tags=tuple(row.get("concept_tags", [])),
                        vector=row.get("vector"),
                    )
                )
    return chunks


def is_in_principle(reference: str, principle: str) -> bool:
    principle = principle.lower()
    return (
        reference == principle
        or reference.startswith(principle + ".")
        or reference.startswith(principle + "(")
    )


def select_candidates(
    query: GoldQuery,
    chunks: Sequence[Chunk],
    use_concept_tags: bool,
) -> list[Chunk]:
    principle = normalize_reference(query.principle)
    candidates = [
        chunk
        for chunk in chunks
        if chunk.law == query.law
        and chunk.reference
        and is_in_principle(chunk.reference, principle)
    ]
    if use_concept_tags:
        tagged = [
            chunk for chunk in candidates if query.concept in chunk.concept_tags
        ]
        if tagged:
            return tagged
    return candidates


def get_candidate_grade(reference: str, grades: dict[str, int]) -> int:
    """A parent chunk can satisfy a gold reference to one of its lettered items."""
    matching = [
        grade
        for gold_reference, grade in grades.items()
        if gold_reference == reference or gold_reference.startswith(reference + "(")
    ]
    return max(matching, default=0)


def matched_gold_references(reference: str, grades: dict[str, int]) -> set[str]:
    return {
        gold_reference
        for gold_reference, grade in grades.items()
        if grade > 0
        and (
            gold_reference == reference
            or gold_reference.startswith(reference + "(")
        )
    }


def evaluate_ranking(
    ranked: Sequence[tuple[Chunk, float]],
    grades: dict[str, int],
    k: int,
) -> dict[str, float | int]:
    top = ranked[:k]
    hit_count = sum(get_candidate_grade(chunk.reference, grades) > 0 for chunk, _ in top)
    relevant_refs = {reference for reference, grade in grades.items() if grade > 0}
    strict_refs = {reference for reference, grade in grades.items() if grade == 2}

    found_refs: set[str] = set()
    found_strict_refs: set[str] = set()
    dcg = 0.0
    reciprocal_rank = 0.0
    for rank, (chunk, _) in enumerate(top, start=1):
        grade = get_candidate_grade(chunk.reference, grades)
        found_refs.update(matched_gold_references(chunk.reference, grades))
        found_strict_refs.update(
            reference
            for reference in matched_gold_references(chunk.reference, grades)
            if grades[reference] == 2
        )
        dcg += (2**grade - 1) / math.log2(rank + 1)
        if reciprocal_rank == 0.0 and grade > 0:
            reciprocal_rank = 1.0 / rank

    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    return {
        "precision_at_k": hit_count / k,
        "recall_at_k": len(found_refs) / len(relevant_refs) if relevant_refs else 0.0,
        "grade2_recall_at_k": (
            len(found_strict_refs) / len(strict_refs) if strict_refs else 0.0
        ),
        "mrr_at_k": reciprocal_rank,
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
        "n_relevant_refs": len(relevant_refs),
        "n_grade2_refs": len(strict_refs),
        "n_relevant_chunks_at_k": hit_count,
    }


def parse_csv_list(value: str, cast, option: str):
    try:
        return [cast(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid {option}: {value!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "retrieval_gold_standard.csv",
    )
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "chunks.db")
    parser.add_argument(
        "--embedder",
        type=Path,
        default=PROJECT_ROOT / "data" / "bm25" / "chunks_PIPEDA_embedder.pkl",
        help="Pickled BM25Embedder that matches the BM25 vectors in --db",
    )
    parser.add_argument(
        "--methods",
        default="bm25,semantic,hybrid",
        help="Comma-separated methods: bm25, semantic, hybrid",
    )
    parser.add_argument(
        "--bm25-weights",
        default="0.2,0.5,0.8",
        help="BM25 weights for hybrid runs, from 0 to 1",
    )
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-base")
    parser.add_argument("--k", type=int, default=5, help="Cutoff for @k metrics")
    parser.add_argument(
        "--use-concept-tags",
        action="store_true",
        help="Apply concept-tag candidate filtering (same for every method)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "retrieval_results",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    if args.k < 1:
        parser.error("--k must be at least 1")
    methods = [method.strip().lower() for method in args.methods.split(",")]
    allowed = {"bm25", "semantic", "hybrid"}
    unknown = set(methods) - allowed
    if not methods or unknown:
        parser.error(
            f"unknown method(s): {', '.join(sorted(unknown)) or 'none'}; "
            f"choose from {', '.join(sorted(allowed))}"
        )
    required_paths = [args.gold, args.db]
    if "bm25" in methods or "hybrid" in methods:
        required_paths.append(args.embedder)
    for input_path in required_paths:
        if not input_path.exists():
            parser.error(f"file not found: {input_path}")

    try:
        bm25_weights = parse_csv_list(args.bm25_weights, float, "--bm25-weights")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if any(not 0.0 <= weight <= 1.0 for weight in bm25_weights):
        parser.error("--bm25-weights values must be between 0 and 1")
    if "hybrid" in methods and not bm25_weights:
        parser.error("--bm25-weights must contain at least one value for hybrid")

    queries, warnings = load_gold(args.gold)
    chunks = load_chunks(args.db)
    if not queries:
        parser.error(f"no query rows found in {args.gold}")
    candidate_sets = [
        select_candidates(query, chunks, args.use_concept_tags)
        for query in queries
    ]
    if any(not candidates for candidates in candidate_sets):
        missing = [
            query.query_id
            for query, candidates in zip(queries, candidate_sets)
            if not candidates
        ]
        parser.error("no candidate chunks found for query ids: " + ", ".join(missing))

    bm25_embedder: BM25Embedder | None = None
    bm25_setup_seconds = 0.0
    if "bm25" in methods or "hybrid" in methods:
        missing_vectors = sum(
            chunk.vector is None
            for candidates in candidate_sets
            for chunk in candidates
        )
        if missing_vectors:
            parser.error(
                f"{missing_vectors} candidate chunk(s) have no stored BM25 vector; "
                "re-ingest with the BM25 embedder or select semantic-only"
            )
        started = time.perf_counter()
        with args.embedder.open("rb") as stream:
            loaded = pickle.load(stream)
        if not isinstance(loaded, BM25Embedder):
            parser.error(
                f"{args.embedder} contains {type(loaded).__name__}, not BM25Embedder"
            )
        bm25_embedder = loaded
        bm25_setup_seconds = time.perf_counter() - started

    semantic_embedder: SentenceTransformerEmbedder | None = None
    semantic_vectors: dict[str, list[float]] = {}
    semantic_setup_seconds = 0.0
    if "semantic" in methods or "hybrid" in methods:
        try:
            semantic_embedder = SentenceTransformerEmbedder(args.model_name)
            unique_candidates = {
                chunk.chunk_id: chunk
                for candidates in candidate_sets
                for chunk in candidates
            }
            semantic_chunks = list(unique_candidates.values())
            started = time.perf_counter()
            vectors = semantic_embedder.embed_batch(
                [chunk.text for chunk in semantic_chunks]
            )
            semantic_setup_seconds = time.perf_counter() - started
            semantic_vectors = {
                chunk.chunk_id: vector
                for chunk, vector in zip(semantic_chunks, vectors)
            }
        except ImportError as exc:
            parser.error(
                f"semantic methods need sentence-transformers; install it with "
                f"`pip install sentence-transformers`. Details: {exc}"
            )

    runs: list[tuple[str, float | None]] = []
    if "bm25" in methods:
        runs.append(("bm25", None))
    if "semantic" in methods:
        runs.append(("semantic", None))
    if "hybrid" in methods:
        runs.extend((f"hybrid_bm25_{weight:.2f}", weight) for weight in bm25_weights)

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    per_query_metrics: dict[str, list[dict[str, float | int]]] = {
        method_name: [] for method_name, _ in runs
    }
    per_query_seconds: dict[str, list[float]] = {method_name: [] for method_name, _ in runs}

    for query, candidates in zip(queries, candidate_sets):
        bm25_ranked: list[tuple[Chunk, float]] = []
        dense_ranked: list[tuple[Chunk, float]] = []
        bm25_seconds = 0.0
        dense_seconds = 0.0

        if bm25_embedder is not None:
            started = time.perf_counter()
            bm25_candidates = [
                (chunk.chunk_id, chunk.vector)
                for chunk in candidates
                if chunk.vector is not None
            ]
            query_vector = bm25_embedder.embed(query.query)
            chunks_by_id = {chunk.chunk_id: chunk for chunk in candidates}
            bm25_ranked = [
                (chunks_by_id[chunk_id], score)
                for chunk_id, score in rank_by_similarity(
                    query_vector,
                    bm25_candidates,
                    top_k=len(bm25_candidates),
                )
            ]
            bm25_seconds = time.perf_counter() - started

        if semantic_embedder is not None:
            started = time.perf_counter()
            query_vector = semantic_embedder.embed(query.query)
            dense_ranked = [
                (chunk, cosine_similarity(query_vector, semantic_vectors[chunk.chunk_id]))
                for chunk in candidates
            ]
            dense_ranked.sort(key=lambda result: result[1], reverse=True)
            dense_seconds = time.perf_counter() - started

        for method_name, weight in runs:
            if method_name == "bm25":
                ranked = bm25_ranked
                elapsed = bm25_seconds
            elif method_name == "semantic":
                ranked = dense_ranked
                elapsed = dense_seconds
            else:
                started = time.perf_counter()
                bm25_by_id = {chunk.chunk_id: chunk for chunk in candidates}
                fused = hybrid_rank_by_similarity(
                    [(chunk.chunk_id, score) for chunk, score in bm25_ranked],
                    [(chunk.chunk_id, score) for chunk, score in dense_ranked],
                    top_k=len(candidates),
                    bm25_weight=weight if weight is not None else 0.5,
                )
                ranked = [(bm25_by_id[chunk_id], score) for chunk_id, score in fused]
                elapsed = bm25_seconds + dense_seconds + time.perf_counter() - started

            metrics = evaluate_ranking(ranked, query.grades, args.k)
            per_query_metrics[method_name].append(metrics)
            per_query_seconds[method_name].append(elapsed)
            detail_rows.append(
                {
                    "query_id": query.query_id,
                    "method": method_name,
                    "k": args.k,
                    **metrics,
                    "retrieved_references": ";".join(
                        chunk.reference for chunk, _ in ranked[:args.k]
                    ),
                }
            )

    for method_name, weight in runs:
        metrics_list = per_query_metrics[method_name]
        all_with_gold = [m for m in metrics_list if m["n_relevant_refs"] > 0]

        def average(key: str, rows: Sequence[dict[str, float | int]]) -> float:
            return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0

        setup_seconds = 0.0
        if method_name == "bm25":
            setup_seconds = bm25_setup_seconds
        elif method_name == "semantic":
            setup_seconds = semantic_setup_seconds
        elif method_name.startswith("hybrid"):
            setup_seconds = bm25_setup_seconds + semantic_setup_seconds
        summary_rows.append(
            {
                "method": method_name,
                "bm25_weight": "" if weight is None else weight,
                "n_queries": len(queries),
                "n_queries_with_gold": len(all_with_gold),
                "precision_at_k": average("precision_at_k", metrics_list),
                "recall_at_k": average("recall_at_k", all_with_gold),
                "grade2_recall_at_k": average("grade2_recall_at_k", all_with_gold),
                "mrr_at_k": average("mrr_at_k", metrics_list),
                "ndcg_at_k": average("ndcg_at_k", metrics_list),
                "avg_latency_ms": average(
                    "latency_seconds",
                    [
                        {"latency_seconds": value}
                        for value in per_query_seconds[method_name]
                    ],
                )
                * 1000,
                "setup_seconds": setup_seconds,
            }
        )

    summary_rows.sort(
        key=lambda row: (
            float(row["ndcg_at_k"]),
            float(row["recall_at_k"]),
            float(row["mrr_at_k"]),
        ),
        reverse=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "retrieval_summary.csv"
    detail_path = args.output_dir / "retrieval_per_query.csv"
    write_csv(summary_path, summary_rows)
    write_csv(detail_path, detail_rows)

    for warning in warnings:
        LOG.warning(warning)
    print(f"Candidate scope: principle + descendant clauses ({len(chunks)} stored chunks)")
    print(f"Concept-tag filter: {'on' if args.use_concept_tags else 'off'}")
    print(
        f"Gold rows: {len(queries)} ({sum(bool(query.grades) for query in queries)} "
        "with relevant references)"
    )
    print(f"Cutoff: k={args.k}; graded relevance: 2=standalone evidence, 1=supporting")
    print(
        f"{'Method':<22} {'P@k':>7} {'R@k':>7} {'R2@k':>7} "
        f"{'MRR@k':>7} {'nDCG@k':>7} {'ms/query':>10}"
    )
    for row in summary_rows:
        print(
            f"{row['method']:<22} {row['precision_at_k']:>7.3f} "
            f"{row['recall_at_k']:>7.3f} {row['grade2_recall_at_k']:>7.3f} "
            f"{row['mrr_at_k']:>7.3f} {row['ndcg_at_k']:>7.3f} "
            f"{row['avg_latency_ms']:>10.2f}"
        )
    print(f"Summary: {summary_path}")
    print(f"Per-query results: {detail_path}")
    return 0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
