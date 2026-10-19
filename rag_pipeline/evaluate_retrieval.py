#!/usr/bin/env python3
"""Evaluate retrieval quality independently of LLM extraction.

Gold CSV columns: query_id,law,article_ref,concept,relevant_chunk_ids
Use semicolons between relevant chunk IDs in relevant_chunk_ids.
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from pathlib import Path
from typing import Any
from rag_pipeline.retriever import Retriever

try:
    import psutil
except ImportError:
    psutil = None


def current_rss_mb() -> float | None:
    """Current resident memory of this process, or None without psutil."""
    if psutil is None:
        return None
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except (psutil.Error, OSError):
        return None


def peak_rss_mb() -> float | None:
    """Best available process peak RSS estimate across supported platforms."""
    # Windows exposes peak_wset through psutil; Unix resource reports peak RSS.
    if psutil is not None:
        try:
            info = psutil.Process().memory_info()
            if hasattr(info, "peak_wset"):
                return info.peak_wset / (1024 * 1024)
        except (psutil.Error, OSError):
            pass
    try:
        import resource
        import platform
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux and most other Unix platforms report KiB.
        return peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024
    except (ImportError, OSError, AttributeError):
        return None

REQUIRED_COLUMNS = {"query_id", "law", "article_ref", "concept", "relevant_chunk_ids"}

def parse_relevant_ids(value: str) -> set[str]:
    return {item.strip() for item in (value or "").split(";") if item.strip()}

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / k

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0

def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, cid in enumerate(retrieved[:k], 1) if cid in relevant)
    ideal_count = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0

def load_gold(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Gold CSV missing columns: {', '.join(sorted(missing))}")
        rows = []
        for line, row in enumerate(reader, 2):
            clean = {key: (value or "").strip() for key, value in row.items() if key}
            for required in ("query_id", "law", "concept", "relevant_chunk_ids"):
                if not clean.get(required):
                    raise ValueError(f"Missing {required} on CSV line {line}")
            rows.append(clean)
    if not rows:
        raise ValueError(f"No queries found in {path}")
    return rows

def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_gold(args.gold)
    if args.law:
        rows = [r for r in rows if r["law"].upper() == args.law.upper()]
        if not rows:
            raise ValueError(f"No gold queries found for law {args.law}")
    laws = {r["law"].upper() for r in rows}
    embedder_paths = {law: args.embedder_pickle for law in laws}
    per_query = []
    total_wall_seconds = 0.0
    total_cpu_seconds = 0.0
    rss_before_mb = current_rss_mb()
    retriever = Retriever(args.db, embedder_paths)
    try:
        for row in rows:
            relevant = parse_relevant_ids(row["relevant_chunk_ids"])
            rss_before_query_mb = current_rss_mb()
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            results = retriever.retrieve_chunks(
                law=row["law"].upper(), article_ref=row.get("article_ref", ""),
                concept=row["concept"], top_k=args.top_k,
                use_concept_tags=not args.disable_concept_tags,
            )
            query_cpu_seconds = time.process_time() - cpu_start
            query_wall_seconds = time.perf_counter() - wall_start
            total_wall_seconds += query_wall_seconds
            total_cpu_seconds += query_cpu_seconds
            retrieved = [result.chunk_id for result in results]
            per_query.append({
                "query_id": row["query_id"], "law": row["law"].upper(),
                "article_ref": row.get("article_ref", ""), "concept": row["concept"],
                "num_relevant": len(relevant), "num_retrieved": len(retrieved),
                "precision_at_k": precision_at_k(retrieved, relevant, args.top_k),
                "recall_at_k": recall_at_k(retrieved, relevant, args.top_k),
                "mrr_at_k": mrr_at_k(retrieved, relevant, args.top_k),
                "ndcg_at_k": ndcg_at_k(retrieved, relevant, args.top_k),
                "retrieval_wall_time_seconds": query_wall_seconds,
                "retrieval_cpu_time_seconds": query_cpu_seconds,
                "rss_before_query_mb": rss_before_query_mb,
                "rss_after_query_mb": current_rss_mb(),
                "retrieved_chunk_ids": retrieved,
                "relevant_chunk_ids": sorted(relevant),
                "retrieved_scores": [{"chunk_id": r.chunk_id, "score": r.score} for r in results],
            })
    finally:
        retriever.close()
    metrics = ("precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k")
    summary = {m: sum(float(r[m]) for r in per_query) / len(per_query) for m in metrics}
    summary.update({
        "total_retrieval_wall_time_seconds": total_wall_seconds,
        "mean_retrieval_wall_time_seconds": total_wall_seconds / len(per_query),
        "total_retrieval_cpu_time_seconds": total_cpu_seconds,
        "mean_retrieval_cpu_time_seconds": total_cpu_seconds / len(per_query),
        "queries_per_second": len(per_query) / total_wall_seconds if total_wall_seconds > 0 else None,
        "rss_before_retrieval_mb": rss_before_mb,
        "rss_after_retrieval_mb": current_rss_mb(),
        "peak_process_rss_mb": peak_rss_mb(),
        "psutil_available": psutil is not None,
    })
    return {"method": args.method, "top_k": args.top_k, "num_queries": len(per_query),
            "db": str(args.db), "embedder_pickle": str(args.embedder_pickle),
            "concept_tags_enabled": not args.disable_concept_tags,
            "computational_measurement_notes": {
                "wall_time": "Times retrieve_chunks() only; excludes LLM extraction.",
                "cpu_time": "Process CPU time spent during retrieve_chunks().",
                "memory": "RSS/peak RSS are process-level and include Python, libraries, and loaded embedder/index; not isolated retriever allocations.",
                "first_query": "The first query may include lazy embedder loading, so later queries may be faster.",
            },
            "summary": summary, "per_query": per_query}

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate top-K retrieval against chunk-level relevance judgments.")
    parser.add_argument("--gold", type=Path, required=True, help="Retrieval gold-standard CSV")
    parser.add_argument("--db", type=Path, required=True, help="Chunk database, e.g. data/bm25/chunks.db")
    parser.add_argument("--embedder-pickle", type=Path, required=True, help="Embedder pickle for this method/law")
    parser.add_argument("--law", help="Optional law filter, e.g. PIPEDA")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--method", default="unspecified", help="Label saved in output")
    parser.add_argument("--disable-concept-tags", action="store_true")
    parser.add_argument("--out", type=Path, help="Output JSON path")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    for label, path in (("Gold CSV", args.gold), ("Chunk DB", args.db), ("Embedder pickle", args.embedder_pickle)):
        if not path.exists():
            parser.error(f"{label} not found: {path}")
    try:
        result = evaluate(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"\nRetrieval evaluation: {result['method']} ({result['num_queries']} queries, K={result['top_k']})")
    print("\nQuality metrics:")
    for metric in ("precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k"):
        print(f"{metric}: {result['summary'][metric]:.4f}")
    summary = result["summary"]
    print("\nComputational metrics:")
    print(f"Total retrieval wall time: {summary['total_retrieval_wall_time_seconds']:.4f} s")
    print(f"Mean retrieval wall time:  {summary['mean_retrieval_wall_time_seconds']:.6f} s/query")
    print(f"Total retrieval CPU time:  {summary['total_retrieval_cpu_time_seconds']:.4f} s")
    print(f"Mean retrieval CPU time:   {summary['mean_retrieval_cpu_time_seconds']:.6f} s/query")
    if summary["queries_per_second"] is not None:
        print(f"Throughput:                {summary['queries_per_second']:.2f} queries/s")
    if summary["peak_process_rss_mb"] is not None:
        print(f"Peak process RSS:          {summary['peak_process_rss_mb']:.2f} MiB")
    else:
        print("Peak process RSS:          unavailable (install psutil for RSS snapshots)")
    if summary["rss_before_retrieval_mb"] is not None:
        print(f"RSS before retrieval:      {summary['rss_before_retrieval_mb']:.2f} MiB")
    if summary["rss_after_retrieval_mb"] is not None:
        print(f"RSS after retrieval:       {summary['rss_after_retrieval_mb']:.2f} MiB")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Detailed results written to: {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
