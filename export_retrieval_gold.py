#!/usr/bin/env python3
"""
Export a retrieval gold-standard CSV from the chunk database.

Usage:
    python export_retrieval_gold.py --db data/bm25/chunks.db --law PIPEDA --out data/retrieval_gold.csv

This writes rows with headers:
    query_id,law,article_ref,concept,relevant_chunk_ids

You can manually edit the CSV afterward to decide which chunk IDs are truly relevant.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

CONCEPTS = [
    "LegalBasis",
    "ProcessingActivity",
    "Actor",
    "Purpose",
    "Right",
    "Constraint",
    "RetentionPolicy",
    "DataTransfer",
    "ConsentWithdrawal",
]


def fetch_articles(conn: sqlite3.Connection, law: str):
    rows = conn.execute(
        """
        SELECT DISTINCT article_ref
        FROM chunks
        WHERE law = ?
        ORDER BY article_ref
        """,
        (law.upper(),),
    ).fetchall()
    return [r[0] for r in rows]


def build_rows(conn: sqlite3.Connection, law: str, article_refs: list[str]):
    rows = []
    for article_ref in article_refs:
        chunks = conn.execute(
            """
            SELECT id, concept_tags, text
            FROM chunks
            WHERE law = ? AND article_ref = ?
            ORDER BY id
            """,
            (law.upper(), article_ref),
        ).fetchall()

        if not chunks:
            continue

        for concept in CONCEPTS:
            chunk_ids = []
            for c in chunks:
                tags = json.loads(c[1] or "[]")
                if concept in tags:
                    chunk_ids.append(c[0])

            query_id = f"{law.upper()}-{article_ref}-{concept}".replace(" ", "_")
            rows.append({
                "query_id": query_id,
                "law": law.upper(),
                "article_ref": article_ref,
                "concept": concept,
                "relevant_chunk_ids": ";".join(chunk_ids),
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export retrieval gold CSV from chunk db")
    parser.add_argument("--db", required=True, type=Path, help="Path to chunks DB, e.g. data/bm25/chunks.db")
    parser.add_argument("--law", required=True, help="Law code, e.g. PIPEDA")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path")
    args = parser.parse_args()

    if not args.db.exists():
        raise FileNotFoundError(f"Chunk DB not found: {args.db}")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        article_refs = fetch_articles(conn, args.law)
        rows = build_rows(conn, args.law, article_refs)
    finally:
        conn.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query_id", "law", "article_ref", "concept", "relevant_chunk_ids"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")
    print("Edit the relevant_chunk_ids column to mark the truly relevant chunk IDs for each query.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
