from __future__ import annotations

"""
rag_pipeline/store.py — SQLite ChunkStore and ingest_file()

Public interface (unchanged from TF-IDF version — run_pipeline.py needs no edits):

    ingest_file(path, law, db_path, embedder_path) -> {"chunks_produced": N, "chunks_written": N}
    ChunkStore(db_path)   — context manager
        .laws()           -> list[str]
        .article_refs()   -> list[{"article_ref": str, "level": str}]
        .get_chunks()     -> list[dict]
        .insert_chunks()  -> int

Key change from TF-IDF: the embedder is now a BM25Embedder.

  • At ingest: BM25Embedder.fit(all_chunk_texts) is called once per law file.
    doc_vectors() extracts the L2-normalised BM25 score row for each chunk and
    stores it as a BLOB in SQLite.  The fitted embedder is pickled alongside.

  • At retrieval (see retriever.py): the pickled BM25Embedder is loaded, the
    query is embedded with embed(), and cosine similarity ranks the stored rows.

Concept-tag tagging
-------------------
Each chunk is tagged at ingest time by keyword matching against CONCEPT_KEYWORDS.
This powers the O(1) concept-tag pre-filter in Retriever.retrieve(), which reduces
the candidate set before the more expensive cosine reranking step.

The tagger mirrors the known-issue fix documented in run_pipeline.py:
  [ISSUE-1] "TF-IDF embedder — zero retrieval scores for Actor and Constraint."
BM25 handles these concepts correctly because it does not penalise short, keyword-
dense terms the way IDF can when they appear in almost every article.
"""

import hashlib
import json
import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from rag_pipeline.chunker import chunk_file
from rag_pipeline.embedder import TFIDFEmbedder, BM25Embedder

log = logging.getLogger(__name__)

# ── Embedder selection ────────────────────────────────────────────────────────
# Change this ONE line to switch the embedder used by ingest_file().
# All three classes satisfy the Embedder protocol (fit/embed/embed_batch/doc_vectors).
#
#   TFIDFEmbedder               — original, fast, known ISSUE-1 (Actor/Constraint)
#   BM25Embedder                — fixes ISSUE-1, same interface, recommended
#   SentenceTransformerEmbedder — best quality; needs: pip install sentence-transformers
#
# After changing, delete data/chunks_*_embedder.pkl and re-run --stage ingest.
EMBEDDER_CLASS = BM25Embedder  # swap to TFIDFEmbedder or SentenceTransformerEmbedder

# ---------------------------------------------------------------------------
# Keyword → concept tag mapping used by the ingest-time tagger.
# Keys must match the concept names in PASS1_CONCEPTS (run_pipeline.py).
# ---------------------------------------------------------------------------
CONCEPT_KEYWORDS: dict[str, list[str]] = {
    "LegalBasis": [
        "legal basis", "lawful", "consent", "legitimate interest",
        "legal obligation", "vital interest", "public task", "authorised",
        "grounds for", "basis for processing",
    ],
    "ProcessingActivity": [
        "collect", "collect", "use", "disclose", "process", "store",
        "retain", "record", "share", "transmit", "access", "transfer",
        "handling", "processing",
    ],
    "Actor": [
        "organization", "organisation", "controller", "processor",
        "data subject", "individual", "third party", "recipient",
        "commissioner", "authority", "entity", "person",
    ],
    "Purpose": [
        "purpose", "reason", "objective", "goal", "intended use",
        "for the purpose of", "in order to", "to fulfill", "to provide",
    ],
    "Right": [
        "right", "access", "correction", "rectification", "erasure",
        "deletion", "portability", "object", "withdraw", "complaint",
        "redress", "request",
    ],
    "Constraint": [
        "limit", "limitation", "must not", "shall not", "prohibited",
        "restriction", "condition", "requirement", "obligation", "safeguard",
        "security measure", "only if", "except",
    ],
    "RetentionPolicy": [
        "retain", "retention", "kept", "stored", "period", "duration",
        "no longer than", "as long as", "destroy", "delete after",
        "archive", "disposal",
    ],
    "DataTransfer": [
        "transfer", "cross-border", "third country", "outside", "abroad",
        "international", "adequacy", "standard contractual", "binding",
        "mechanism", "disclose to",
    ],
    "ConsentWithdrawal": [
        "withdraw", "withdrawal", "revoke", "opt out", "opt-out",
        "unsubscribe", "refuse", "withhold", "retract",
    ],
}


def _tag_chunk(text: str) -> list[str]:
    """
    Return the concept tags that apply to a chunk by keyword matching.

    A tag is applied when at least one of its keywords appears in the
    lower-cased chunk text.  Multiple tags may apply to a single chunk.
    """
    text_lower = text.lower()
    tags: list[str] = []
    for concept, keywords in CONCEPT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(concept)
    return tags


def _chunk_id(law: str, article_ref: str, index: int, text: str) -> str:
    """
    Deterministic chunk identifier: sha1 of (law + article_ref + index + text[:64]).

    Using content in the hash means re-ingesting the same file produces the same
    IDs, so INSERT OR IGNORE in insert_chunks() is idempotent.
    """
    raw = f"{law}|{article_ref}|{index}|{text[:64]}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------
_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    law          TEXT    NOT NULL,
    article_ref  TEXT    NOT NULL,
    level        TEXT    NOT NULL,
    text         TEXT    NOT NULL,
    concept_tags TEXT    NOT NULL DEFAULT '[]',
    vector       BLOB
);

CREATE INDEX IF NOT EXISTS idx_chunks_law     ON chunks (law);
CREATE INDEX IF NOT EXISTS idx_chunks_article ON chunks (law, article_ref);
CREATE INDEX IF NOT EXISTS idx_chunks_level   ON chunks (law, level);
"""


# ---------------------------------------------------------------------------
# ChunkStore — thin SQLite wrapper
# ---------------------------------------------------------------------------

class ChunkStore:
    """
    SQLite-backed store for document chunks.

    Usage::

        with ChunkStore(db_path) as store:
            chunks = store.get_chunks(law="PIPEDA", article_ref="Principle 4.1")
    """

    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "ChunkStore":
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()
        return self

    def __exit__(self, *_) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Read helpers ──────────────────────────────────────────────────────────

    def laws(self) -> list[str]:
        """Return sorted list of law names present in the store."""
        cur = self._conn.execute(
            "SELECT DISTINCT law FROM chunks ORDER BY law"
        )
        return [row["law"] for row in cur.fetchall()]

    def article_refs(
        self,
        law: str,
        levels: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Return unique (article_ref, level) pairs for *law*, optionally
        filtered to *levels* (e.g. ``["article", "principle"]``).
        """
        if levels:
            placeholders = ",".join("?" * len(levels))
            cur = self._conn.execute(
                f"SELECT DISTINCT article_ref, level "
                f"FROM chunks "
                f"WHERE law=? AND level IN ({placeholders}) "
                f"ORDER BY article_ref",
                [law] + levels,
            )
        else:
            cur = self._conn.execute(
                "SELECT DISTINCT article_ref, level "
                "FROM chunks WHERE law=? ORDER BY article_ref",
                [law],
            )
        return [{"article_ref": row["article_ref"], "level": row["level"]}
                for row in cur.fetchall()]

    def list_articles(
        self,
        law: str,
        levels: list[str] | None = None,
    ) -> list[dict]:
        """
        Return one record per distinct article_ref for the given law.
        concept_tags are aggregated (union) across ALL chunk levels for
        the article — not just the article-level chunk — so that tags
        found only in clause-level sub-chunks are not missed.
        """
        cur = self._conn.cursor()
        if levels:
            placeholders = ",".join("?" * len(levels))
            # Step 1: get the canonical article_refs at the requested levels
            cur.execute(
                f"SELECT DISTINCT article_ref FROM chunks "
                f"WHERE law=? AND level IN ({placeholders}) "
                f"ORDER BY article_ref",
                (law.upper(), *levels),
            )
            article_refs = [row["article_ref"] for row in cur.fetchall()]
        else:
            cur.execute(
                "SELECT DISTINCT article_ref FROM chunks "
                "WHERE law=? ORDER BY article_ref",
                (law.upper(),),
            )
            article_refs = [row["article_ref"] for row in cur.fetchall()]

        # Step 2: for each article_ref, aggregate concept_tags from ALL
        # chunk levels (including clause sub-chunks) via GROUP_CONCAT
        results = []
        for article_ref in article_refs:
            cur.execute(
                """
                SELECT GROUP_CONCAT(concept_tags, '|||') AS all_tags
                FROM   chunks
                WHERE  law = ? AND article_ref = ?
                """,
                (law.upper(), article_ref),
            )
            row = cur.fetchone()
            merged_tags: set[str] = set()
            if row and row["all_tags"]:
                for tags_json in row["all_tags"].split("|||"):
                    try:
                        merged_tags.update(json.loads(tags_json))
                    except (json.JSONDecodeError, TypeError):
                        pass
            results.append({
                "article_ref":  article_ref,
                "concept_tags": merged_tags,
            })

        return results


    def get_chunks(
        self,
        law: str,
        article_ref: Optional[str] = None,
        concept_tags: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve chunks for *law*, optionally filtered by *article_ref* and/or
        *concept_tags* (OR-semantics: any matching tag qualifies a chunk).

        Returns a list of dicts with keys:
          id, law, article_ref, level, text, concept_tags (list), vector (list[float] | None)
        """
        sql = (
            "SELECT id, law, article_ref, level, text, concept_tags, vector "
            "FROM chunks WHERE law=?"
        )
        params: list = [law]

        if article_ref:
            sql += " AND article_ref=?"
            params.append(article_ref)

        cur = self._conn.execute(sql, params)
        results: list[dict] = []

        for row in cur.fetchall():
            tags: list[str] = json.loads(row["concept_tags"])

            # Concept-tag pre-filter (OR semantics)
            if concept_tags and not any(t in tags for t in concept_tags):
                continue

            vector: Optional[list[float]] = None
            if row["vector"] is not None:
                try:
                    vector = pickle.loads(row["vector"])
                except Exception:
                    pass  # corrupt blob — treat as no vector

            results.append({
                "id": row["id"],
                "law": row["law"],
                "article_ref": row["article_ref"],
                "level": row["level"],
                "text": row["text"],
                "concept_tags": tags,
                "vector": vector,
            })

        return results

    # ── Write helpers ─────────────────────────────────────────────────────────

    def insert_chunks(self, chunks: list[dict]) -> int:
        """
        Bulk-insert chunks.  Skips rows whose *id* already exists
        (INSERT OR IGNORE), making re-ingest idempotent.

        Returns the number of rows actually written.
        """
        written = 0
        for chunk in chunks:
            vector_blob: Optional[bytes] = None
            if chunk.get("vector") is not None:
                try:
                    vector_blob = pickle.dumps(chunk["vector"])
                except Exception as exc:
                    log.warning(f"Could not pickle vector for {chunk['id']}: {exc}")

            try:
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO chunks "
                    "(id, law, article_ref, level, text, concept_tags, vector) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk["id"],
                        chunk["law"],
                        chunk["article_ref"],
                        chunk["level"],
                        chunk["text"],
                        json.dumps(chunk.get("concept_tags", [])),
                        vector_blob,
                    ),
                )
                written += cur.rowcount
            except Exception as exc:
                log.warning(f"Failed to insert chunk {chunk.get('id')}: {exc}")

        self._conn.commit()
        return written


# ---------------------------------------------------------------------------
# ingest_file — public entry point called by run_pipeline.stage_ingest()
# ---------------------------------------------------------------------------

def ingest_file(
    path: Path,
    law: str,
    db_path: "Path | str | ChunkStore",
    embedder_path: Path,
) -> dict:
    """
    Chunk, embed, and persist one law file.

    Steps
    -----
    1. chunk_file() → hierarchical chunks (article/principle/clause level).
    2. Keyword tagger annotates each chunk with relevant concept tags.
    3. EMBEDDER_CLASS().fit(all_texts) — one index per law file.
    4. doc_vectors() → L2-normalised score rows (one per chunk).
    5. Chunks + vectors → SQLite via ChunkStore.insert_chunks().
    6. Fitted embedder pickled to *embedder_path* for query-time retrieval.

    Parameters
    ----------
    path          : Path to the input PDF or TXT file.
    law           : Short law name, e.g. "PIPEDA" or "GDPR".
    db_path       : Path to the SQLite chunks database, OR an already-open
                    ChunkStore instance (run_pipeline.py may pass either).
    embedder_path : Path where the fitted embedder will be pickled.

    Returns
    -------
    {"chunks_produced": int, "chunks_written": int}
    """
    path = Path(path)
    embedder_path = Path(embedder_path)

    log.info(f"[ingest] {law} ← {path}")

    # ── 1. Chunk ──────────────────────────────────────────────────────────────
    raw_chunks = chunk_file(path=path, law=law)
    n_produced = len(raw_chunks)
    log.info(f"[ingest] {law}: {n_produced} chunks produced")

    if n_produced == 0:
        log.warning(f"[ingest] {law}: no chunks produced — check the input file")
        return {"chunks_produced": 0, "chunks_written": 0}

    # ── 1b. Normalise to plain dicts ──────────────────────────────────────────
    # chunk_file() may return dataclass instances (Chunk), namedtuples, or dicts
    import dataclasses
    chunks: list[dict] = []
    for c in raw_chunks:
        if isinstance(c, dict):
            chunks.append(c)
        elif dataclasses.is_dataclass(c) and not isinstance(c, type):
            chunks.append(dataclasses.asdict(c))
        elif hasattr(c, "_asdict"):
            chunks.append(c._asdict())
        else:
            chunks.append(vars(c))

    # ── 2. Concept-tag annotation ─────────────────────────────────────────────
    texts: list[str] = []
    for chunk in chunks:
        if not chunk.get("concept_tags"):
            chunk["concept_tags"] = _tag_chunk(chunk.get("text", ""))
        texts.append(chunk.get("text", ""))

    # ── 3. Fit embedder ───────────────────────────────────────────────────────
    embedder = EMBEDDER_CLASS()
    embedder.fit(texts)

    # ── 4. Extract per-document vectors ───────────────────────────────────────
    doc_vecs: np.ndarray = embedder.doc_vectors()  # (n_docs, n_dims)

    for i, chunk in enumerate(chunks):
        if not chunk.get("id"):
            chunk["id"] = _chunk_id(
                law, chunk.get("article_ref", ""), i, chunk.get("text", "")
            )
        if i < doc_vecs.shape[0]:
            chunk["vector"] = doc_vecs[i].tolist()
        else:
            chunk["vector"] = None
            log.warning(
                f"[ingest] chunk index {i} exceeds doc_vecs rows "
                f"({doc_vecs.shape[0]}) — vector will be NULL"
            )

    # ── 5. Persist chunks ─────────────────────────────────────────────────────
    # Accept either an open ChunkStore or a db path — run_pipeline.py may
    # pass either depending on version.
    if isinstance(db_path, ChunkStore):
        # Already-open store: use directly, don't close it
        n_written = db_path.insert_chunks(chunks)
        db_name = str(getattr(db_path, '_db_path', 'store'))
    else:
        # Path: open, write, close
        _db_path = Path(db_path)
        with ChunkStore(_db_path) as store:
            n_written = store.insert_chunks(chunks)
        db_name = _db_path.name

    log.info(f"[ingest] {law}: {n_written}/{n_produced} chunks written to {db_name}")

    # ── 6. Pickle the fitted embedder ─────────────────────────────────────────
    embedder_path.parent.mkdir(parents=True, exist_ok=True)
    with open(embedder_path, "wb") as fh:
        pickle.dump(embedder, fh, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(f"[ingest] {law}: {EMBEDDER_CLASS.__name__} embedder saved → {embedder_path}")

    return {"chunks_produced": n_produced, "chunks_written": n_written}