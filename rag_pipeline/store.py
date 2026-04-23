"""
store.py — SQLite chunk store for the RAG pipeline.

Schema design
-------------
One table: `chunks`

    chunk_id      TEXT PRIMARY KEY   — SHA-1 of (law, article_ref, text[:64])
    law           TEXT               — canonical law name: GDPR, CCPA, LGPD, ...
    article_ref   TEXT               — structural ref: "Art.6", "§1798.100"
    parent_ref    TEXT               — enclosing unit: "CHAPTER II" or ""
    level         TEXT               — hierarchy level: article, clause, chapter
    concept_tags  TEXT               — JSON array: ["LegalBasis","Right"]
    text          TEXT               — raw chunk text
    vector        TEXT               — JSON array of floats (TF-IDF embedding)
    char_offset   INTEGER            — character offset in source document
    ingested_at   TEXT               — ISO-8601 timestamp

Indexes
-------
    idx_law             — fast filter by law name
    idx_law_concept     — fast filter by (law, concept_tag) for RAG pre-filtering
    idx_article_ref     — fast lookup by article reference

Why SQLite?
-----------
  - Zero infrastructure — no vector DB server to spin up.
  - The entire index fits in a single portable file.
  - For a research pipeline with <100k chunks (typical for 5-10 privacy laws),
    SQLite + Python cosine similarity is fast enough.
  - When scaling to production, swap store.py for a Qdrant or Chroma backend;
    the retriever.py interface stays unchanged.

Ingest pipeline
---------------
    chunk_file()  →  TFIDFEmbedder.fit()  →  ingest_chunks()
                                              ↓
                                          SQLite DB
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .chunker import Chunk
from .embedder import TFIDFEmbedder

log = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    law           TEXT    NOT NULL,
    article_ref   TEXT    NOT NULL,
    parent_ref    TEXT    NOT NULL DEFAULT '',
    level         TEXT    NOT NULL,
    concept_tags  TEXT    NOT NULL DEFAULT '[]',
    text          TEXT    NOT NULL,
    vector        TEXT    NOT NULL DEFAULT '[]',
    char_offset   INTEGER NOT NULL DEFAULT 0,
    ingested_at   TEXT    NOT NULL
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_law         ON chunks(law);",
    "CREATE INDEX IF NOT EXISTS idx_law_concept ON chunks(law, concept_tags);",
    "CREATE INDEX IF NOT EXISTS idx_article_ref ON chunks(article_ref);",
]


# ── ChunkStore ────────────────────────────────────────────────────────────────

class ChunkStore:
    """
    SQLite-backed store for embedded legal text chunks.

    Usage
    -----
        store = ChunkStore("data/chunks.db")
        store.ingest(chunks, embedder)          # embed + store in one call
        results = store.search("LegalBasis", "GDPR", query_text, embedder, top_k=5)
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        log.info(f"ChunkStore opened: {self.db_path}")

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(_CREATE_TABLE)
        for idx in _CREATE_INDEXES:
            cur.execute(idx)
        self._conn.commit()

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest(
        self,
        chunks: list[Chunk],
        embedder: TFIDFEmbedder,
        batch_size: int = 256,
        replace: bool = False,
    ) -> int:
        """
        Embed and store a list of Chunk objects.

        Parameters
        ----------
        chunks     : output of chunker.chunk_file()
        embedder   : a fitted TFIDFEmbedder (or any Embedder)
        batch_size : embedding batch size
        replace    : if True, replace existing chunks with same chunk_id

        Returns
        -------
        Number of chunks written.
        """
        if not chunks:
            log.warning("ingest() called with empty chunk list")
            return 0

        now = datetime.now(timezone.utc).isoformat()
        texts = [ch.text for ch in chunks]

        log.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}")
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_vectors.extend(embedder.embed_batch(batch))

        insert_sql = (
            "INSERT OR REPLACE INTO chunks "
            "(chunk_id, law, article_ref, parent_ref, level, "
            " concept_tags, text, vector, char_offset, ingested_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)"
            if replace else
            "INSERT OR IGNORE INTO chunks "
            "(chunk_id, law, article_ref, parent_ref, level, "
            " concept_tags, text, vector, char_offset, ingested_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)"
        )

        rows = [
            (
                ch.chunk_id,
                ch.law,
                ch.article_ref,
                ch.parent_ref,
                ch.level,
                json.dumps(ch.concept_tags),
                ch.text,
                json.dumps(vec),
                ch.char_offset,
                now,
            )
            for ch, vec in zip(chunks, all_vectors)
        ]

        cur = self._conn.cursor()
        cur.executemany(insert_sql, rows)
        self._conn.commit()
        written = cur.rowcount
        log.info(f"Wrote {written} chunks to {self.db_path}")
        return written

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_chunks_by_concept(
        self,
        concept: str,
        law: Optional[str] = None,
    ) -> list[sqlite3.Row]:
        """
        Return all chunks tagged with `concept`, optionally filtered by law.
        concept_tags is stored as a JSON array — we use a LIKE search.
        """
        cur = self._conn.cursor()
        if law:
            cur.execute(
                "SELECT * FROM chunks WHERE law=? AND concept_tags LIKE ?",
                (law.upper(), f'%"{concept}"%'),
            )
        else:
            cur.execute(
                "SELECT * FROM chunks WHERE concept_tags LIKE ?",
                (f'%"{concept}"%',),
            )
        return cur.fetchall()

    def get_chunk_by_id(self, chunk_id: str) -> Optional[sqlite3.Row]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,))
        return cur.fetchone()

    def get_chunks_by_article(
        self,
        article_ref: str,
        law: Optional[str] = None,
    ) -> list[sqlite3.Row]:
        cur = self._conn.cursor()
        if law:
            cur.execute(
                "SELECT * FROM chunks WHERE law=? AND article_ref LIKE ?",
                (law.upper(), f"%{article_ref}%"),
            )
        else:
            cur.execute(
                "SELECT * FROM chunks WHERE article_ref LIKE ?",
                (f"%{article_ref}%",),
            )
        return cur.fetchall()

    def laws(self) -> list[str]:
        """Return list of distinct law names in the store."""
        cur = self._conn.cursor()
        cur.execute("SELECT DISTINCT law FROM chunks ORDER BY law")
        return [row[0] for row in cur.fetchall()]

    def stats(self) -> dict:
        """Return a summary dict of what is stored."""
        cur = self._conn.cursor()
        cur.execute("SELECT law, COUNT(*) as n FROM chunks GROUP BY law")
        law_counts = {row["law"]: row["n"] for row in cur.fetchall()}
        cur.execute("SELECT COUNT(*) FROM chunks")
        total = cur.fetchone()[0]
        return {"total": total, "by_law": law_counts}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Full ingest pipeline ──────────────────────────────────────────────────────

def ingest_file(
    path: Path | str,
    law: str,
    db_path: Path | str,
    embedder_path: Optional[Path | str] = None,
    min_chunk_chars: int = 100,
    max_chunk_chars: int = 4000,
) -> dict:
    """
    Full pipeline: file → chunks → embedder fit → store.

    Parameters
    ----------
    path            : path to .pdf or .txt legal text file
    law             : canonical law name, e.g. "GDPR"
    db_path         : path to SQLite database file (created if missing)
    embedder_path   : path to save/load fitted TF-IDF embedder pickle.
                      If None, saves alongside the db as <db_stem>_<law>_embedder.pkl
    min_chunk_chars : passed to chunk_file()
    max_chunk_chars : passed to chunk_file()

    Returns
    -------
    dict with keys: law, chunks_produced, chunks_written, embedder_path, db_path
    """
    from .chunker import chunk_file

    path     = Path(path)
    db_path  = Path(db_path)
    law_upper = law.upper()

    if embedder_path is None:
        embedder_path = db_path.parent / f"{db_path.stem}_{law_upper}_embedder.pkl"
    embedder_path = Path(embedder_path)

    # ── Step 1: chunk ─────────────────────────────────────────────────────────
    log.info(f"=== Ingesting {path.name} as {law_upper} ===")
    chunks = chunk_file(path, law_upper, min_chunk_chars, max_chunk_chars)
    log.info(f"Produced {len(chunks)} chunks")

    # ── Step 2: fit embedder ──────────────────────────────────────────────────
    if embedder_path.exists():
        log.info(f"Loading existing embedder from {embedder_path}")
        embedder = TFIDFEmbedder.load(embedder_path)
    else:
        embedder = TFIDFEmbedder(max_features=4096, ngram_range=(1, 2))
        embedder.fit([ch.text for ch in chunks])
        embedder.save(embedder_path)

    # ── Step 3: store ─────────────────────────────────────────────────────────
    with ChunkStore(db_path) as store:
        written = store.ingest(chunks, embedder)
        stats = store.stats()

    log.info(f"Store stats: {stats}")

    return {
        "law":             law_upper,
        "chunks_produced": len(chunks),
        "chunks_written":  written,
        "embedder_path":   str(embedder_path),
        "db_path":         str(db_path),
        "store_stats":     stats,
    }
