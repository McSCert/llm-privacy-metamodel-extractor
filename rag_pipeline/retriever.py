"""
retriever.py — RAG retrieval interface for the extraction pipeline.

This is the component that the extraction pipeline calls:

    retriever = Retriever(db_path, embedder_path_map)
    chunks = retriever.retrieve("LegalBasis", "GDPR", top_k=3)
    # → [Chunk, Chunk, Chunk] — the 3 most relevant chunks for LegalBasis in GDPR

Retrieval strategy: two-stage
------------------------------
Stage 1 — Pre-filter by concept tag (fast, SQL-based):
    SELECT * FROM chunks WHERE law=? AND concept_tags LIKE '%"LegalBasis"%'
    This eliminates ~80-90% of chunks without any vector math.

Stage 2 — Re-rank by cosine similarity (numpy dot product):
    Compute cosine similarity between the query vector and each candidate.
    Return top_k by score.

The query vector is produced by embedding a concept-specific query string
that captures the metamodel concept's semantic centre:

    "LegalBasis"         → "lawful basis legal grounds processing consent contract"
    "RetentionPolicy"    → "retention period storage limitation delete no longer necessary"
    "ConsentWithdrawal"  → "withdraw consent revoke opt-out as easy as"
    "DataTransfer"       → "transfer third country adequacy standard contractual clauses"

These canonical queries are tuned for legal text — they use the vocabulary
that appears in GDPR, LGPD, CCPA, and PIPEDA for the same concepts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .chunker import Chunk
from .embedder import TFIDFEmbedder, cosine_similarity
from .store import ChunkStore

log = logging.getLogger(__name__)


# ── Canonical concept queries ─────────────────────────────────────────────────
# These are embedded at query time to find semantically similar chunks.
# Extend this dict when new metamodel concepts are added.

CONCEPT_QUERIES: dict[str, str] = {
    "LegalBasis": (
        "lawful basis legal grounds processing consent contract legitimate interest "
        "legal obligation vital interest public task consentimento base legal hipótese"
    ),
    "ProcessingActivity": (
        "processing personal data collect store use share transfer delete disclose "
        "processing operation treatment dados pessoais tratamento"
    ),
    "RetentionPolicy": (
        "retention period storage limitation no longer than necessary delete erase "
        "duration keep data prazo conservação armazenamento"
    ),
    "ConsentWithdrawal": (
        "withdraw consent revoke opt-out as easy as giving consent unsubscribe "
        "retirar consentimento revogar reti"
    ),
    "DataTransfer": (
        "transfer third country international adequacy decision standard contractual "
        "clauses binding corporate rules cross-border recipient country "
        "transferência internacional país terceiro"
    ),
    "Right": (
        "right to access rectification erasure restriction portability objection "
        "automated decision opt-out data subject rights direito acesso"
    ),
    "Purpose": (
        "processing purpose specific purpose objective goal finalidade "
        "purpose limitation specific consented purpose"
    ),
    "PersonalData": (
        "personal data sensitive special category biometric health financial "
        "location behavioral dados pessoais dados sensíveis informação pessoal"
    ),
}


# ── RetrievedChunk ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever, with its relevance score."""
    chunk: Chunk
    score: float

    def to_prompt_text(self) -> str:
        """
        Format the chunk for injection into an LLM prompt.
        Includes citation metadata so the LLM can populate source_clause.
        """
        return (
            f"[Source: {self.chunk.law} | {self.chunk.article_ref} | "
            f"score={self.score:.3f}]\n"
            f"{self.chunk.text}"
        )


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """
    Two-stage retriever: concept-tag pre-filter + cosine similarity re-ranking.

    Parameters
    ----------
    db_path         : path to the SQLite chunk store
    embedder_paths  : dict mapping law name to embedder pickle path,
                      e.g. {"GDPR": "data/chunks_GDPR_embedder.pkl"}
                      If a law is not in the map, raises KeyError.

    Usage
    -----
        retriever = Retriever("data/chunks.db", {
            "GDPR":   "data/chunks_GDPR_embedder.pkl",
            "CCPA":   "data/chunks_CCPA_embedder.pkl",
            "LGPD":   "data/chunks_LGPD_embedder.pkl",
        })
        chunks = retriever.retrieve("LegalBasis", "GDPR", top_k=3)
        for rc in chunks:
            print(rc.to_prompt_text())
    """

    def __init__(
        self,
        db_path: Path | str,
        embedder_paths: dict[str, Path | str],
    ):
        self.db_path = Path(db_path)
        self._embedder_paths = {
            law.upper(): Path(p) for law, p in embedder_paths.items()
        }
        self._embedders: dict[str, TFIDFEmbedder] = {}   # lazy-loaded
        self._store = ChunkStore(self.db_path)

    def _get_embedder(self, law: str) -> TFIDFEmbedder:
        law = law.upper()
        if law not in self._embedders:
            path = self._embedder_paths.get(law)
            if path is None:
                raise KeyError(
                    f"No embedder registered for law '{law}'. "
                    f"Available: {list(self._embedder_paths)}"
                )
            self._embedders[law] = TFIDFEmbedder.load(path)
        return self._embedders[law]

    def retrieve(
        self,
        concept: str,
        law: str,
        top_k: int = 3,
        query_text: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a metamodel concept in a law.

        Parameters
        ----------
        concept     : metamodel class name, e.g. "LegalBasis"
        law         : canonical law name, e.g. "GDPR"
        top_k       : maximum number of chunks to return
        query_text  : custom query string; defaults to CONCEPT_QUERIES[concept]
        min_score   : minimum cosine similarity threshold (0.0 = no filter)

        Returns
        -------
        List of RetrievedChunk, sorted by descending score.
        """
        law = law.upper()
        embedder = self._get_embedder(law)

        # ── Stage 1: pre-filter by concept tag ───────────────────────────────
        candidates = self._store.get_chunks_by_concept(concept, law)

        if not candidates:
            log.warning(
                f"No chunks found for concept='{concept}', law='{law}'. "
                f"Check that the law was ingested and concept tagger tagged it."
            )
            return []

        log.debug(f"Pre-filter: {len(candidates)} candidates for {concept}/{law}")

        # ── Stage 2: embed the query and re-rank ──────────────────────────────
        query = query_text or CONCEPT_QUERIES.get(concept, concept)
        query_vec = embedder.embed(query)

        scored: list[RetrievedChunk] = []
        for row in candidates:
            vec = json.loads(row["vector"])
            if not vec:
                continue
            score = cosine_similarity(query_vec, vec)
            if score >= min_score:
                chunk = Chunk(
                    chunk_id    = row["chunk_id"],
                    law         = row["law"],
                    article_ref = row["article_ref"],
                    parent_ref  = row["parent_ref"],
                    level       = row["level"],
                    text        = row["text"],
                    char_offset = row["char_offset"],
                    concept_tags= json.loads(row["concept_tags"]),
                )
                scored.append(RetrievedChunk(chunk=chunk, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        result = scored[:top_k]

        log.info(
            f"Retrieved {len(result)} chunks for {concept}/{law} "
            f"(scores: {[f'{r.score:.3f}' for r in result]})"
        )
        return result

    def retrieve_for_prompt(
        self,
        concept: str,
        law: str,
        top_k: int = 3,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Convenience method — returns a single string ready for prompt injection.
        Each chunk is prefixed with its citation metadata.
        """
        chunks = self.retrieve(concept, law, top_k)
        if not chunks:
            return f"[No relevant text found for {concept} in {law}]"
        return separator.join(rc.to_prompt_text() for rc in chunks)

    def stats(self) -> dict:
        """Return store statistics."""
        return self._store.stats()

    def close(self) -> None:
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Standalone ingest + retrieve demo ────────────────────────────────────────

def build_retriever_from_files(
    law_files: dict[str, Path | str],
    db_path: Path | str = "data/chunks.db",
    data_dir: Path | str = "data",
) -> Retriever:
    """
    Convenience function: ingest multiple law files and return a ready Retriever.

    Parameters
    ----------
    law_files : dict mapping law name to file path,
                e.g. {"GDPR": "gdpr.pdf", "CCPA": "ccpa.txt"}
    db_path   : SQLite database path
    data_dir  : directory for embedder pickle files

    Returns
    -------
    Configured Retriever ready for retrieve() calls.
    """
    from .store import ingest_file

    db_path  = Path(db_path)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    embedder_paths = {}
    for law, file_path in law_files.items():
        emb_path = data_dir / f"chunks_{law.upper()}_embedder.pkl"
        result = ingest_file(
            path=file_path,
            law=law,
            db_path=db_path,
            embedder_path=emb_path,
        )
        embedder_paths[law] = emb_path
        log.info(f"Ingested {law}: {result}")

    return Retriever(db_path, embedder_paths)
