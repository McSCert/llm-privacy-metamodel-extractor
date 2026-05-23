from __future__ import annotations

"""
rag_pipeline/retriever.py — Concept-aware BM25 retriever

Public interface (unchanged — run_pipeline.py needs no edits):

    retriever = Retriever(store=store, embedder_paths=embedder_paths)
    rag_text  = retriever.retrieve(
                    law=law,
                    article_ref=article_ref,
                    concept=concept,
                    top_k=top_k,
                    use_concept_tags=use_concept_tags,
                )

Retrieval pipeline (two-stage):
--------------------------------
Stage 1 — Concept-tag pre-filter  (O(1) SQL, no embedding required)
  • Pull only chunks whose concept_tags column contains the requested concept.
  • Falls back to ALL chunks for the article if no tagged chunks exist.
  • Can be disabled with use_concept_tags=False (~3× more calls, see run_pipeline).

Stage 2 — BM25 cosine rerank  (replaces old TF-IDF cosine rerank)
  • Query text: "<concept> <article_ref>" — compact but discriminative.
  • embed(query) → L2-normalised binary indicator vector.
  • rank_by_similarity(query_vec, [(id, doc_vec), ...], top_k) picks top chunks.
  • doc_vec rows were written at ingest by store.ingest_file() using
    BM25Embedder.doc_vectors() — each row is the L2-normalised BM25 score
    profile for that document.  The cosine between indicator query vec and
    BM25 document row equals the standard BM25 relevance score.

Why BM25 fixes the TF-IDF Actor / Constraint zero-score bug (ISSUE-1):
  TF-IDF assigns low IDF weight to terms that appear in nearly every article
  (e.g. "organization", "shall", "must").  BM25 saturates TF and applies a
  length penalty, giving reasonable scores even for these ubiquitous terms.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .chunker import Chunk
from .embedder import BM25Embedder, cosine_similarity
from .store import ChunkStore
from rag_pipeline.embedder import BM25Embedder, rank_by_similarity
from rag_pipeline.store import ChunkStore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concept → query expansion terms
#
# Exported as CONCEPT_QUERIES to match the __init__.py public interface.
# The query "<concept> <article_ref>" is extended with these characteristic
# terms to improve BM25 token overlap with relevant chunks.
# ---------------------------------------------------------------------------
CONCEPT_QUERIES: dict[str, str] = {
    "LegalBasis":          "legal basis consent lawful grounds",
    "ProcessingActivity":  "processing collect use disclose activity",
    "Actor":               "actor organization controller processor individual",
    "Purpose":             "purpose objective reason goal",
    "Right":               "right access correction deletion request",
    "Constraint":          "constraint limitation obligation safeguard",
    "RetentionPolicy":     "retention period duration keep destroy",
    "DataTransfer":        "transfer third country cross-border mechanism",
    "ConsentWithdrawal":   "withdraw withdrawal revoke opt-out consent",
}

# Keep the private alias so internal methods still work
_CONCEPT_QUERY_EXPANSION = CONCEPT_QUERIES


# ---------------------------------------------------------------------------
# RetrievedChunk — structured result returned by retrieve_chunks()
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """
    A single retrieved chunk with its metadata and similarity score.

    Attributes
    ----------
    chunk_id    : Unique chunk identifier (from SQLite).
    law         : Law name, e.g. "PIPEDA".
    article_ref : Article or principle identifier, e.g. "Principle 4.1".
    level       : Hierarchy level: "article", "principle", "clause", etc.
    text        : Raw chunk text injected into the LLM prompt.
    concept_tags: Concept tags assigned at ingest time.
    score       : Cosine similarity score (0.0–1.0); -1.0 if unranked.
    """
    chunk_id:     str
    law:          str
    article_ref:  str
    level:        str
    text:         str
    concept_tags: list[str]
    score:        float = -1.0


class Retriever:
    """
    Two-stage BM25 retriever for privacy law chunks.

    Parameters
    ----------
    store_or_path  : Either an already-open ChunkStore, or a path to the
                     SQLite database file (string or Path).  When a path is
                     given the Retriever opens and owns the connection —
                     this matches the original call convention used by
                     run_pipeline.stage_extract():
                         Retriever(db_path, embedder_paths)
    embedder_paths : Mapping of law name → path to pickled embedder pickle,
                     as returned by stage_ingest().
    """

    def __init__(
        self,
        store_or_path: "ChunkStore | Path | str",
        embedder_paths: dict[str, "Path | str"],
    ) -> None:
        # Accept either an open ChunkStore or a db path (original interface)
        if isinstance(store_or_path, ChunkStore):
            self._store = store_or_path
            self._owns_store = False          # caller manages the connection
        else:
            # Path or str — open and own the connection
            self._store = ChunkStore(Path(store_or_path))
            self._store.__enter__()
            self._owns_store = True

        self._embedder_paths: dict[str, Path] = {
            k.upper(): Path(v) for k, v in embedder_paths.items()
        }
        self._embedder_cache: dict[str, BM25Embedder] = {}

    def close(self) -> None:
        """
        Close the owned ChunkStore connection.
        Called explicitly by run_pipeline.stage_extract() after each law.
        Safe to call multiple times.
        """
        if getattr(self, "_owns_store", False):
            try:
                self._store.__exit__(None, None, None)
            except Exception:
                pass
            self._owns_store = False

    def __del__(self) -> None:
        """Fallback cleanup if close() was not called explicitly."""
        self.close()

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        law: str,
        article_ref: str,
        concept: str,
        top_k: int = 3,
        use_concept_tags: bool = True,
    ) -> str:
        """
        Retrieve and return the concatenated text of the *top_k* most relevant
        chunks for a given (law, article, concept) triple.

        Parameters
        ----------
        law              : Short law name, e.g. "PIPEDA".
        article_ref      : Article or principle identifier, e.g. "Principle 4.1".
        concept          : One of the nine Pass-1 concept names, e.g. "LegalBasis".
        top_k            : Number of chunks to return (default 3, CLI --top-k).
        use_concept_tags : When True (default), pre-filter by concept tags before
                           cosine reranking.  Set False to consider all chunks.

        Returns
        -------
        Newline-separated chunk texts, ready for injection into the LLM prompt.
        Empty string when no chunks are found.
        """
        law = law.upper()

        # ── Stage 1: candidate retrieval ─────────────────────────────────────
        candidates_raw = self._get_candidates(law, article_ref, concept, use_concept_tags)

        if not candidates_raw:
            log.debug(f"[retrieve] No chunks for {law}/{article_ref}/{concept}")
            return ""

        # ── Stage 2: BM25 cosine rerank ───────────────────────────────────────
        return self._rerank_and_format(law, article_ref, concept, candidates_raw, top_k)

    def retrieve_for_prompt(
        self,
        concept: str,
        law: str,
        article_ref: Optional[str] = None,
        top_k: int = 3,
        use_concept_tags: bool = True,
    ) -> str:
        """
        Prompt-ready retrieval — matches the original call signature used by
        run_pipeline.stage_extract():

            retriever.retrieve_for_prompt(concept=concept, law=law, top_k=top_k)

        article_ref is optional.  When omitted (None), retrieval is law-wide
        for the given concept rather than scoped to a single article.
        retrieve() requires article_ref; use retrieve_for_prompt() when calling
        from the pipeline where article context is managed externally.
        """
        return self.retrieve(
            law=law,
            article_ref=article_ref or "",
            concept=concept,
            top_k=top_k,
            use_concept_tags=use_concept_tags,
        )

    # ── Stage 1 helpers ───────────────────────────────────────────────────────

    def _get_candidates(
        self,
        law: str,
        article_ref: str,
        concept: str,
        use_concept_tags: bool,
    ) -> list[dict]:
        """
        Pull candidate chunks from the ChunkStore.

        Two-tier fallback:
        1. Chunks for (law, article_ref) filtered by concept tag.
        2. If empty, all chunks for (law, article_ref).
        3. If still empty (article has no standalone chunks), all chunks for law.
           (Rare; occurs when the chunker groups multiple short articles together.)
        """
        tag_filter = [concept] if use_concept_tags else None

        # When article_ref is empty/None, go straight to law-wide search
        if not article_ref:
            log.debug(f"[retrieve] No article_ref — law-wide search for {concept}@{law}")
            return self._store.get_chunks(law=law, concept_tags=tag_filter)

        # Primary: article-scoped, concept-filtered
        chunks = self._store.get_chunks(
            law=law,
            article_ref=article_ref,
            concept_tags=tag_filter,
        )

        # Fallback 1: drop concept-tag filter, keep article scope
        if not chunks and use_concept_tags:
            log.debug(
                f"[retrieve] Concept-tag filter empty for {concept}@{article_ref} "
                f"— falling back to all chunks in article"
            )
            chunks = self._store.get_chunks(law=law, article_ref=article_ref)

        # Fallback 2: no chunks at all for this article — search law-wide
        if not chunks:
            log.debug(
                f"[retrieve] No chunks for {law}/{article_ref} "
                f"— falling back to all law chunks (law-wide search)"
            )
            chunks = self._store.get_chunks(law=law, concept_tags=tag_filter)

        return chunks

    # ── Stage 2 helpers ───────────────────────────────────────────────────────

    def _rerank_and_format(
        self,
        law: str,
        article_ref: str,
        concept: str,
        candidates_raw: list[dict],
        top_k: int,
    ) -> str:
        """
        Embed the query with BM25, rerank *candidates_raw* by cosine similarity,
        and return the concatenated texts of the top-k results.
        """
        # Build discriminative query text
        expansion = _CONCEPT_QUERY_EXPANSION.get(concept, concept)
        query_text = f"{concept} {article_ref} {expansion}"

        # Get the fitted BM25Embedder for this law
        embedder = self._load_embedder(law)

        if embedder is None:
            # No embedder available — return first top_k chunks in DB order
            log.warning(
                f"[retrieve] No embedder for {law} — returning top-{top_k} by DB order"
            )
            texts = [c["text"] for c in candidates_raw[:top_k]]
            return "\n\n".join(texts)

        # Embed query
        try:
            query_vec = embedder.embed(query_text)
        except Exception as exc:
            log.warning(f"[retrieve] embed() failed for {law}: {exc} — using DB order")
            texts = [c["text"] for c in candidates_raw[:top_k]]
            return "\n\n".join(texts)

        # Build (chunk_id, vector) pairs; skip chunks with NULL vectors
        candidates_for_ranking = [
            (c["id"], c["vector"])
            for c in candidates_raw
            if c.get("vector") is not None
        ]

        if not candidates_for_ranking:
            # All chunks have NULL vectors (e.g. ingest ran without BM25)
            log.debug(
                f"[retrieve] All candidates have NULL vectors for "
                f"{law}/{article_ref}/{concept} — using DB order"
            )
            texts = [c["text"] for c in candidates_raw[:top_k]]
            return "\n\n".join(texts)

        # Cosine rerank
        ranked = rank_by_similarity(query_vec, candidates_for_ranking, top_k=top_k)

        log.debug(
            f"[retrieve] {law}/{article_ref}/{concept}: "
            f"{len(candidates_for_ranking)} candidates → top-{len(ranked)} selected "
            f"(best score={ranked[0][1]:.3f})"
        )

        # Build lookup and assemble output
        text_lookup = {c["id"]: c["text"] for c in candidates_raw}
        texts = [
            text_lookup[chunk_id]
            for chunk_id, _ in ranked
            if chunk_id in text_lookup
        ]
        return "\n\n".join(texts)

    # ── Embedder loading ──────────────────────────────────────────────────────

    def _load_embedder(self, law: str) -> Optional[BM25Embedder]:
        """
        Load and cache the BM25Embedder for *law* from its pickle file.

        Returns None (with a warning) if the path is missing or the pickle
        is unreadable, so retrieval can degrade gracefully instead of crashing.
        """
        if law in self._embedder_cache:
            return self._embedder_cache[law]

        emb_path = self._embedder_paths.get(law)

        if emb_path is None:
            log.warning(f"[retrieve] No embedder path registered for law={law!r}")
            return None

        if not emb_path.exists():
            log.warning(
                f"[retrieve] Embedder pickle not found: {emb_path} "
                f"— did ingest run for {law}?"
            )
            return None

        try:
            with open(emb_path, "rb") as fh:
                embedder: BM25Embedder = pickle.load(fh)
            log.info(f"[retrieve] Loaded BM25 embedder for {law} from {emb_path}")
            self._embedder_cache[law] = embedder
            return embedder
        except Exception as exc:
            log.error(f"[retrieve] Failed to load embedder for {law}: {exc}")
            return None

    def retrieve_chunks(
        self,
        law: str,
        article_ref: str,
        concept: str,
        top_k: int = 3,
        use_concept_tags: bool = True,
    ) -> list[RetrievedChunk]:
        """
        Same as retrieve() but returns structured RetrievedChunk objects
        instead of a concatenated string — useful for inspection and testing.

        Parameters
        ----------
        law, article_ref, concept, top_k, use_concept_tags
            Same as retrieve().

        Returns
        -------
        List of RetrievedChunk sorted by descending similarity score.
        """
        law = law.upper()
        candidates_raw = self._get_candidates(law, article_ref, concept, use_concept_tags)
        if not candidates_raw:
            return []

        expansion = _CONCEPT_QUERY_EXPANSION.get(concept, concept)
        query_text = f"{concept} {article_ref} {expansion}"
        embedder = self._load_embedder(law)

        scored: list[tuple[dict, float]] = []

        if embedder is not None:
            try:
                query_vec = embedder.embed(query_text)
                ranking_input = [
                    (c["id"], c["vector"])
                    for c in candidates_raw
                    if c.get("vector") is not None
                ]
                if ranking_input:
                    ranked_ids = {
                        chunk_id: score
                        for chunk_id, score in rank_by_similarity(query_vec, ranking_input, top_k=top_k)
                    }
                    scored = [
                        (c, ranked_ids[c["id"]])
                        for c in candidates_raw
                        if c["id"] in ranked_ids
                    ]
                    scored.sort(key=lambda x: x[1], reverse=True)
            except Exception as exc:
                log.warning(f"[retrieve_chunks] embed() failed: {exc} — using DB order")

        if not scored:
            scored = [(c, -1.0) for c in candidates_raw[:top_k]]

        return [
            RetrievedChunk(
                chunk_id=c["id"],
                law=c["law"],
                article_ref=c["article_ref"],
                level=c["level"],
                text=c["text"],
                concept_tags=c.get("concept_tags", []),
                score=score,
            )
            for c, score in scored
        ]


# ---------------------------------------------------------------------------
# build_retriever_from_files — convenience constructor used by __init__.py
# ---------------------------------------------------------------------------

def build_retriever_from_files(
    db_path: "str | Path",
    embedder_paths: dict[str, "str | Path"],
) -> Retriever:
    """
    Construct a Retriever that owns its own SQLite connection.

    The Retriever opens the ChunkStore on construction and closes it when
    garbage-collected.  No context manager needed::

        from rag_pipeline import build_retriever_from_files
        retriever = build_retriever_from_files(
            "data/chunks.db",
            {"PIPEDA": "data/chunks_PIPEDA_embedder.pkl"},
        )
        text = retriever.retrieve_for_prompt("LegalBasis", "PIPEDA", top_k=3)

    Parameters
    ----------
    db_path        : Path to the SQLite chunk database.
    embedder_paths : {LAW: path_to_pkl} mapping, same format as stage_ingest().

    Returns
    -------
    Retriever with an open, owned ChunkStore connection.
    """
    return Retriever(store_or_path=db_path, embedder_paths=embedder_paths)