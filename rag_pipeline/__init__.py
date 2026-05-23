"""
rag_pipeline — RAG chunking and retrieval for the privacy extraction pipeline.

Usage
-----
    # Ingest a law
    from rag_pipeline.store import ingest_file
    result = ingest_file("gdpr.pdf", law="GDPR", db_path="data/chunks.db")

    # Retrieve chunks for a concept (returns plain text for LLM prompt)
    from rag_pipeline.retriever import Retriever
    store, retriever = build_retriever_from_files(
        "data/chunks.db",
        {"GDPR": "data/chunks_GDPR_embedder.pkl"},
    )
    with store:
        text   = retriever.retrieve("GDPR", "Art.6", "LegalBasis", top_k=3)
        chunks = retriever.retrieve_chunks("GDPR", "Art.6", "LegalBasis", top_k=3)

    # Switch embedder (one line in store.py)
    from rag_pipeline.embedder import TFIDFEmbedder, BM25Embedder
    # edit store.EMBEDDER_CLASS = TFIDFEmbedder  ← then re-run --stage ingest

Files
-----
    chunker.py   — PDF parsing and hierarchical chunk splitting
    embedder.py  — TFIDFEmbedder / BM25Embedder / SentenceTransformerEmbedder
    store.py     — SQLite chunk store, ingest pipeline, EMBEDDER_CLASS selector
    retriever.py — Two-stage retrieval: concept tag filter + cosine similarity
"""

from .chunker  import Chunk, chunk_file, chunk_text, concept_tagger
from .embedder import (
    TFIDFEmbedder,
    BM25Embedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
    rank_by_similarity,
)
from .store    import ChunkStore, ingest_file, EMBEDDER_CLASS
from .retriever import (
    Retriever,
    RetrievedChunk,
    build_retriever_from_files,
    CONCEPT_QUERIES,
)

__all__ = [
    # chunker
    "Chunk", "chunk_file", "chunk_text", "concept_tagger",
    # embedder
    "TFIDFEmbedder", "BM25Embedder", "SentenceTransformerEmbedder",
    "cosine_similarity", "rank_by_similarity",
    # store
    "ChunkStore", "ingest_file", "EMBEDDER_CLASS",
    # retriever
    "Retriever", "RetrievedChunk", "build_retriever_from_files", "CONCEPT_QUERIES",
]