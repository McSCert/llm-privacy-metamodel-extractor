"""
rag_pipeline — RAG chunking and retrieval for the privacy extraction pipeline.

Usage
-----
    # Ingest a law
    from rag_pipeline.store import ingest_file
    result = ingest_file("gdpr.pdf", law="GDPR", db_path="data/chunks.db")

    # Retrieve chunks for a concept
    from rag_pipeline.retriever import Retriever
    retriever = Retriever("data/chunks.db", {"GDPR": "data/chunks_GDPR_embedder.pkl"})
    chunks = retriever.retrieve("LegalBasis", "GDPR", top_k=3)

    # Get prompt-ready text
    text = retriever.retrieve_for_prompt("LegalBasis", "GDPR", top_k=3)

Files
-----
    chunker.py   — PDF parsing and hierarchical chunk splitting
    embedder.py  — TF-IDF embedder (swap-in point for sentence-transformers)
    store.py     — SQLite chunk store and ingest pipeline
    retriever.py — Two-stage retrieval: concept tag filter + cosine similarity
"""

from .chunker  import Chunk, chunk_file, chunk_text, concept_tagger
from .embedder import TFIDFEmbedder, SentenceTransformerEmbedder, cosine_similarity
from .store    import ChunkStore, ingest_file
from .retriever import Retriever, RetrievedChunk, build_retriever_from_files, CONCEPT_QUERIES

__all__ = [
    "Chunk", "chunk_file", "chunk_text", "concept_tagger",
    "TFIDFEmbedder", "SentenceTransformerEmbedder", "cosine_similarity",
    "ChunkStore", "ingest_file",
    "Retriever", "RetrievedChunk", "build_retriever_from_files", "CONCEPT_QUERIES",
]