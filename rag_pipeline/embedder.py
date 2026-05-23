"""
embedder.py — Chunk embedding for the RAG pipeline.

Strategy: TF-IDF with sublinear TF scaling, fitted per law corpus.
-----------------------------------------------------------------
Why TF-IDF and not a transformer model?
  1. No network access needed — the pipeline runs fully offline.
  2. Deterministic — the same text always produces the same vector,
     which matters for research reproducibility.
  3. Legal text is domain-specific and keyword-rich. TF-IDF handles
     Latin legal terminology (e.g. "consentimento", "tratamento") and
     statutory references ("Art.6(1)(a)") without fine-tuning.

Swap-in point for sentence-transformers
  Replace TFIDFEmbedder with SentenceTransformerEmbedder (stub below)
  and the rest of the pipeline is unchanged — both implement embed().

Similarity metric: cosine similarity (via sklearn).
Vector storage: JSON array (portable, SQLite-compatible).
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Protocol
import bm25s
import numpy as np

log = logging.getLogger(__name__)


# ── Embedder protocol — the swap-in interface ─────────────────────────────────

class Embedder(Protocol):
    """
    Any embedder must implement these two methods.
    The retriever calls embed() at query time; fit() is called once at ingest.
    """

    def fit(self, texts: list[str]) -> None:
        """Fit the vectoriser on a corpus of texts."""
        ...

    def embed(self, text: str) -> list[float]:
        """Return a dense float vector for a single text."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return vectors for a list of texts (may be more efficient than looping)."""
        ...


# ── TF-IDF Embedder (default) ─────────────────────────────────────────────────

class TFIDFEmbedder:
    """
    TF-IDF vectoriser with sublinear TF scaling and L2 normalisation.
    Produces sparse vectors stored as dense JSON arrays (truncated to
    `max_features` dimensions to keep storage manageable).

    Parameters
    ----------
    max_features  : vocabulary size (top N terms by corpus frequency)
    ngram_range   : (1, 2) captures both unigrams and bigrams —
                    important for legal phrases like "legal obligation",
                    "legitimate interest", "standard contractual clauses"
    """

    def __init__(self, max_features: int = 4096, ngram_range: tuple = (1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,          # log(1+tf) — reduces dominance of high-freq terms
            strip_accents="unicode",    # handles LGPD Portuguese accents
            analyzer="word",
            min_df=1,
        )
        self._fitted = False
        self.max_features = max_features

    def fit(self, texts: list[str]) -> None:
        """Fit the vectoriser on the corpus. Must be called before embed()."""
        log.info(f"Fitting TF-IDF on {len(texts)} texts (max_features={self.max_features})")
        self._vectorizer.fit(texts)
        self._fitted = True
        vocab_size = len(self._vectorizer.vocabulary_)
        log.info(f"Vocabulary size: {vocab_size} terms")

    def embed(self, text: str) -> list[float]:
        """Return a normalised dense float vector for one text."""
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")
        vec = self._vectorizer.transform([text])
        # L2 normalise for cosine similarity via dot product
        norm = np.linalg.norm(vec.toarray())
        dense = (vec.toarray()[0] / norm if norm > 0 else vec.toarray()[0])
        return dense.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed — more efficient than calling embed() in a loop."""
        if not self._fitted:
            raise RuntimeError("Call fit() before embed_batch()")
        matrix = self._vectorizer.transform(texts).toarray()
        # L2 normalise each row
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = matrix / norms
        return normed.tolist()

    def save(self, path: Path) -> None:
        """Persist the fitted vectoriser to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._vectorizer, f)
        log.info(f"Embedder saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "TFIDFEmbedder":
        """Load a previously fitted vectoriser from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Embedder not found: {path}")
        instance = cls.__new__(cls)
        with open(path, "rb") as f:
            instance._vectorizer = pickle.load(f)
        instance._fitted = True
        instance.max_features = instance._vectorizer.max_features
        log.info(f"Embedder loaded from {path}")
        return instance

# ── BM25 Embedder ─────────────────────────────────────────────────
class BM25Embedder:
    """
    BM25 embedder that produces L2 normalized query vectors. 

    Parameters
    ----------
    k1  : TF saturation, how quickly more occurances of the same token lose value (default 1.2)
    b   : length penatlty, how much longer documents are penalized relative to average doc legnth (default 0.75)
    """
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self._bm25: bm25s.BM25 
        self._fitted = False
        self.k1 = k1
        self.b = b

    def fit(self, texts: list[str]) -> None:
        """Fit BM25 index on a corpus of texts. Must be called before embed()."""
        log.info(f"Fitting BM25 on {len(texts)} texts")

        # tokenize text (required for BM25)
        tokens = bm25s.tokenize(texts)

        # create BM25 index 
        self._bm25 = bm25s.BM25(k1=self.k1, b=self.b)
        self._bm25.index(tokens)

        # create matrix of scores
        self._create_score_matrix()
        self._fitted = True
        log.info(f"BM25 vocabulary size: {self._n_vocab} terms")

    def embed(self, text: str) -> list[float]:
        """Return a normalised dense float vector for one text."""
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")
        
        # tokenize
        query_tokens = bm25s.tokenize([text], return_ids=False)

        # construct indicator vector
        vec = np.zeros(self._n_vocab, dtype=np.float32)
        for token in query_tokens[0]:
            token_id = self._bm25.vocab_dict.get(token)
            if token_id is not None and token_id < self._n_vocab:
                vec[token_id] = 1.0

        # normalize
        norm = np.linalg.norm(vec)
        dense = (vec / norm if norm > 0 else vec)
        return dense.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed — more efficient than calling embed() in a loop."""
        # call tokenize once for all texts for efficiency
        token_lists = bm25s.tokenize(texts, return_ids=False)

        matrix = np.zeros((len(texts), self._n_vocab), dtype=np.float32)
        for i, tokens in enumerate(token_lists):
            for token in tokens:
                token_id = self._bm25.vocab_dict.get(token)
                if token_id is not None and token_id < self._n_vocab:
                    matrix[i, token_id] = 1.0

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).tolist()

    def save(self, path: Path) -> None:
        """Persist BM25 index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(path), corpus=None)
        log.info(f"BM25 embedder saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "BM25Embedder":
        """Load a previously fitted BM25 index from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 embedder not found: {path}")
        instance = cls.__new__(cls)
        instance._bm25 = bm25s.BM25.load(str(path), load_corpus=False)
        instance._fitted = True
        instance._create_score_matrix()
        log.info(f"BM25 embedder loaded from {path}")
        return instance

    def _create_score_matrix(self):
        import scipy.sparse as sp
        """BM25 specific helper method that builds a scipy sparse matrix of scores"""
        scores = self._bm25.scores
        self._n_vocab = len(self._bm25.vocab_dict) - 1 #exclude empty string
        n_docs = scores["num_docs"]
        self._scores_matrix = sp.csc_matrix(
            (scores["data"], scores["indices"], scores["indptr"]),
            shape=(n_docs, self._n_vocab),
        )

# ── Sentence-Transformer stub (swap-in when network available) ────────────────
class SentenceTransformerEmbedder:
    """
    Drop-in replacement for TFIDFEmbedder using sentence-transformers.

    Usage (when network/GPU available):
        pip install sentence-transformers
        embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-base")
        # multilingual-e5 handles GDPR (English), LGPD (Portuguese), etc.

    The fit() method is a no-op — transformer models don't need corpus fitting.
    Replace TFIDFEmbedder with this class in ingest.py and retriever.py.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "pip install sentence-transformers"
                )

    def fit(self, texts: list[str]) -> None:
        """No-op — transformer models don't require corpus fitting."""
        self._load()

    def embed(self, text: str) -> list[float]:
        self._load()
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._load()
        return self._model.encode(
            texts, normalize_embeddings=True, batch_size=32,
        ).tolist()

    def save(self, path: Path) -> None:
        """Model name is sufficient to reload — save as a text file."""
        Path(path).write_text(self.model_name)

    @classmethod
    def load(cls, path: Path) -> "SentenceTransformerEmbedder":
        model_name = Path(path).read_text().strip()
        return cls(model_name)


# ── Cosine similarity helper ──────────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Cosine similarity between two L2-normalised vectors.
    Since both are normalised, this is just the dot product.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def rank_by_similarity(
    query_vec: list[float],
    candidates: list[tuple[str, list[float]]],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Re-rank a list of (chunk_id, vector) candidates by cosine similarity
    to the query vector.

    Returns list of (chunk_id, score) sorted descending.
    """
    scored = [
        (chunk_id, cosine_similarity(query_vec, vec))
        for chunk_id, vec in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
