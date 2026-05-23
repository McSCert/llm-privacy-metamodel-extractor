from __future__ import annotations

"""
rag_pipeline/embedder.py

Three concrete embedders, all satisfying the Embedder protocol:

  TFIDFEmbedder               — original implementation, scikit-learn TF-IDF.
                                 Fast, zero extra deps, but scores near-zero for
                                 terms that appear in almost every article (known
                                 ISSUE-1: Actor, Constraint concepts).

  BM25Embedder                — new implementation, bm25s library.
                                 TF saturation + length penalty fixes ISSUE-1.
                                 Drop-in replacement: same fit/embed/embed_batch/
                                 doc_vectors interface, same pickle round-trip.

  SentenceTransformerEmbedder — semantic dense encoder stub (multilingual-e5-base).
                                 Requires: pip install sentence-transformers
                                 Does NOT need fit() — encode on the fly.
                                 Best retrieval quality; slowest at ingest.

store.py selects the embedder via the EMBEDDER_CLASS constant at the top of
that file.  Swap the constant and re-run --stage ingest; nothing else changes.
"""

import logging
import pickle
import scipy.sparse as sp
import bm25s
from pathlib import Path
from typing import Protocol
import bm25s
import numpy as np

log = logging.getLogger(__name__)


# ── Embedder protocol — the swap-in interface ─────────────────────────────────

class Embedder(Protocol):
    """
    Structural protocol satisfied by all three embedder classes.

    store.py  calls fit() once at ingest, then doc_vectors() to get all
              document embeddings in one shot.
    retriever.py calls embed() per query at extraction time.
    """

    def fit(self, texts: list[str]) -> None:
        """Fit the vectoriser on a corpus of texts (called once at ingest)."""
        ...

    def embed(self, text: str) -> list[float]:
        """Return a dense, L2-normalised float vector for one text."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return L2-normalised vectors for a list of texts."""
        ...

    def doc_vectors(self) -> np.ndarray:
        """
        Return all document vectors as a (n_docs, n_dims) float32 array.
        Called by store.ingest_file() immediately after fit().
        Each row must be L2-normalised so that cosine similarity reduces
        to a plain dot product at retrieval time.
        """
        ...


# ── 1. TF-IDF Embedder (original) ────────────────────────────────────────────

class TFIDFEmbedder:
    """
    Original TF-IDF embedder backed by scikit-learn TfidfVectorizer.

    Produces L2-normalised sparse→dense vectors.  Fast and dependency-light,
    but IDF down-weights terms common across articles (e.g. "organization",
    "shall"), causing near-zero retrieval scores for Actor and Constraint
    concepts (known ISSUE-1 in run_pipeline.py).

    Use BM25Embedder to fix ISSUE-1 without changing any other code.

    Dependencies: scikit-learn, numpy  (both already in requirements.txt)
    """

    def __init__(
        self,
        max_features: int = 8_000,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
    ):
        self._max_features = max_features
        self._ngram_range = ngram_range
        self._sublinear_tf = sublinear_tf
        self._vectorizer = None
        self._doc_matrix = None   # (n_docs, n_features) dense float32, L2-normalised
        self._fitted = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> None:
        """Fit TF-IDF on the full chunk corpus and store the document matrix."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        log.info(f"Fitting TF-IDF on {len(texts)} texts")
        self._vectorizer = TfidfVectorizer(
            max_features=self._max_features,
            ngram_range=self._ngram_range,
            sublinear_tf=self._sublinear_tf,
        )
        sparse = self._vectorizer.fit_transform(texts)          # (n_docs, n_features)
        normalised = normalize(sparse, norm="l2")               # L2 per row
        self._doc_matrix = np.array(normalised.todense(), dtype=np.float32)
        self._fitted = True
        log.info(f"TF-IDF vocabulary size: {len(self._vectorizer.vocabulary_)} terms")

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Transform and L2-normalise one query text."""
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")
        from sklearn.preprocessing import normalize
        vec = self._vectorizer.transform([text])
        norm_vec = normalize(vec, norm="l2")
        return np.array(norm_vec.todense(), dtype=np.float32).flatten().tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Transform and L2-normalise a batch of texts."""
        if not self._fitted:
            raise RuntimeError("Call fit() before embed_batch()")
        from sklearn.preprocessing import normalize
        vecs = self._vectorizer.transform(texts)
        norm_vecs = normalize(vecs, norm="l2")
        return np.array(norm_vecs.todense(), dtype=np.float32).tolist()

    def doc_vectors(self) -> np.ndarray:
        """Return the precomputed (n_docs, n_features) L2-normalised matrix."""
        if not self._fitted:
            raise RuntimeError("Call fit() before doc_vectors()")
        return self._doc_matrix


# ── 2. BM25 Embedder (new — fixes ISSUE-1) ───────────────────────────────────

class BM25Embedder:
    """
    BM25 embedder that produces L2-normalised document and query vectors.

    Fixes known ISSUE-1 (TF-IDF zero scores for Actor / Constraint concepts):
    BM25's TF saturation and document-length penalty give reasonable scores
    even for terms that appear in nearly every article of a law document.

    At ingest (fit + doc_vectors):
      bm25s builds a (n_docs, n_vocab) sparse score matrix.  Each row is the
      BM25 score profile for one document.  doc_vectors() L2-normalises each
      row and returns a dense array for storage in SQLite.

    At query time (embed):
      The query is tokenised → binary indicator vector (1.0 for in-vocab
      tokens, 0.0 otherwise) → L2-normalised.  The dot product of this vector
      with a stored document row equals the standard BM25 relevance score, so
      cosine similarity == BM25 ranking when both sides are normalised.

    Parameters
    ----------
    k1 : TF saturation (default 1.2 — BM25 standard)
    b  : length penalty (default 0.75 — BM25 standard)

    Dependencies: bm25s, scipy, numpy
      pip install bm25s scipy
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self._bm25: bm25s.BM25
        self._fitted = False
        self.k1 = k1
        self.b = b

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> None:
        """Fit BM25 index on a corpus of texts.  Must be called before embed()."""
        log.info(f"Fitting BM25 on {len(texts)} texts")
        tokens = bm25s.tokenize(texts)
        self._bm25 = bm25s.BM25(k1=self.k1, b=self.b)
        self._bm25.index(tokens)
        self._build_score_matrix()
        self._fitted = True
        log.info(f"BM25 vocabulary size: {self._n_vocab} terms")

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """
        Return a normalised dense float vector for one query text.

        Tokens present in the vocabulary produce a 1.0; all others are 0.0.
        L2-normalised so dot product with a stored doc row == BM25 score.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")
        query_tokens = bm25s.tokenize([text], return_ids=False)
        vec = np.zeros(self._n_vocab, dtype=np.float32)
        for token in query_tokens[0]:
            token_id = self._bm25.vocab_dict.get(token)
            if token_id is not None and token_id < self._n_vocab:
                vec[token_id] = 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm > 0 else vec).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed — tokenises all texts in one pass, then vectorises."""
        if not self._fitted:
            raise RuntimeError("Call fit() before embed_batch()")
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

    def doc_vectors(self) -> np.ndarray:
        """
        Return L2-normalised BM25 document score matrix as a (n_docs, n_vocab)
        float32 dense array.  Row i == normalised BM25 score profile for chunk i.
        Called by store.ingest_file() to persist per-chunk vectors to SQLite.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before doc_vectors()")
        dense = np.array(self._scores_matrix.todense(), dtype=np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save BM25 index to a directory using bm25s native format."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(path), corpus=None)
        log.info(f"BM25 index saved to {path}/")

    @classmethod
    def load(cls, path: Path) -> "BM25Embedder":
        """Load a BM25 index saved with save()."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index directory not found: {path}")
        instance = cls.__new__(cls)
        instance._bm25 = bm25s.BM25.load(str(path), load_corpus=False)
        instance._fitted = True
        instance._build_score_matrix()
        log.info(f"BM25 index loaded from {path}/")
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
    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_score_matrix(self) -> None:
        """
        Build a (n_docs × n_vocab) scipy CSC sparse matrix from the raw
        bm25s score arrays.  Excludes the empty-string artefact key.
        """
        scores = self._bm25.scores
        self._n_vocab = len(self._bm25.vocab_dict) - 1   # exclude empty string
        n_docs = scores["num_docs"]
        self._scores_matrix = sp.csc_matrix(
            (scores["data"], scores["indices"], scores["indptr"]),
            shape=(n_docs, self._n_vocab),
        )


# ── 3. SentenceTransformer Embedder (stub — future work) ─────────────────────

class SentenceTransformerEmbedder:
    """
    Dense semantic encoder backed by a SentenceTransformer model.

    Does NOT require fit() — the pre-trained model encodes on the fly.
    Calling fit() is a no-op so the class is still a valid drop-in for
    store.ingest_file(), which always calls fit() before doc_vectors().

    Recommended model: "intfloat/multilingual-e5-base"
      • 768-dim, supports 100+ languages, strong on legal text.
      • ~1 GB on disk; runs on CPU but much faster on GPU.

    Alternative: "BAAI/bge-small-en-v1.5"  (384-dim, English only, very fast)

    Dependencies: sentence-transformers
      pip install sentence-transformers

    Usage — swap the constant in store.py:
      EMBEDDER_CLASS = SentenceTransformerEmbedder
      # optionally pass model_name:
      EMBEDDER_CLASS = lambda: SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self._model_name = model_name
        self._model = None
        self._doc_matrix: np.ndarray | None = None
        self._fitted = False   # always True after fit() no-op

    def _get_model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed.\n"
                    "Run: pip install sentence-transformers"
                )
            log.info(f"Loading SentenceTransformer model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # ── Fitting (no-op — pre-trained model needs no corpus) ───────────────────

    def fit(self, texts: list[str]) -> None:
        """
        Pre-encodes all corpus texts and caches the matrix so doc_vectors()
        can return it instantly without re-encoding.
        """
        log.info(
            f"SentenceTransformerEmbedder: pre-encoding {len(texts)} texts "
            f"with {self._model_name}"
        )
        model = self._get_model()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        self._doc_matrix = np.array(vecs, dtype=np.float32)
        self._fitted = True
        log.info(f"Encoded {len(texts)} texts → shape {self._doc_matrix.shape}")

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Encode and L2-normalise one text."""
        model = self._get_model()
        vec = model.encode([text], normalize_embeddings=True)
        return np.array(vec[0], dtype=np.float32).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode and L2-normalise a batch of texts."""
        model = self._get_model()
        vecs = model.encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32).tolist()

    def doc_vectors(self) -> np.ndarray:
        """Return the pre-encoded document matrix cached by fit()."""
        if self._doc_matrix is None:
            raise RuntimeError("Call fit() before doc_vectors()")
        return self._doc_matrix


# ── Similarity utilities ──────────────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Cosine similarity between two dense float vectors.

    All three embedders produce L2-normalised vectors, so this is just a
    dot product.  The explicit norm division is retained as a safety fallback
    for unnormalised inputs.
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
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
    Re-rank (chunk_id, vector) candidates by cosine similarity to query_vec.

    Works identically for all three embedders — the vector space differs
    (sparse TF-IDF / BM25 score space / dense semantic space) but the
    cosine ranking logic is embedder-agnostic.

    Returns list of (chunk_id, score) sorted descending, truncated to top_k.
    """
    scored = [
        (chunk_id, cosine_similarity(query_vec, vec))
        for chunk_id, vec in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ── Hybrid retrieval notes (future work) ─────────────────────────────────────
#
# To combine BM25 + SentenceTransformer scores:
#
# 1. Fit both embedders on the same corpus.
# 2. Embed the query with both → bm25_q_vec, dense_q_vec.
# 3. Both are already L2-normalised → scores are on [0, 1].
# 4. Fuse with:
#    a) Reciprocal Rank Fusion (RRF):
#         score_rrf(d) = Σ_i  1 / (k + rank_i(d))    k=60 is a common default
#    b) Weighted sum:
#         score(d) = 0.3 * bm25_score(d) + 0.7 * dense_score(d)
#       (legal text: BM25 contributes keyword precision, dense adds semantics)