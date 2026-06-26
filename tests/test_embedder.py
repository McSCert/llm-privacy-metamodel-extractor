# test_embedder.py
import pickle, tempfile
from pathlib import Path
from rag_pipeline.embedder import BM25Embedder, cosine_similarity, rank_by_similarity

docs = [
    "The organization must obtain consent before collecting personal data.",
    "Data subjects have the right to access their personal information.",
    "Personal data shall not be retained longer than necessary for the purpose.",
    "Cross-border transfer requires an adequacy decision or appropriate safeguards.",
    "The data controller is responsible for compliance with this regulation.",
]

emb = BM25Embedder()
emb.fit(docs)

# 1. Vocabulary built
assert emb._fitted
print(f"vocab size: {emb._n_vocab}")

# 2. doc_vectors shape matches corpus
vecs = emb.doc_vectors()
assert vecs.shape == (5, emb._n_vocab), f"Bad shape: {vecs.shape}"
print(f"doc_vectors shape: {vecs.shape}  OK")

# 3. All rows are L2-normalised (norm ≈ 1.0)
import numpy as np
norms = np.linalg.norm(vecs, axis=1)
assert np.allclose(norms, 1.0, atol=1e-5), f"Not normalised: {norms}"
print("doc_vectors are L2-normalised  OK")

# 4. Query embedding has the right length
q = emb.embed("consent personal data")
assert len(q) == emb._n_vocab
print(f"embed() length: {len(q)}  OK")

# 5. Semantic ranking — "consent" query should rank doc[0] highest
candidates = [(f"doc{i}", vecs[i].tolist()) for i in range(5)]
ranked = rank_by_similarity(q, candidates, top_k=3)
print(f"Top-3 for 'consent personal data': {[r[0] for r in ranked]}")
assert ranked[0][0] == "doc0", f"Expected doc0 first, got {ranked[0][0]}"
print("Semantic ranking correct  OK")

# 6. Pickle round-trip (what store.py does)
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "emb.pkl"
    with open(path, "wb") as f:
        pickle.dump(emb, f)
    with open(path, "rb") as f:
        emb2 = pickle.load(f)
    q2 = emb2.embed("consent personal data")
    assert q == q2
    print("Pickle round-trip  OK")

print("\nAll embedder tests passed.")