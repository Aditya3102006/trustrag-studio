"""
retriever.py — FAISS-backed semantic retriever for TrustRAG.

Hosts a shared SentenceTransformer singleton (all-MiniLM-L6-v2) so that
scorer.py can import get_sbert() and reuse the same model — avoiding
double memory allocation.
"""
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ── Shared model singleton ─────────────────────────────────────────────────────
_sbert = SentenceTransformer('all-MiniLM-L6-v2')


def get_sbert():
    """Return the shared SentenceTransformer instance (used by scorer.py)."""
    return _sbert


# ── Index building ─────────────────────────────────────────────────────────────

def build_index(documents: list):
    """
    Encode a list of text strings and build a FAISS IndexFlatIP.
    Vectors are L2-normalised so inner-product == cosine similarity.

    Returns: (index, embeddings_array)
    """
    if not documents:
        return None, None

    embeddings = _sbert.encode(
        documents,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve(query: str, documents: list, index, top_k: int = 5) -> list:
    """Return top_k most relevant documents (strings only, deduplicated)."""
    pairs = retrieve_with_scores(query, documents, index, top_k)
    return [doc for doc, _ in pairs]


def retrieve_with_scores(query: str, documents: list, index, top_k: int = 5) -> list:
    """
    Return a list of (doc_text, cosine_similarity) tuples,
    sorted descending by similarity, deduplicated.
    """
    if index is None or not documents:
        return []

    q_emb = _sbert.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    actual_k = min(top_k, len(documents))
    scores, indices = index.search(q_emb, actual_k)

    seen = set()
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(documents):
            continue
        doc = documents[idx]
        if doc not in seen:
            seen.add(doc)
            results.append((doc, float(score)))

    return results