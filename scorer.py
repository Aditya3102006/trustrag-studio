"""
scorer.py — Sentence-level trust scoring for TrustRAG.

Imports the shared SBERT model from retriever.py to avoid loading it twice.
"""
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import util

# Reuse the model that retriever.py already loaded — no double memory cost
from retriever import get_sbert

_sbert = get_sbert()


# ── Token-level entropy ────────────────────────────────────────────────────────

def token_entropy(scores: list) -> list:
    """
    Compute Shannon entropy for each generated token position.

    Args:
        scores: list of logit tensors (one per generated token) from model.generate()

    Returns:
        List of float entropy values.
    """
    entropies = []
    for score in scores:
        probs = torch.softmax(score[0], dim=-1)
        h = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        entropies.append(h)
    return entropies


# ── Sentence-level entropy aggregation ────────────────────────────────────────

def sentence_entropies(text: str, token_entropies: list, tokenizer) -> list:
    """
    Map token-level entropies onto NLTK sentences by counting tokens per sentence.

    Returns: list of (sentence_text, mean_entropy) tuples.
    """
    sentences = sent_tokenize(text)
    results = []
    token_pos = 0
    for sent in sentences:
        n_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
        chunk = token_entropies[token_pos : token_pos + n_tokens]
        # Guard: fall back to 0 if alignment drifts (e.g. BPE edge cases)
        sent_entropy = float(np.mean(chunk)) if chunk else 0.0
        token_pos += n_tokens
        results.append((sent, sent_entropy))
    return results


# ── Sentence similarity to retrieved context ───────────────────────────────────

def sentence_similarity(sentences: list, retrieved_docs: list) -> list:
    """
    Compute cosine similarity between each generated sentence and the retrieved chunks.
    Uses max similarity across retrieved chunks (batch computed via SBERT matrix multiplication)
    so that adding more chunks (higher top-k) does NOT dilute the score of a sentence
    that is directly grounded in a specific chunk.

    Returns: list of float similarity scores in [0, 1].
    """
    if not retrieved_docs or not sentences:
        return [0.0] * len(sentences)

    chunk_embs = _sbert.encode(retrieved_docs, convert_to_tensor=True)
    sent_embs = _sbert.encode(sentences, convert_to_tensor=True)

    # cos_matrix shape: [num_sentences, num_chunks]
    cos_matrix = util.cos_sim(sent_embs, chunk_embs)

    # Max similarity across all retrieved chunks for each sentence
    max_sims = torch.max(cos_matrix, dim=1).values.cpu().tolist()
    return [round(max(0.0, float(s)), 3) for s in max_sims]


# ── Trust score ────────────────────────────────────────────────────────────────

def compute_trust(
    similarity: float,
    entropy: float,
    sim_weight: float = 0.6,
    ent_weight: float = 0.4,
) -> float:
    """
    Combine semantic similarity and model confidence into a single trust score.

      trust = sim_weight * similarity + ent_weight * (1 / (1 + entropy))

    Higher similarity  → answer is grounded in retrieved context.
    Lower entropy      → model was more confident (deterministic).
    """
    normalized_entropy = 1.0 / (1.0 + entropy)
    return sim_weight * similarity + ent_weight * normalized_entropy