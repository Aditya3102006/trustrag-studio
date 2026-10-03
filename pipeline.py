"""
pipeline.py — Full TrustRAG pipeline orchestration.

Steps
-----
1. Semantic Retrieval   — FAISS + MiniLM (top-k chunks, with similarity scores)
2. Prompt Construction  — context-grounded prompt fed to selected LLM engine
3. Text Generation      — Local Flan-T5, Google Gemini (Free), Groq (Free), or OpenAI
4. Trust Scoring        — per-sentence entropy/certainty + context-similarity → trust score
"""
import time
import nltk

# Ensure punkt tokenizer is available
for tok in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{tok}")
    except LookupError:
        nltk.download(tok, quiet=True)

from nltk.tokenize import sent_tokenize
from retriever import retrieve_with_scores
from generator import (
    generate_with_logits,
    generate_with_gemini,
    generate_with_groq,
    generate_with_openai,
    tokenizer,
)
from scorer import token_entropy, sentence_entropies, sentence_similarity, compute_trust


def run_pipeline(
    query: str,
    documents: list,
    index,
    trust_threshold: float = 0.3,
    top_k: int = 5,
    llm_provider: str = "Local (Flan-T5)",
    api_key: str = "",
) -> tuple:
    """
    Run the full RAG pipeline for a user query using local or cloud LLM.

    Args:
        query:           User's natural-language question.
        documents:       List of text chunks (default KB + uploaded docs).
        index:           FAISS index corresponding to *documents*.
        trust_threshold: Sentences with trust >= this are labelled RELIABLE.
        top_k:           Number of context chunks to retrieve.
        llm_provider:    "Local (Flan-T5)", "Google Gemini (Free API)", "Groq (Free Llama-3)", "OpenAI (GPT-4o-mini)"
        api_key:         Cloud API key if cloud provider is chosen.

    Returns:
        (results, retrieved_docs, pipeline_meta)
    """
    t0 = time.perf_counter()

    # ── 1. Retrieve ────────────────────────────────────────────────────────────
    retrieved_pairs = retrieve_with_scores(query, documents, index, top_k=top_k)
    retrieved        = [doc   for doc, _     in retrieved_pairs]
    retrieval_scores = [score for _,   score in retrieved_pairs]
    context          = "\n\n".join([f"--- Evidence Chunk #{i+1} (Sim: {s:.3f}) ---\n{d}" for i, (d, s) in enumerate(retrieved_pairs)]) if retrieved else "No matching context found."
    t_retrieve = time.perf_counter() - t0

    # ── 2. Construct Prompt & Generate ────────────────────────────────────────
    t1 = time.perf_counter()
    
    if "Local" in llm_provider:
        prompt = (
            "You are a knowledgeable AI assistant. "
            "Using ONLY the context provided below, write a detailed and complete answer "
            "to the question. Explain concepts clearly, do not repeat yourself, "
            "and stay grounded in the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Detailed Answer:"
        )
        answer, scores, token_ids = generate_with_logits(prompt)
    elif "Gemini" in llm_provider:
        prompt = (
            "You are an expert AI research assistant. Answer the user's question with thorough, comprehensive, "
            "and structured technical depth based on the provided document context.\n\n"
            "INSTRUCTIONS:\n"
            "1. Base your direct answers primarily on the retrieved evidence below.\n"
            "2. If the user asks for advancements, features, or architectural recommendations, synthesize practical technical improvements from the context.\n"
            "3. Structure your response with clear headings, bullet points, and concise explanations.\n\n"
            f"### RETRIEVED CONTEXT:\n{context}\n\n"
            f"### USER QUESTION:\n{query}\n\n"
            "### COMPREHENSIVE RESPONSE:"
        )
        answer = generate_with_gemini(prompt, api_key=api_key, model_name="gemini-1.5-flash")
        scores = None
    elif "Groq" in llm_provider:
        prompt = (
            "You are an expert AI research assistant. Answer the user's question with thorough, comprehensive, "
            "and structured technical depth based on the provided document context.\n\n"
            f"### RETRIEVED CONTEXT:\n{context}\n\n"
            f"### USER QUESTION:\n{query}\n\n"
            "### COMPREHENSIVE RESPONSE:"
        )
        answer = generate_with_groq(prompt, api_key=api_key, model_name="llama-3.3-70b-versatile")
        scores = None
    elif "OpenAI" in llm_provider:
        prompt = (
            "You are an expert AI research assistant. Answer the user's question with thorough, comprehensive, "
            "and structured technical depth based on the provided document context.\n\n"
            f"### RETRIEVED CONTEXT:\n{context}\n\n"
            f"### USER QUESTION:\n{query}\n\n"
            "### COMPREHENSIVE RESPONSE:"
        )
        answer = generate_with_openai(prompt, api_key=api_key, model_name="gpt-4o-mini")
        scores = None
    else:
        # Fallback to local
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        answer, scores, _ = generate_with_logits(prompt)

    t_generate = time.perf_counter() - t1

    # ── 3. Score Each Sentence ────────────────────────────────────────────────
    t2 = time.perf_counter()

    raw_sentences = [s.strip() for s in sent_tokenize(answer) if s.strip() and len(s.strip()) > 8]
    if not raw_sentences:
        raw_sentences = [answer] if answer else ["No text generated."]

    if scores is not None and "Local" in llm_provider:
        entropies = token_entropy(scores)
        sent_ent = sentence_entropies(answer, entropies, tokenizer)
        sentences = [s[0] for s in sent_ent] if sent_ent else raw_sentences
        ent_values = [s[1] for s in sent_ent] if sent_ent else [0.2] * len(sentences)
    else:
        # For Cloud LLMs: compute calibrated entropy based on semantic grounding
        sentences = raw_sentences
        sim_values = sentence_similarity(sentences, retrieved) if retrieved else [0.0] * len(sentences)
        # Higher grounding similarity = lower entropy (higher confidence)
        ent_values = [round(max(0.05, (1.0 - sim) * 0.8), 3) for sim in sim_values]

    sim_values = sentence_similarity(sentences, retrieved) if retrieved else [0.0] * len(sentences)

    results = []
    trust_scores = []
    for i, sent in enumerate(sentences):
        sim = sim_values[i] if i < len(sim_values) else 0.0
        ent = ent_values[i] if i < len(ent_values) else 0.2
        trust = compute_trust(sim, ent)
        trust_scores.append(trust)
        label = "✅ RELIABLE" if trust >= trust_threshold else "❌ UNRELIABLE"
        results.append({
            "sentence":    sent,
            "similarity":  round(sim, 3),
            "entropy":     round(ent, 3),
            "trust_score": round(trust, 3),
            "label":       label,
        })

    t_score = time.perf_counter() - t2
    t_total = time.perf_counter() - t0

    reliable_n = sum(1 for r in results if "RELIABLE" in r["label"])
    avg_trust = round(sum(trust_scores) / len(trust_scores), 3) if trust_scores else 0.0
    avg_sim = round(sum(sim_values) / len(sim_values), 3) if sim_values else 0.0
    avg_ent = round(sum(ent_values) / len(ent_values), 3) if ent_values else 0.0
    unreliable_ratio = ((len(results) - reliable_n) / len(results) * 100) if results else 0.0

    pipeline_meta = {
        # Corpus
        "total_chunks_indexed": len(documents),
        # Retrieval
        "chunks_retrieved":     len(retrieved),
        "retrieval_scores":     [round(s, 3) for s in retrieval_scores],
        "avg_retrieval_score":  round(
            sum(retrieval_scores) / len(retrieval_scores), 3
        ) if retrieval_scores else 0.0,
        # Generation
        "llm_provider":         llm_provider,
        "prompt":               prompt,
        "answer_text":          answer,
        "answer_word_count":    len(answer.split()),
        "sentence_count":       len(sentences),
        # Scoring & Evaluation
        "avg_trust_score":      avg_trust,
        "avg_similarity":       avg_sim,
        "avg_entropy":          avg_ent,
        "reliable_count":       reliable_n,
        "unreliable_count":     len(results) - reliable_n,
        "reliability_rate":     round((reliable_n / len(results) * 100), 1) if results else 0.0,
        "hallucination_risk":   round(unreliable_ratio, 1),
        "trust_threshold":      trust_threshold,
        "top_k":                top_k,
        # Timings (seconds)
        "times": {
            "retrieve_s": round(t_retrieve, 2),
            "generate_s": round(t_generate, 2),
            "score_s":    round(t_score, 2),
            "total_s":    round(t_total, 2),
        },
    }

    return results, retrieved, pipeline_meta


def generate_document_roadmap(
    doc_name: str,
    chunks: list,
    full_text: str = "",
    llm_provider: str = "Google Gemini (Free API)",
    api_key: str = "",
) -> tuple:
    """
    Generate a full-scope PRD Analysis, Technical Breakdown & Future Roadmap
    for an uploaded document (e.g., Arcana PRD).

    Returns: (roadmap_markdown, meta_dict)
    """
    t0 = time.perf_counter()

    # Sample top key passages from start, middle, end of document (up to 12 chunks)
    if len(chunks) <= 12:
        sampled_chunks = chunks
    else:
        # Uniform sampling across document
        step = len(chunks) / 12.0
        sampled_chunks = [chunks[int(i * step)] for i in range(12)]

    doc_context = "\n\n---\n\n".join([f"Passage #{i+1}:\n{c}" for i, c in enumerate(sampled_chunks)])

    prompt = (
        f"You are a Principal AI Architect & Senior Product Strategy Specialist. "
        f"Analyze the uploaded document '{doc_name}' and produce a comprehensive, structured, "
        f"and highly technical PRD Analysis & Next-Gen Advancement Roadmap.\n\n"
        f"### DOCUMENT PASSAGES:\n{doc_context}\n\n"
        f"### INSTRUCTIONS:\n"
        f"Produce a detailed technical report formatted in Markdown with the following exact sections:\n\n"
        f"## 1. 🎯 Core Objective & Problem Statement\n"
        f"- Target problem domain and commercial/industrial need.\n"
        f"- Core value proposition and system goal.\n\n"
        f"## 2. ⚙️ Key Functional Architecture & Datasets\n"
        f"- Core components, inspection pipelines, or algorithmic workflows.\n"
        f"- Datasets, annotations, sample counts, and image/data characteristics mentioned.\n\n"
        f"## 3. ⚠️ Technical Bottlenecks, Domain Gaps & Constraints\n"
        f"- Limitations in data diversity, sample scarcity, or synthetic-to-real domain gap.\n"
        f"- Computational latency, manual annotation dependencies, or camera/hardware constraints.\n\n"
        f"## 4. 🚀 Recommended Next-Gen Technical Advancements\n"
        f"Propose 4-5 concrete, state-of-the-art engineering upgrades (e.g., Multi-Modal Vision Transformers, Self-Supervised Defect Pretraining, Edge TensorRT deployment for real-time AOI, Active Learning loops, Diffusion-based Defect Augmentation).\n"
        f"- Explain the rationale and technical architecture for each.\n\n"
        f"## 5. 📊 Implementation Roadmap & Feasibility Matrix\n"
        f"Provide a Markdown table showing: Phase (Near-term, Mid-term, Long-term) | Proposed Upgrade | Expected Impact | Engineering Complexity (Low/Med/High).\n\n"
        f"Provide a thorough, high-depth, professional report."
    )

    t1 = time.perf_counter()
    if "Gemini" in llm_provider:
        roadmap_text = generate_with_gemini(prompt, api_key=api_key, model_name="gemini-3.6-flash")
    elif "Groq" in llm_provider:
        roadmap_text = generate_with_groq(prompt, api_key=api_key, model_name="llama-3.3-70b-versatile")
    elif "OpenAI" in llm_provider:
        roadmap_text = generate_with_openai(prompt, api_key=api_key, model_name="gpt-4o-mini")
    else:
        # Local model fallback
        short_prompt = f"Summarize project {doc_name} and list key features and advancements:\n{doc_context[:1200]}"
        roadmap_text, _, _ = generate_with_logits(short_prompt)

    t_total = time.perf_counter() - t0

    # Calculate overall grounding similarity against the document chunks
    sentences = [s.strip() for s in sent_tokenize(roadmap_text) if len(s.strip()) > 10][:15]
    sims = sentence_similarity(sentences, chunks[:15]) if sentences and chunks else [0.75]
    avg_grounding = round(sum(sims) / len(sims), 3) if sims else 0.80

    meta = {
        "doc_name": doc_name,
        "chunks_analyzed": len(sampled_chunks),
        "total_doc_chunks": len(chunks),
        "word_count": len(roadmap_text.split()),
        "avg_grounding": avg_grounding,
        "generation_time_s": round(t_total, 2),
        "llm_provider": llm_provider,
    }

    return roadmap_text, meta