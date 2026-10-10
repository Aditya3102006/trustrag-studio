"""
app.py — TrustRAG: Multi-Engine Sentence-Level Grounding & Trust Evaluation for RAG

Theme: Deep Navy / Electric Blue / Crimson & Soft Rose Accents.
Features:
- Dual Engine Mode: Local Offline (Flan-T5 with Shannon Entropy) & Cloud Frontier (Google Gemini Free Tier, Groq Free Tier, OpenAI)
- Free Tier Cloud API Integration with zero-cost keys from Google AI Studio
- Dynamic Document Ingestion & FAISS Vector Indexing
- High-Contrast KPI Dashboard & Balanced Full-Width Evaluation Layout
"""
import os
import importlib
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

import pipeline
importlib.reload(pipeline)
from pipeline import run_pipeline, generate_document_roadmap

import chunker
importlib.reload(chunker)
from chunker import process_document

import retriever
importlib.reload(retriever)
from retriever import build_index

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrustRAG — Multi-Engine AI Trust Platform",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT KNOWLEDGE BASE (Curated ML / AI / RAG concepts)
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_DOCS = [
    "Artificial Intelligence is the simulation of human intelligence in machines that are programmed to think, learn, and make decisions like humans.",
    "Machine learning is a field of artificial intelligence that enables systems to learn patterns from data without being explicitly programmed.",
    "Supervised learning trains models on labeled data where the correct output is known for each input.",
    "Unsupervised learning finds hidden patterns in data without any labeled examples.",
    "Semi-supervised learning uses a small amount of labeled data combined with a large amount of unlabeled data.",
    "Self-supervised learning generates its own supervisory signal from the structure of the input data.",
    "Overfitting occurs when a model learns noise in the training data and fails to generalize to new examples.",
    "Underfitting happens when a model is too simple to capture the underlying pattern in the data.",
    "Regularization techniques like L1 and L2 penalize large weights to reduce overfitting.",
    "Cross-validation splits data into multiple folds to give a more reliable estimate of model performance.",
    "Bias-variance tradeoff describes the tension between a model that is too simple and one that is too complex.",
    "Deep learning is a subset of machine learning that uses multi-layer neural networks to model complex patterns.",
    "A neural network consists of layers of interconnected nodes that transform input data through learned weights.",
    "Backpropagation computes gradients of the loss with respect to each weight using the chain rule.",
    "Stochastic gradient descent updates model weights using the gradient estimated from a small batch of data.",
    "Activation functions like ReLU, sigmoid, and tanh introduce non-linearity into neural networks.",
    "Batch normalization normalizes layer activations to stabilize and accelerate training.",
    "Dropout randomly sets a fraction of activations to zero during training to prevent overfitting.",
    "Convolutional neural networks use shared filters to detect spatial patterns in images.",
    "Recurrent neural networks process sequential data by maintaining a hidden state across time steps.",
    "Long short-term memory networks address the vanishing gradient problem in recurrent networks.",
    "Residual networks use skip connections to allow gradients to flow more easily through deep architectures.",
    "Generative adversarial networks consist of a generator and discriminator trained in opposition to each other.",
    "Variational autoencoders learn a compressed latent representation and can generate new samples.",
    "Transformers are neural network architectures that rely on self-attention mechanisms for processing sequences.",
    "Self-attention allows each token in a sequence to attend to all other tokens, capturing long-range dependencies.",
    "Multi-head attention runs several attention operations in parallel and concatenates their outputs.",
    "Positional encodings give the transformer information about the order of tokens in a sequence.",
    "BERT is a transformer-based model pre-trained with masked language modeling for understanding context.",
    "GPT models are autoregressive transformers that generate text one token at a time from left to right.",
    "Large Language Models are neural networks trained on massive text corpora to perform language-related tasks.",
    "Instruction tuning fine-tunes language models on examples of tasks described in natural language.",
    "RLHF uses human feedback to fine-tune language models to follow instructions more helpfully.",
    "Chain-of-thought prompting encourages language models to reason step by step before giving a final answer.",
    "In-context learning allows language models to adapt to new tasks given only a few examples in the prompt.",
    "Token prediction is the fundamental training objective of autoregressive language models.",
    "Temperature controls the randomness of language model outputs by scaling the logit distribution.",
    "Top-k and top-p sampling constrain which tokens are considered at each decoding step.",
    "Retrieval-Augmented Generation combines information retrieval with text generation to improve factual accuracy.",
    "A retriever searches a knowledge base and returns the most relevant documents for a given query.",
    "A generator uses retrieved information and the user query to produce a final grounded response.",
    "RAG systems reduce hallucinations by providing supporting evidence before text generation.",
    "Dense retrieval uses vector representations instead of keyword matching to find relevant documents.",
    "Sparse retrieval methods like BM25 rely on term frequency and inverse document frequency.",
    "Hybrid retrieval combines sparse and dense methods to balance keyword precision and semantic recall.",
    "Reranking re-scores retrieved documents with a more powerful model before passing them to the generator.",
    "Chunking splits long documents into smaller passages so the retriever can match at a finer granularity.",
    "Knowledge-grounded generation improves reliability and trustworthiness of AI system responses.",
    "Context window limits how much retrieved text a language model can process in a single call.",
    "Multi-hop retrieval performs several retrieval steps to answer questions requiring multiple facts.",
    "Sentence Transformers convert text into dense vector embeddings that capture semantic meaning.",
    "Embeddings are numerical vector representations of text that preserve semantic relationships.",
    "Vector databases store embeddings and support efficient approximate nearest-neighbor search.",
    "FAISS is an open-source library developed by Meta for efficient similarity search on dense vectors.",
    "Cosine similarity measures the angle between two vectors and is commonly used to compare embeddings.",
    "Dot product similarity is equivalent to cosine similarity when vectors are L2-normalized.",
    "Approximate nearest neighbor algorithms trade a small accuracy loss for much faster search at scale.",
    "HNSW is a graph-based index structure that enables fast approximate nearest-neighbor lookup.",
    "Dimensionality of embeddings controls the trade-off between expressiveness and computational cost.",
    "Semantic search retrieves documents based on meaning rather than exact word overlap.",
    "Natural Language Processing focuses on enabling computers to understand and generate human language.",
    "Tokenization splits raw text into smaller units such as words, subwords, or characters.",
    "Byte-pair encoding is a subword tokenization algorithm that balances vocabulary size and coverage.",
    "Named entity recognition identifies and classifies entities such as people, places, and organizations.",
    "Part-of-speech tagging assigns grammatical labels like noun, verb, or adjective to each token.",
    "Dependency parsing analyzes the grammatical structure and relationships between words in a sentence.",
    "Coreference resolution determines which noun phrases in a text refer to the same real-world entity.",
    "Sentiment analysis classifies the emotional tone of a piece of text as positive, negative, or neutral.",
    "Text summarization condenses a long document into a shorter version while preserving key information.",
    "Machine translation converts text from one natural language to another.",
    "Question answering systems extract or generate an answer given a question and a supporting passage.",
    "Explainable AI aims to make machine learning model decisions more transparent and interpretable.",
    "SHAP explains model predictions by estimating the marginal contribution of each input feature.",
    "LIME explains individual predictions by approximating a complex model with a simpler local surrogate.",
    "Attention weights in transformers are sometimes used as a proxy for feature importance, though imperfectly.",
    "Model cards document the intended use, performance, and limitations of a machine learning model.",
    "Algorithmic fairness requires that model predictions do not systematically disadvantage protected groups.",
    "Counterfactual explanations describe the smallest change to an input that would alter the model output.",
    "Feature importance measures how much each input variable contributes to a model's predictions overall.",
    "Hallucination occurs when a language model generates information not supported by the input or evidence.",
    "Calibration measures how well a model's predicted probabilities match observed frequencies of outcomes.",
    "Federated learning enables multiple devices to train a shared model without sharing raw data.",
    "Differential privacy adds carefully calibrated noise to data or gradients to protect individual records.",
    "Secure aggregation allows a server to compute the sum of client updates without seeing individual updates.",
    "Model poisoning attacks attempt to corrupt a shared model by injecting malicious updates.",
    "Homomorphic encryption allows computation on encrypted data without decrypting it first.",
    "Logistic regression models the probability of a binary outcome using a linear combination of features.",
    "Decision trees partition the feature space by choosing splits that maximize information gain.",
    "Random forests average the predictions of many decorrelated decision trees to reduce variance.",
    "Gradient boosting builds an ensemble of trees sequentially, each correcting the errors of the previous.",
    "Support vector machines find the hyperplane that maximizes the margin between two classes.",
    "K-nearest neighbors classifies a new point by majority vote among its closest training examples.",
    "Principal Component Analysis reduces dimensionality while preserving maximum variance in the data.",
    "K-Means clustering partitions data into clusters by iteratively minimizing intra-cluster distances.",
    "DBSCAN identifies clusters based on data density and can detect noise points as outliers.",
    "Naive Bayes applies Bayes theorem with the assumption that features are conditionally independent.",
    "Learning rate determines how large a step the optimizer takes in the direction of the gradient.",
    "Adam optimizer combines momentum and adaptive learning rates for faster and more stable training.",
    "Early stopping halts training when validation loss stops improving to prevent overfitting.",
    "Data augmentation artificially expands training data by applying label-preserving transformations.",
    "Transfer learning reuses weights from a model trained on one task as a starting point for another.",
    "Fine-tuning adapts a pre-trained model to a specific downstream task with a smaller learning rate.",
    "Prompt engineering crafts input text carefully to steer a language model toward a desired output.",
    "Hyperparameter tuning searches for the best configuration of settings that are not learned from data.",
    "Mixed precision training uses 16-bit floats for speed while keeping critical computations in 32-bit.",
    "Gradient clipping caps gradient norms to prevent exploding gradients during deep network training.",
]

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def _get_api_key(key_name: str, default: str = "") -> str:
    """Read API key from Streamlit Cloud Secrets (st.secrets) or OS environment."""
    try:
        if hasattr(st, "secrets") and key_name in st.secrets:
            val = str(st.secrets[key_name]).strip()
            if val:
                return val
    except Exception:
        pass
    return os.environ.get(key_name, default).strip()

def _init_state():
    defaults = {
        "uploaded_docs":     [],
        "custom_chunks":     [],
        "combined_index":    None,
        "combined_docs":     None,
        "processed_names":   set(),
        "last_query":        "",
        "last_results":      None,
        "last_retrieved":    None,
        "last_meta":         None,
        "roadmap_reports":   {},       # map doc_name -> {"markdown", "meta"}
        "isolate_custom_docs": True,   # Isolate uploaded docs from default 106 base chunks to prevent cross-topic pollution
        "gemini_api_key":    _get_api_key("GEMINI_API_KEY"),
        "groq_api_key":      _get_api_key("GROQ_API_KEY"),
        "openai_api_key":    _get_api_key("OPENAI_API_KEY"),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ══════════════════════════════════════════════════════════════════════════════
# VECTOR INDEX MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⚡ Initializing Core FAISS Knowledge Base…")
def _load_default_index():
    return build_index(DEFAULT_DOCS)

_default_index, _ = _load_default_index()

def _rebuild_combined_index():
    """Build FAISS index. When custom documents are present and isolation is enabled, uses only custom docs."""
    isolate = st.session_state.get("isolate_custom_docs", True)
    if isolate and st.session_state.custom_chunks:
        all_chunks = list(st.session_state.custom_chunks)
    elif st.session_state.custom_chunks:
        all_chunks = list(DEFAULT_DOCS) + list(st.session_state.custom_chunks)
    else:
        all_chunks = list(DEFAULT_DOCS)

    idx, _ = build_index(all_chunks)
    st.session_state.combined_index = idx
    st.session_state.combined_docs = all_chunks

def get_active_index():
    """Return the currently active (index, docs) pair."""
    if st.session_state.combined_index is not None and st.session_state.combined_docs is not None:
        return st.session_state.combined_index, st.session_state.combined_docs
    return _default_index, DEFAULT_DOCS

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Prevent Material Icon Text Distortion ──────────────────────────────── */
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons, [class*="material-symbols"] {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 20px !important;
    display: inline-block !important;
    line-height: 1 !important;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    overflow-x: hidden !important;
    background: #f1f5fd !important;
    color: #0f172a !important;
}

p, div, span, label, input, button, select {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    box-sizing: border-box !important;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1400px !important;
    overflow-x: hidden !important;
}

/* ── Main Headings ──────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4 {
    color: #0f296b !important;
    font-weight: 800 !important;
    letter-spacing: -0.3px !important;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081024 0%, #0d1b3e 50%, #11224f 100%) !important;
    border-right: 1px solid #1e3a8a !important;
    overflow-x: hidden !important;
    box-shadow: 4px 0 20px rgba(10, 25, 60, 0.25);
}

section[data-testid="stSidebar"] > div {
    overflow-x: hidden !important;
    padding-left: 0.9rem !important;
    padding-right: 0.9rem !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #e2eeff !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #cbdcfc !important;
}

section[data-testid="stSidebar"] label {
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    color: #93b4ff !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(59, 130, 246, 0.25) !important;
    margin: 0.8rem 0 !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    border: 2px dashed #2563eb !important;
    border-radius: 10px !important;
    background: rgba(17, 34, 79, 0.45) !important;
    padding: 6px !important;
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
}

section[data-testid="stSidebar"] button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    padding: 0.45rem 0.8rem !important;
    transition: all 0.2s ease;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25);
    width: 100% !important;
}

section[data-testid="stSidebar"] button:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    border-color: #60a5fa !important;
    transform: translateY(-1px);
}

.sb-doc-card {
    background: rgba(17, 34, 79, 0.65);
    border: 1px solid #1e3a8a;
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 6px;
    width: 100%;
    overflow: hidden;
    word-break: break-word;
}

.sb-doc-name {
    font-size: 0.82rem;
    font-weight: 700;
    color: #e2eeff !important;
    margin-bottom: 3px;
    word-break: break-word;
    line-height: 1.3;
}

.sb-doc-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 4px;
}

.sb-chip {
    background: rgba(37, 99, 235, 0.25);
    border: 1px solid rgba(59, 130, 246, 0.4);
    border-radius: 10px;
    padding: 2px 6px;
    font-size: 0.68rem;
    font-weight: 600;
    color: #93b4ff !important;
}

/* ── Main Header ─────────────────────────────────────────────────────────── */
.main-header {
    text-align: center;
    padding: 0.8rem 0 0.6rem 0;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 45%, #e11d48 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.8px;
    margin-bottom: 0.2rem;
}

.main-subtitle {
    font-size: 0.95rem;
    color: #2b4374;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

.badge-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 0.2rem;
}

.tech-badge {
    background: #ffffff;
    border: 1px solid #bfdbfe;
    color: #1e3a8a !important;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.74rem;
    font-weight: 700;
    box-shadow: 0 2px 5px rgba(37, 99, 235, 0.06);
}

/* ── Pipeline Stepper ───────────────────────────────────────────────────── */
.pipeline-container {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 10px 14px;
    margin: 0.9rem 0;
    gap: 4px;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.05);
    overflow: hidden;
}

.step-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1 1 auto;
    min-width: 70px;
    padding: 3px;
}

.step-icon {
    font-size: 1.2rem;
    line-height: 1;
    margin-bottom: 2px;
}

.step-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #1e3a8a;
    text-align: center;
    white-space: nowrap;
}

.step-desc {
    font-size: 0.65rem;
    color: #64748b;
    text-align: center;
    margin-top: 1px;
    white-space: nowrap;
}

.step-arrow {
    color: #93c5fd;
    font-size: 1rem;
    font-weight: 900;
    padding: 0 2px;
    flex-shrink: 0;
}

/* ── Input Box & Vivid Cursor ───────────────────────────────────────────── */
textarea {
    border: 2px solid #60a5fa !important;
    border-radius: 10px !important;
    background: #ffffff !important;
    color: #0f172a !important;
    caret-color: #1d4ed8 !important;
    cursor: text !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.06) !important;
    font-size: 0.96rem !important;
    line-height: 1.5 !important;
}

textarea:focus {
    border-color: #1d4ed8 !important;
    caret-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.25) !important;
    outline: none !important;
}

.stButton > button,
.stDownloadButton > button,
.stDownloadButton > a,
div[data-testid="stDownloadButton"] button,
div[data-testid="stDownloadButton"] a,
button[kind="secondary"],
button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0 4px 12px rgba(29, 78, 216, 0.25) !important;
    transition: all 0.2s ease !important;
    text-decoration: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.stButton > button *,
.stDownloadButton > button *,
.stDownloadButton > a *,
div[data-testid="stDownloadButton"] button *,
div[data-testid="stDownloadButton"] a *,
button[kind="secondary"] *,
button[kind="primary"] * {
    color: #ffffff !important;
    fill: #ffffff !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stDownloadButton > a:hover,
div[data-testid="stDownloadButton"] button:hover,
div[data-testid="stDownloadButton"] a:hover,
button[kind="secondary"]:hover,
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%) !important;
    box-shadow: 0 6px 18px rgba(29, 78, 216, 0.35) !important;
    transform: translateY(-1px) !important;
    color: #ffffff !important;
    text-decoration: none !important;
}

.stButton > button:hover *,
.stDownloadButton > button:hover *,
.stDownloadButton > a:hover *,
div[data-testid="stDownloadButton"] button:hover *,
div[data-testid="stDownloadButton"] a:hover *,
button[kind="secondary"]:hover *,
button[kind="primary"]:hover * {
    color: #ffffff !important;
    fill: #ffffff !important;
}

/* ── Streamlit Tabs Styling (Single-Row, Balanced, High Contrast) ───────── */
[data-testid="stTabs"] {
    background: transparent !important;
    margin-top: 1.2rem !important;
    width: 100% !important;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 12px !important;
    padding: 6px !important;
    gap: 6px !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.05) !important;
    display: flex !important;
    flex-wrap: nowrap !important;
    width: 100% !important;
    box-sizing: border-box !important;
    overflow-x: auto !important;
}

[data-testid="stTabs"] [data-baseweb="tab"],
[data-testid="stTabs"] button[role="tab"] {
    flex: 1 1 0px !important;
    background: #f1f5fd !important;
    color: #1e3a8a !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    border: 1px solid #dbeafe !important;
    border-radius: 8px !important;
    padding: 8px 6px !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
    height: auto !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    text-align: center !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    box-sizing: border-box !important;
    min-width: 0 !important;
}

[data-testid="stTabs"] [data-baseweb="tab"]:hover,
[data-testid="stTabs"] button[role="tab"]:hover {
    background: #e0edff !important;
    color: #1d4ed8 !important;
    border-color: #93c5fd !important;
}

[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
[data-testid="stTabs"] [aria-selected="true"] p,
[data-testid="stTabs"] [aria-selected="true"] div,
[data-testid="stTabs"] [aria-selected="true"] span {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    border: 1px solid #1d4ed8 !important;
    font-weight: 800 !important;
    box-shadow: 0 3px 10px rgba(29, 78, 216, 0.25) !important;
}

[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="false"],
[data-testid="stTabs"] button[role="tab"][aria-selected="false"],
[data-testid="stTabs"] [aria-selected="false"] p,
[data-testid="stTabs"] [aria-selected="false"] div,
[data-testid="stTabs"] [aria-selected="false"] span {
    color: #1e3a8a !important;
    font-weight: 700 !important;
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Evaluation Section Header & Backdrop ───────────────────────────────── */
.eval-overview-header {
    background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 60%, #fff1f2 100%);
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 12px 18px;
    margin: 1rem 0 0.8rem 0;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.06);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.eval-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #0f296b !important;
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.eval-subtitle {
    font-size: 0.78rem;
    color: #475569;
    font-weight: 500;
}

/* ── KPI Metric Cards ───────────────────────────────────────────────────── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 10px 14px;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.05);
    display: flex;
    flex-direction: column;
}

.kpi-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

.kpi-value {
    font-size: 1.45rem;
    font-weight: 900;
    color: #0f296b;
    margin-top: 2px;
}

.kpi-sub {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 1px;
}

/* ── Trust Score Cards ──────────────────────────────────────────────────── */
.trust-box-reliable {
    background: #f0f7ff;
    border-left: 5px solid #2563eb;
    border-top: 1px solid #bfdbfe;
    border-right: 1px solid #bfdbfe;
    border-bottom: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.05);
}

.trust-box-unreliable {
    background: #fff5f5;
    border-left: 5px solid #dc2626;
    border-top: 1px solid #fecaca;
    border-right: 1px solid #fecaca;
    border-bottom: 1px solid #fecaca;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    box-shadow: 0 2px 8px rgba(220, 38, 38, 0.05);
}

.badge-tag-reliable {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: #ffffff;
    padding: 3px 10px;
    border-radius: 16px;
    font-size: 0.72rem;
    font-weight: 800;
}

.badge-tag-unreliable {
    background: linear-gradient(135deg, #b91c1c, #dc2626);
    color: #ffffff;
    padding: 3px 10px;
    border-radius: 16px;
    font-size: 0.72rem;
    font-weight: 800;
}

.score-pills-row {
    font-size: 0.78rem;
    color: #334155;
    margin-left: 10px;
    font-weight: 600;
}

.score-pills-row .s-val { color: #1d4ed8; font-weight: 800; }
.score-pills-row .e-val { color: #4338ca; font-weight: 800; }
.score-pills-row .t-val { color: #dc2626; font-weight: 800; }

.sentence-content {
    color: #0f172a;
    font-size: 0.96rem;
    line-height: 1.65;
    margin-top: 8px;
}

/* ── Context & Passage Cards ────────────────────────────────────────────── */
.retrieved-doc-card {
    background: #ffffff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #1e293b;
    line-height: 1.6;
    box-shadow: 0 1px 4px rgba(37, 99, 235, 0.04);
}

.doc-match-chip {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
    font-weight: 700;
    font-size: 0.72rem;
    border-radius: 6px;
    padding: 2px 8px;
    display: inline-block;
    margin-bottom: 6px;
}

/* ── Streamlit Expander Styling (Light, High-Contrast Theme) ────────────── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.04) !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] details {
    border: none !important;
    background: #ffffff !important;
}

[data-testid="stExpander"] summary {
    background: #f8faff !important;
    color: #0f296b !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid #e0edff !important;
    border-radius: 9px 9px 0 0 !important;
    transition: all 0.2s ease !important;
}

[data-testid="stExpander"] summary:hover {
    background: #eff6ff !important;
    color: #1d4ed8 !important;
}

[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary div,
[data-testid="stExpander"] summary label {
    color: #0f296b !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
}

[data-testid="stExpander"] summary svg {
    fill: #1d4ed8 !important;
    color: #1d4ed8 !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #ffffff !important;
    padding: 12px 16px !important;
    border-top: none !important;
    color: #1e293b !important;
}

/* ── Full Answer Container Card ─────────────────────────────────────────── */
.full-answer-container {
    background: #ffffff;
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.05);
}

.full-answer-title {
    font-size: 1rem;
    font-weight: 800;
    color: #0f296b;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.full-answer-body {
    font-size: 0.96rem;
    line-height: 1.7;
    color: #1e293b;
    background: #f8faff;
    border: 1px solid #dbeafe;
    border-radius: 8px;
    padding: 14px 18px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — DOCUMENT INGESTION, CORPUS & PIPELINE CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.4rem 0 0.3rem 0;">
      <div style="font-size:1.45rem; font-weight:900; background: linear-gradient(135deg, #60a5fa, #93c5fd, #f87171); -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-0.5px;">
        🔷 TrustRAG Studio
      </div>
      <div style="font-size:0.73rem; color:#93b4ff; margin-top:2px;">
        Multi-Engine Grounding & Trust Auditor
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── 1. Upload Document Section (TOP PRIORITY) ──────────────────────────────
    st.markdown("### 📤 Ingest Documents")
    st.markdown("""
    <div style="font-size:0.74rem; color:#93b4ff; margin-bottom:4px;">
      Upload PDF, DOCX, TXT, MD, CSV, or JSON:
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["txt", "md", "pdf", "docx", "csv", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_names]
        if new_files:
            if st.button(f"⚡ Ingest {len(new_files)} Document(s)", use_container_width=True):
                added_chunks_count = 0
                for uf in new_files:
                    with st.spinner(f"Extracting & chunking {uf.name}…"):
                        res = process_document(uf)
                    if res.get("error"):
                        st.error(f"❌ {uf.name}: {res['error']}")
                    else:
                        st.session_state.uploaded_docs.append({
                            "name": uf.name,
                            "chunks": res["chunks"],
                            "word_count": res["word_count"],
                            "chunk_count": res["chunk_count"],
                            "preview": res["text"][:300] if res["text"] else "",
                        })
                        st.session_state.custom_chunks.extend(res["chunks"])
                        st.session_state.processed_names.add(uf.name)
                        added_chunks_count += res["chunk_count"]

                if added_chunks_count > 0:
                    with st.spinner("🔢 Rebuilding FAISS Index…"):
                        _rebuild_combined_index()
                    st.success(f"✅ Ingested {len(new_files)} doc(s) ({added_chunks_count} chunks indexed)!")
                    st.rerun()

    st.divider()

    # ── 2. Knowledge Base Index Status & Document Registry ─────────────────────
    st.markdown("### 📚 Knowledge Corpus")
    n_default = len(DEFAULT_DOCS)
    n_custom = len(st.session_state.custom_chunks)

    if n_custom > 0:
        isolate = st.toggle(
            "🔒 Isolate Uploaded Docs",
            value=st.session_state.get("isolate_custom_docs", True),
            help="When enabled, vector search uses ONLY your uploaded documents and excludes the 106 default AI/ML chunks to eliminate cross-topic pollution.",
            key="toggle_isolate_docs"
        )
        if isolate != st.session_state.get("isolate_custom_docs", True):
            st.session_state.isolate_custom_docs = isolate
            _rebuild_combined_index()
            st.rerun()

    active_idx, active_corpus = get_active_index()
    n_total = len(active_corpus) if active_corpus else n_default

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="sb-doc-card" style="text-align:center;">
          <div style="font-size:1.25rem; font-weight:900; color:#60a5fa;">{n_total}</div>
          <div style="font-size:0.7rem; color:#93b4ff;">Active Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="sb-doc-card" style="text-align:center;">
          <div style="font-size:1.25rem; font-weight:900; color:#f87171;">{len(st.session_state.uploaded_docs)}</div>
          <div style="font-size:0.7rem; color:#93b4ff;">Custom Docs</div>
        </div>
        """, unsafe_allow_html=True)

    if n_custom > 0 and st.session_state.get("isolate_custom_docs", True):
        badge_html = f"🔒 Isolated: {n_custom} Custom Chunks (Base AI/ML Chunks Excluded)"
    elif n_custom > 0:
        badge_html = f"🔷 {n_default} Base Chunks &nbsp;·&nbsp; 📄 {n_custom} Custom Chunks"
    else:
        badge_html = f"🔷 {n_default} Base Chunks Active"

    st.markdown(f"""
    <div style="font-size:0.72rem; color:#93b4ff; margin-top:2px; margin-bottom:8px; text-align:center;">
      {badge_html}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🗂️ Document Registry")
    if not st.session_state.uploaded_docs:
        st.markdown("""
        <div class="sb-doc-card" style="text-align:center; font-size:0.75rem; color:#93b4ff;">
          No custom files uploaded yet.<br>Using built-in ML / AI knowledge base.
        </div>
        """, unsafe_allow_html=True)
    else:
        for idx, doc in enumerate(st.session_state.uploaded_docs):
            ext = doc["name"].split(".")[-1].upper() if "." in doc["name"] else "DOC"
            st.markdown(f"""
            <div class="sb-doc-card">
              <div class="sb-doc-name">📄 {doc['name']}</div>
              <div class="sb-doc-chips">
                <span class="sb-chip">✂️ {doc['chunk_count']} chunks</span>
                <span class="sb-chip">📝 {doc['word_count']} words</span>
                <span class="sb-chip">🏷️ {ext}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Reset to Default KB", use_container_width=True):
            st.session_state.uploaded_docs = []
            st.session_state.custom_chunks = []
            st.session_state.processed_names = set()
            st.session_state.combined_index = None
            st.session_state.combined_docs = None
            st.success("Custom documents cleared!")
            st.rerun()

    st.divider()

    # ── 3. Pipeline Tuning Parameters ──────────────────────────────────────────
    st.markdown("### ⚙️ Pipeline Parameters")
    trust_threshold = st.slider(
        "Reliability Threshold (Trust Score)",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05,
        help="Sentences scoring above this threshold are classified as RELIABLE.",
    )
    top_k = st.slider(
        "Semantic Chunks Retrieved (top-k)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of most relevant context chunks retrieved from FAISS.",
    )

    # MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Title & Branding ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div class="main-title">🔷 TrustRAG Engine 🔷</div>
  <div class="main-subtitle">
    Multi-Engine Grounding, Shannon Entropy Confidence & Hallucination Diagnostics for RAG
  </div>
  <div class="badge-row">
    <span class="tech-badge">✨ Google Gemini Cloud</span>
    <span class="tech-badge">⚡ Local Flan-T5 Base</span>
    <span class="tech-badge">🔍 FAISS + all-MiniLM-L6-v2</span>
    <span class="tech-badge">📐 Sentence Trust Scorer</span>
    <span class="tech-badge">📄 Multiformat Doc Ingestion</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Live Pipeline Stepper ─────────────────────────────────────────────────────
_, current_active_docs = get_active_index()
total_chunks_ready = len(current_active_docs) if current_active_docs else len(DEFAULT_DOCS)
current_model_choice = st.session_state.get("selected_model_engine", "✨ Google Gemini")
short_engine_name = "Gemini" if "Gemini" in current_model_choice else ("Flan-T5" if "Local" in current_model_choice else ("Llama-3" if "Groq" in current_model_choice else "GPT-4o"))

st.markdown(f"""
<div class="pipeline-container">
  <div class="step-node">
    <div class="step-icon">📄</div>
    <div class="step-title">1. Ingestion</div>
    <div class="step-desc">{len(st.session_state.uploaded_docs)} Custom Files</div>
  </div>
  <div class="step-arrow">›</div>
  <div class="step-node">
    <div class="step-icon">✂️</div>
    <div class="step-title">2. Chunking</div>
    <div class="step-desc">150w / 30w Overlap</div>
  </div>
  <div class="step-arrow">›</div>
  <div class="step-node">
    <div class="step-icon">🔢</div>
    <div class="step-title">3. Embedding</div>
    <div class="step-desc">384-d MiniLM L2</div>
  </div>
  <div class="step-arrow">›</div>
  <div class="step-node">
    <div class="step-icon">🗄️</div>
    <div class="step-title">4. FAISS Index</div>
    <div class="step-desc">{total_chunks_ready} Total Chunks</div>
  </div>
  <div class="step-arrow">›</div>
  <div class="step-node">
    <div class="step-icon">🔍</div>
    <div class="step-title">5. Retrieval</div>
    <div class="step-desc">Top-{top_k} Nearest</div>
  </div>
  <div class="step-arrow">›</div>
  <div class="step-node">
    <div class="step-icon">🤖</div>
    <div class="step-title">6. {short_engine_name}</div>
    <div class="step-desc">Grounded LLM</div>
  </div>
  <div class="step-arrow">›</div>
  <div class="step-node">
    <div class="step-icon">📊</div>
    <div class="step-title">7. Trust Scorer</div>
    <div class="step-desc">Similarity + Entropy</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Interactive Query Bar ─────────────────────────────────────────────────────
st.markdown("""
<div style="font-weight:700; color:#0f296b; font-size:1rem; margin-top:0.3rem; margin-bottom:3px;">
  🔍 Ask a Question to Evaluate Trust & Grounding
</div>
<div style="font-size:0.78rem; color:#475569; margin-bottom:6px;">
  Try questions:
  <span style="color:#1d4ed8; font-weight:600;">What is this project and its key roles?</span> &nbsp;|&nbsp;
  <span style="color:#1d4ed8; font-weight:600;">What advancements and improvements can we do?</span> &nbsp;|&nbsp;
  <span style="color:#1d4ed8; font-weight:600;">Explain the architecture and datasets</span>
</div>
""", unsafe_allow_html=True)

query_input = st.text_area(
    label="Query",
    placeholder="Enter your query (Press Enter to Run, Shift+Enter for new line)…",
    height=85,
    label_visibility="collapsed",
)

# ── Enter to Submit / Shift+Enter for Newline Client Listener ─────────────────
components.html("""
<script>
(function() {
    function setupEnterKey() {
        const parentDoc = window.parent.document;
        const textareas = parentDoc.querySelectorAll('textarea');
        textareas.forEach(ta => {
            if (!ta.dataset.enterBound) {
                ta.dataset.enterBound = 'true';
                ta.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        const buttons = Array.from(parentDoc.querySelectorAll('button'));
                        const runBtn = buttons.find(b => b.innerText && b.innerText.includes('Run TrustRAG Pipeline'));
                        if (runBtn) {
                            runBtn.click();
                        }
                    }
                });
            }
        });
    }
    setupEnterKey();
    setInterval(setupEnterKey, 700);
})();
</script>
""", height=0, width=0)

btn_col1, btn_col2, btn_col3 = st.columns([2.6, 1.8, 0.8])
with btn_col1:
    execute_clicked = st.button("🚀 Run TrustRAG Pipeline", use_container_width=True)
with btn_col2:
    engine_choice = st.selectbox(
        "Model",
        options=[
            "✨ Google Gemini",
            "⚡ Local Flan-T5 (Offline)",
            "🔥 Groq Llama-3.3",
            "🌐 OpenAI GPT-4o-mini",
        ],
        index=0,
        key="selected_model_engine",
        label_visibility="collapsed",
        help="Select Generation Model",
    )
with btn_col3:
    if st.button("🔄 Clear", use_container_width=True):
        st.session_state.last_results = None
        st.session_state.last_retrieved = None
        st.session_state.last_meta = None
        st.rerun()

# Automatically resolve active API key behind the scenes
if "Gemini" in engine_choice:
    active_api_key = (st.session_state.get("gemini_api_key") or _get_api_key("GEMINI_API_KEY")).strip()
elif "Groq" in engine_choice:
    active_api_key = (st.session_state.get("groq_api_key") or _get_api_key("GROQ_API_KEY")).strip()
elif "OpenAI" in engine_choice:
    active_api_key = (st.session_state.get("openai_api_key") or _get_api_key("OPENAI_API_KEY")).strip()
else:
    active_api_key = ""

short_engine_name = "Gemini" if "Gemini" in engine_choice else ("Flan-T5" if "Local" in engine_choice else ("Llama-3" if "Groq" in engine_choice else "GPT-4o"))

# ── Pipeline Execution ────────────────────────────────────────────────────────
if execute_clicked:
    if not query_input.strip():
        st.warning("⚠️ Please enter a question first.")
    elif ("Gemini" in engine_choice or "Groq" in engine_choice or "OpenAI" in engine_choice) and not active_api_key.strip():
        st.error(f"❌ Please enter your {engine_choice.split()[1]} API Key in the left sidebar to generate detailed answers.")
    else:
        with st.spinner(f"⚡ Running Semantic Search ➔ {short_engine_name} Inference ➔ Grounding & Trust Scoring…"):
            try:
                active_index, active_docs = get_active_index()
                results, retrieved, meta = run_pipeline(
                    query=query_input,
                    documents=active_docs,
                    index=active_index,
                    trust_threshold=trust_threshold,
                    top_k=top_k,
                    llm_provider=engine_choice,
                    api_key=active_api_key,
                )
                st.session_state.last_query = query_input
                st.session_state.last_results = results
                st.session_state.last_retrieved = retrieved
                st.session_state.last_meta = meta
            except Exception as e:
                st.error(f"❌ Pipeline Execution Error: {e}")

# ── Display Pipeline Results & Tabs ───────────────────────────────────────────
if st.session_state.last_results is not None:
    results = st.session_state.last_results
    retrieved = st.session_state.last_retrieved
    meta = st.session_state.last_meta
    t = meta["times"]

    # ── High Contrast Executive Header ────────────────────────────────────────
    st.markdown(f"""
    <div class="eval-overview-header">
      <div>
        <div class="eval-title">📈 Evaluation & Trust Overview</div>
        <div class="eval-subtitle">Engine: <strong>{meta.get('llm_provider', 'LLM')}</strong> &nbsp;·&nbsp; Grounded Confidence & Diagnostics</div>
      </div>
      <div style="font-size:0.75rem; font-weight:700; color:#1d4ed8; background:#eff6ff; border:1px solid #bfdbfe; border-radius:14px; padding:4px 10px;">
        ⏱️ Total Execution: {t['total_s']}s
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    with col_kpi1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Avg Trust Score</div>
          <div class="kpi-value" style="color:#1d4ed8;">{meta['avg_trust_score']:.3f}</div>
          <div class="kpi-sub">Threshold: {meta['trust_threshold']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi2:
        rel_color = "#16a34a" if meta['reliability_rate'] >= 70 else ("#d97706" if meta['reliability_rate'] >= 40 else "#dc2626")
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Reliability Rate</div>
          <div class="kpi-value" style="color:{rel_color};">{meta['reliability_rate']:.1f}%</div>
          <div class="kpi-sub">{meta['reliable_count']}/{meta['sentence_count']} Sentences</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi3:
        risk_color = "#dc2626" if meta['hallucination_risk'] > 40 else "#16a34a"
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Hallucination Risk</div>
          <div class="kpi-value" style="color:{risk_color};">{meta['hallucination_risk']:.1f}%</div>
          <div class="kpi-sub">{meta['unreliable_count']} Unreliable Sentences</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi4:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Avg Context Sim</div>
          <div class="kpi-value" style="color:#4338ca;">{meta['avg_similarity']:.3f}</div>
          <div class="kpi-sub">{meta['chunks_retrieved']} Chunks Retrieved</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi5:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Pipeline Latency</div>
          <div class="kpi-value" style="color:#0f172a;">{t['total_s']}s</div>
          <div class="kpi-sub">R: {t['retrieve_s']}s · G: {t['generate_s']}s · S: {t['score_s']}s</div>
        </div>
        """, unsafe_allow_html=True)

# ── Tabs: Structured, Balanced Layout ──────────────────────────────────────
tab_eval, tab_pipeline, tab_docs, tab_roadmap = st.tabs([
    "📊 Answer & Trust Audit",
    "🔬 Pipeline RAW Inspector",
    "📚 Knowledge Chunks",
    "📑 Document Roadmap",
])

with tab_eval:
    if st.session_state.last_results is not None:
        results = st.session_state.last_results
        retrieved = st.session_state.last_retrieved
        meta = st.session_state.last_meta
        # 1. Full Answer (Rendered Markdown & Rich formatting)
        st.markdown("""
        <div class="full-answer-container">
          <div class="full-answer-title">
            <span>🤖 AI Response Output</span>
            <span style="font-size:0.75rem; color:#64748b; font-weight:600;">""" + str(meta.get('answer_word_count', 0)) + """ words</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Render markdown answer cleanly
        st.markdown(meta.get("answer_text", ""))

        st.divider()

        # 2. Sentence-by-Sentence Diagnostics
        st.markdown("#### 🔍 Sentence-Level Trust & Grounding Breakdown")
        if not results:
            st.info("No sentence output generated.")
        for idx, r in enumerate(results, 1):
            is_rel = r.get("is_reliable", r.get("label", "").startswith("✅"))
            card_class = "trust-box-reliable" if is_rel else "trust-box-unreliable"
            tag = (
                '<span class="badge-tag-reliable">✅ RELIABLE</span>'
                if is_rel else
                '<span class="badge-tag-unreliable">❌ UNRELIABLE</span>'
            )
            st.markdown(f"""
            <div class="{card_class}">
              <div style="display:flex; align-items:center; flex-wrap:wrap; gap:6px;">
                <span style="font-weight:800; font-size:0.8rem; color:#0f296b;">Sentence #{idx}</span>
                {tag}
                <span class="score-pills-row">
                  Context Sim: <span class="s-val">{r['similarity']:.3f}</span> &nbsp;|&nbsp;
                  Token Uncertainty: <span class="e-val">{r['entropy']:.3f}</span> &nbsp;|&nbsp;
                  Trust Score: <span class="t-val">{r['trust_score']:.3f}</span>
                </span>
              </div>
              <div class="sentence-content">{r['sentence']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # 3. Retrieved Context Evidence
        st.markdown("#### 📚 Retrieved Supporting Context Evidence")
        st.caption(f"Retrieved {len(retrieved)} most relevant semantic chunks from your indexed corpus for this query:")

        for i, (doc, score) in enumerate(zip(retrieved, meta["retrieval_scores"]), 1):
            with st.expander(f"📄 Match #{i} &nbsp;·&nbsp; Cosine Similarity: {score:.3f} &nbsp;·&nbsp; ({len(doc.split())} words)", expanded=(i <= 2)):
                st.markdown(f"""
                <div class="retrieved-doc-card">
                  <span class="doc-match-chip">Evidence Match #{i} &nbsp;|&nbsp; Relevance: {score:.3f}</span>
                  <div style="font-size:0.92rem; line-height:1.65; color:#1e293b;">{doc}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("💡 Enter your question in the query box above and click **Run TrustRAG Pipeline** to view the generated grounded answer and sentence-level trust audit.")

with tab_pipeline:
    if st.session_state.last_results is not None:
        meta = st.session_state.last_meta
        results = st.session_state.last_results
        st.markdown("#### 🔬 Detailed Pipeline Stages Breakdown")
        
        st.markdown(f"##### 1. Full Prompt Fed to {meta.get('llm_provider', 'LLM')}")
        st.code(meta.get("prompt", ""), language="markdown")

        st.markdown("##### 2. Raw Text Output")
        st.code(meta.get("answer_text", ""), language="text")

        st.markdown("##### 3. Sentence Evaluation Matrix")
        df_eval = pd.DataFrame(results)
        st.dataframe(df_eval, use_container_width=True)
    else:
        st.info("🔬 Run a query above to inspect raw prompts, model tokens, and evaluation matrix.")

with tab_docs:
    st.markdown("#### 📚 Active Knowledge Corpus Chunks")
    _, all_active = get_active_index()
    st.write(f"Total Chunks Currently in Vector Store: **{len(all_active)}**")
    
    search_filter = st.text_input("Filter indexed chunks by keyword:", "")
    filtered = [c for c in all_active if search_filter.lower() in c.lower()] if search_filter else all_active[:30]
    
    for idx, chunk in enumerate(filtered, 1):
        st.markdown(f"""
        <div class="retrieved-doc-card">
          <span class="doc-match-chip">Chunk #{idx}</span>
          <div>{chunk}</div>
        </div>
        """, unsafe_allow_html=True)
    if len(all_active) > 30 and not search_filter:
        st.caption(f"Showing first 30 of {len(all_active)} chunks. Use filter above to search specific terms.")

with tab_roadmap:
    st.markdown("### 📑 Document Insights & Auto-Advancements Roadmap")
    st.markdown("""
    <p style="color:#475569; font-size:0.88rem; margin-bottom:12px;">
      Automatically analyze your uploaded specification (e.g., <strong>Arcana PRD</strong>) to extract core objectives, 
      identify technical bottlenecks, and generate <strong>actionable Next-Gen Engineering Advancements</strong> (e.g. Vision Transformers, Edge Deployment, Diffusion Defect Augmentation).
    </p>
    """, unsafe_allow_html=True)

    if not st.session_state.uploaded_docs:
        st.info("📎 Please upload a document (PDF, DOCX, TXT, MD) in the left sidebar first to generate an automated PRD analysis & roadmap.")
    else:
        doc_options = [d["name"] for d in st.session_state.uploaded_docs]
        selected_doc_name = st.selectbox("Select Target Document to Analyze:", options=doc_options)
        
        selected_doc = next((d for d in st.session_state.uploaded_docs if d["name"] == selected_doc_name), None)

        if selected_doc:
            col_btn, col_info = st.columns([2, 3])
            with col_btn:
                generate_roadmap_btn = st.button("✨ Generate PRD Analysis & Future Roadmap", use_container_width=True)
            with col_info:
                st.caption(f"Target: **{selected_doc_name}** ({selected_doc['chunk_count']} chunks · {selected_doc['word_count']:,} words)")

            if generate_roadmap_btn:
                with st.spinner(f"🚀 Analyzing {selected_doc_name} with {short_engine_name} & synthesizing technical roadmap…"):
                    try:
                        roadmap_md, r_meta = generate_document_roadmap(
                            doc_name=selected_doc_name,
                            chunks=selected_doc["chunks"],
                            full_text=selected_doc.get("preview", ""),
                            llm_provider=engine_choice,
                            api_key=active_api_key,
                        )
                        st.session_state.roadmap_reports[selected_doc_name] = {
                            "markdown": roadmap_md,
                            "meta": r_meta,
                        }
                        st.success("✅ Technical Analysis & Roadmap Generated Successfully!")
                    except Exception as e:
                        st.error(f"❌ Roadmap Generation Error: {e}")

            if selected_doc_name in st.session_state.roadmap_reports:
                rep = st.session_state.roadmap_reports[selected_doc_name]
                rmeta = rep["meta"]
                
                st.divider()
                st.markdown(f"""
                <div style="display:flex; align-items:center; justify-content:space-between; background:#ffffff; border:1px solid #bfdbfe; border-radius:10px; padding:10px 16px; margin-bottom:14px;">
                  <div>
                    <strong style="color:#0f296b;">📄 {selected_doc_name} — Comprehensive Technical Roadmap</strong><br>
                    <span style="font-size:0.75rem; color:#64748b;">Generated via {rmeta.get('llm_provider', 'LLM')} in {rmeta.get('generation_time_s', '1.2')}s &nbsp;·&nbsp; {rmeta.get('word_count', 0)} words</span>
                  </div>
                  <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px; padding:4px 10px; font-size:0.75rem; font-weight:700; color:#1d4ed8;">
                    🛡️ Grounding Confidence: {rmeta.get('avg_grounding', 0.85):.1%}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(rep["markdown"])

                st.download_button(
                    label="📥 Download Technical Roadmap (.md)",
                    data=rep["markdown"],
                    file_name=f"{selected_doc_name}_Roadmap.md",
                    mime="text/markdown",
                )