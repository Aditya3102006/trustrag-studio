# 🔷 TrustRAG Studio

> **Multi-Engine Grounding & Trust Auditor** — A Retrieval-Augmented Generation (RAG) pipeline with built-in hallucination detection, trust scoring, and multi-model support.

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 What is TrustRAG?

TrustRAG Studio is an AI-powered document Q&A system that:
- 📄 **Ingests** documents (PDF, DOCX, TXT, MD, CSV, JSON)
- 🔍 **Retrieves** the most relevant context using semantic search (FAISS)
- 🧠 **Generates** answers using multiple LLM engines (Gemini, Local Flan-T5, etc.)
- ✅ **Scores** each answer for trustworthiness and hallucination risk
- 📊 **Audits** answers with evidence grounding, semantic similarity, and token overlap

---

## 🚀 Features

| Feature | Description |
|--------|-------------|
| 📤 Document Ingestion | Upload PDF, DOCX, TXT, MD, CSV, JSON files |
| 🔍 Semantic Retrieval | FAISS-powered vector search with chunk scoring |
| 🧠 Multi-Model Support | Gemini API, Local Flan-T5 (offline), extensible |
| 🛡️ Trust Scoring | Hallucination detection with confidence metrics |
| 📊 Evaluation Dashboard | Side-by-side model comparison & trust audit |
| 💬 Chat Interface | Conversational Q&A with history |
| 🎨 Modern UI | Dark-mode Streamlit interface with animations |

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2)
- **Vector Store**: FAISS (CPU)
- **LLM Engines**: Google Gemini API, HuggingFace Transformers (Flan-T5)
- **Document Parsing**: PyPDF, python-docx, pandas
- **Scoring**: Custom trust + semantic similarity metrics

---

## 📁 Project Structure

```
TrustRAG/
├── app.py              # Main Streamlit application
├── pipeline.py         # RAG pipeline orchestrator
├── retriever.py        # FAISS semantic retrieval
├── generator.py        # LLM generation engines
├── chunker.py          # Document chunking logic
├── scorer.py           # Trust & hallucination scoring
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/trustrag-studio.git
cd trustrag-studio
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Set Gemini API Key
Create a `.streamlit/secrets.toml` file:
```toml
GEMINI_API_KEY = "your_api_key_here"
```
> ⚠️ Never commit this file — it's already in `.gitignore`

### 5. Run the App
```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🎯 How to Use

1. **Upload Documents** — Use the sidebar to upload your files
2. **Ask Questions** — Type in the chat box and press **Enter**
3. **Select Model** — Choose between Gemini or Local Flan-T5 from the Model dropdown
4. **Review Results** — See trust scores, retrieved evidence, and generated answers
5. **Compare Models** — Use the Evaluation tab to compare answers side-by-side

---

## 🌐 Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → `app.py` → Deploy
5. Add `GEMINI_API_KEY` in the Secrets section

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io) — UI framework
- [HuggingFace](https://huggingface.co) — Transformers & models
- [Google Gemini](https://ai.google.dev) — LLM API
- [FAISS](https://faiss.ai) — Vector similarity search

---

<div align="center">
  Made with ❤️ using Python & Streamlit
</div>
