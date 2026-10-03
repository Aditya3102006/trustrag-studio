"""
chunker.py — Document ingestion & chunking for TrustRAG.

Supports: .txt, .md, .pdf (requires PyPDF2), .docx (requires python-docx)
Strategy: word-level sliding window with configurable size & overlap.
"""
import re


# ── Text Cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list:
    """
    Split text into overlapping word-level chunks.

    Args:
        text:        Raw input text.
        chunk_size:  Maximum number of words per chunk.
        overlap:     Number of words shared between consecutive chunks.

    Returns:
        List of non-empty text chunks.
    """
    text = clean_text(text)
    words = text.split()
    if not words:
        return []

    step = max(chunk_size - overlap, 1)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.strip()) > 15:
            chunks.append(chunk.strip())
    return chunks


# ── File Text Extraction ───────────────────────────────────────────────────────

def _extract_txt(content: bytes):
    return content.decode("utf-8", errors="ignore"), None


def _extract_pdf(content: bytes):
    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages), None
    except ImportError:
        try:
            import PyPDF2
            import io
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages), None
        except ImportError:
            return None, "PDF support requires pypdf or PyPDF2 — run: pip install pypdf"
    except Exception as exc:
        return None, f"Error reading PDF: {exc}"


def _extract_docx(content: bytes):
    try:
        import docx
        import io
        doc = docx.Document(io.BytesIO(content))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs), None
    except ImportError:
        return None, "DOCX support requires python-docx — run: pip install python-docx"
    except Exception as exc:
        return None, f"Error reading DOCX: {exc}"


def extract_text(uploaded_file):
    """
    Extract raw text from a Streamlit UploadedFile object.
    Returns: (text, error_message) — one of them will be None.
    """
    name = uploaded_file.name.lower()
    content = uploaded_file.read()

    if name.endswith((".txt", ".md", ".csv")):
        return _extract_txt(content)
    if name.endswith(".pdf"):
        return _extract_pdf(content)
    if name.endswith(".docx"):
        return _extract_docx(content)

    try:
        return content.decode("utf-8", errors="ignore"), None
    except Exception:
        return None, "Unsupported file format"


# ── Main Entry Point ───────────────────────────────────────────────────────────

def process_document(uploaded_file, chunk_size: int = 150, overlap: int = 30) -> dict:
    """
    Full ingestion pipeline for a single uploaded file.

    Returns a dict with keys:
        chunks      : list[str]  — text chunks ready for embedding
        text        : str        — full extracted text
        word_count  : int
        chunk_count : int
        error       : str | None
    """
    text, error = extract_text(uploaded_file)

    if error or not text or not text.strip():
        return {
            "chunks":      [],
            "text":        text or "",
            "word_count":  0,
            "chunk_count": 0,
            "error":       error or "Document appears to be empty.",
        }

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    return {
        "chunks":      chunks,
        "text":        text,
        "word_count":  len(text.split()),
        "chunk_count": len(chunks),
        "error":       None,
    }
