"""
config.py — Centralized configuration for RAG PDF Assistant
All settings are loaded from .env with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ── Base Paths ────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", BASE_DIR / "vectorstore"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", DATA_DIR / "cache"))

# Create directories if they don't exist
for d in [DATA_DIR, VECTORSTORE_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Ollama / LLM ─────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

# ── Embeddings ───────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))

# ── Retrieval ────────────────────────────────────────────────
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 5))

# ── UI ───────────────────────────────────────────────────────
MAX_PDF_SIZE_MB: int = int(os.getenv("MAX_PDF_SIZE_MB", 50))

# ── Prompt Template ──────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are a precise and helpful document assistant.
Use ONLY the context provided below to answer the question.
If the answer is not in the context, say "I couldn't find that information in the uploaded documents."
Never make up information. Cite the source page numbers when relevant.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (be concise and accurate):"""
