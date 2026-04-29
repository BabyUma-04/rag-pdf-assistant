"""
pdf_processor.py — PDF ingestion, text extraction, and chunking

Handles:
  - Loading single or multiple PDFs
  - Extracting text with page metadata
  - Smart chunking with RecursiveCharacterTextSplitter
  - Deduplication of chunks
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Tuple

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Text Extraction
# ─────────────────────────────────────────────────────────────

def extract_text_pypdf(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extract text page-by-page using pypdf.
    Returns list of (page_number, text) tuples.
    """
    pages = []
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i + 1, text.strip()))
    except Exception as e:
        logger.warning(f"pypdf extraction failed: {e}. Falling back to pdfplumber.")
        pages = []
    return pages


def extract_text_pdfplumber(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extract text page-by-page using pdfplumber (better for tables/complex layouts).
    Returns list of (page_number, text) tuples.
    """
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((i + 1, text.strip()))
    except Exception as e:
        logger.error(f"pdfplumber extraction failed: {e}")
    return pages


def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Try pypdf first; fall back to pdfplumber if insufficient text is extracted.
    """
    pages = extract_text_pypdf(pdf_path)

    # Fallback if pypdf extracted very little text (likely a scanned PDF)
    total_chars = sum(len(t) for _, t in pages)
    if total_chars < 100:
        logger.info("Switching to pdfplumber for better extraction.")
        pages = extract_text_pdfplumber(pdf_path)

    return pages


# ─────────────────────────────────────────────────────────────
# Document Building with Metadata
# ─────────────────────────────────────────────────────────────

def build_documents(
    pages: List[Tuple[int, str]],
    source_name: str,
    pdf_hash: str,
) -> List[Document]:
    """
    Convert (page_num, text) pairs into LangChain Documents with rich metadata.
    """
    docs = []
    for page_num, text in pages:
        doc = Document(
            page_content=text,
            metadata={
                "source": source_name,
                "page": page_num,
                "pdf_hash": pdf_hash,
                "char_count": len(text),
            },
        )
        docs.append(doc)
    return docs


# ─────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────

def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.
    Preserves source metadata and adds chunk index.

    RecursiveCharacterTextSplitter tries to split on:
      ["\n\n", "\n", " ", ""] — in order of preference
    This keeps paragraphs and sentences intact when possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,  # Adds start_index to metadata
    )

    chunks = splitter.split_documents(documents)

    # Enrich metadata: add chunk index within each source+page
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        # Create a unique ID for deduplication
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
        chunk.metadata["content_hash"] = content_hash

    # Deduplicate by content hash
    seen_hashes = set()
    unique_chunks = []
    for chunk in chunks:
        h = chunk.metadata["content_hash"]
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_chunks.append(chunk)

    logger.info(
        f"Chunking: {len(documents)} pages → {len(chunks)} chunks "
        f"→ {len(unique_chunks)} unique chunks"
    )
    return unique_chunks


# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────

def process_pdf(pdf_path: str) -> Tuple[List[Document], dict]:
    """
    Full pipeline: PDF → extracted pages → LangChain Documents → chunks.

    Returns:
        chunks: List of chunked Document objects ready for embedding
        stats:  Dictionary with processing statistics
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Compute file hash for caching/deduplication
    with open(pdf_path, "rb") as f:
        pdf_hash = hashlib.md5(f.read()).hexdigest()

    source_name = path.name

    logger.info(f"Processing PDF: {source_name} (hash: {pdf_hash[:8]}...)")

    # 1. Extract text
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        raise ValueError(
            f"No text could be extracted from {source_name}. "
            "The PDF may be scanned/image-based and requires OCR."
        )

    # 2. Build Documents
    documents = build_documents(pages, source_name, pdf_hash)

    # 3. Chunk
    chunks = chunk_documents(documents)

    stats = {
        "source": source_name,
        "pdf_hash": pdf_hash,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "total_chars": sum(len(c.page_content) for c in chunks),
        "avg_chunk_size": (
            sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0
        ),
    }

    logger.info(f"Processed: {stats}")
    return chunks, stats


def process_multiple_pdfs(pdf_paths: List[str]) -> Tuple[List[Document], List[dict]]:
    """
    Process multiple PDFs and combine all chunks.
    """
    all_chunks = []
    all_stats = []

    for path in pdf_paths:
        try:
            chunks, stats = process_pdf(path)
            all_chunks.extend(chunks)
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            all_stats.append({"source": path, "error": str(e)})

    return all_chunks, all_stats
