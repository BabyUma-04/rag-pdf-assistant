"""
vector_store.py — Embedding generation and FAISS vector store management

Handles:
  - HuggingFace sentence-transformer embeddings
  - FAISS index creation, saving, and loading
  - Similarity search with metadata-aware retrieval
  - Disk-based caching to avoid re-embedding
"""

import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL, TOP_K_RESULTS, VECTORSTORE_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Embedding Model (singleton, loaded once)
# ─────────────────────────────────────────────────────────────

_embeddings_instance: Optional[HuggingFaceEmbeddings] = None


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Returns a cached HuggingFaceEmbeddings instance.
    Downloads the model on first use (~90MB for MiniLM).
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        logger.info(f"Loading embedding model: {model_name}")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # Use "cuda" if GPU available
            encode_kwargs={
                "normalize_embeddings": True,  # Cosine similarity
                "batch_size": 32,
            },
        )
        logger.info("Embedding model loaded.")
    return _embeddings_instance


# ─────────────────────────────────────────────────────────────
# FAISS Vector Store
# ─────────────────────────────────────────────────────────────

def get_index_path(session_id: str) -> Path:
    """Returns the FAISS index directory for a given session."""
    return VECTORSTORE_DIR / session_id


def build_vectorstore(
    chunks: List[Document],
    session_id: str,
) -> FAISS:
    """
    Build a FAISS index from document chunks and persist it to disk.

    Args:
        chunks:     Chunked LangChain Documents with metadata
        session_id: Unique identifier for this index (e.g., PDF hash)

    Returns:
        FAISS vectorstore ready for retrieval
    """
    if not chunks:
        raise ValueError("Cannot build vector store from empty chunk list.")

    embeddings = get_embeddings()
    index_path = get_index_path(session_id)

    logger.info(f"Building FAISS index for {len(chunks)} chunks...")

    # Build the index — this embeds all chunks
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # Persist to disk
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    logger.info(f"FAISS index saved to {index_path}")

    # Save metadata manifest alongside the index
    manifest = {
        "session_id": session_id,
        "chunk_count": len(chunks),
        "sources": list({c.metadata.get("source", "unknown") for c in chunks}),
        "embedding_model": EMBEDDING_MODEL,
    }
    with open(index_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return vectorstore


def load_vectorstore(session_id: str) -> Optional[FAISS]:
    """
    Load a previously saved FAISS index from disk.
    Returns None if no index exists for the session.
    """
    index_path = get_index_path(session_id)

    if not (index_path / "index.faiss").exists():
        logger.info(f"No existing FAISS index found for session: {session_id}")
        return None

    embeddings = get_embeddings()
    logger.info(f"Loading FAISS index from {index_path}")
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,  # Safe: we wrote this file ourselves
    )
    return vectorstore


def get_or_build_vectorstore(
    chunks: List[Document],
    session_id: str,
    force_rebuild: bool = False,
) -> Tuple[FAISS, bool]:
    """
    Load cached index or build a new one if needed.

    Returns:
        (vectorstore, was_cached): vectorstore and whether it was loaded from cache
    """
    if not force_rebuild:
        cached = load_vectorstore(session_id)
        if cached is not None:
            logger.info("Loaded vector store from cache.")
            return cached, True

    store = build_vectorstore(chunks, session_id)
    return store, False


def retrieve_relevant_chunks(
    vectorstore: FAISS,
    query: str,
    top_k: int = TOP_K_RESULTS,
) -> List[Document]:
    """
    Perform similarity search and return top-k relevant chunks.

    Uses Maximum Marginal Relevance (MMR) for diversity — avoids
    returning near-duplicate chunks from the same section.
    """
    # MMR balances relevance AND diversity
    results = vectorstore.max_marginal_relevance_search(
        query=query,
        k=top_k,
        fetch_k=top_k * 3,  # Fetch more candidates, then pick diverse top_k
        lambda_mult=0.7,     # 0 = max diversity, 1 = max relevance
    )
    return results


def format_retrieved_context(docs: List[Document]) -> Tuple[str, List[dict]]:
    """
    Format retrieved chunks into a single context string for the LLM prompt.
    Also returns structured citation info.

    Returns:
        context_str: Formatted string of all retrieved chunks
        citations:   List of {source, page, snippet} dicts
    """
    context_parts = []
    citations = []

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        text = doc.page_content.strip()

        context_parts.append(
            f"[Source {i+1}: {source}, Page {page}]\n{text}"
        )

        citations.append({
            "index": i + 1,
            "source": source,
            "page": page,
            "snippet": text[:200] + "..." if len(text) > 200 else text,
        })

    context_str = "\n\n---\n\n".join(context_parts)
    return context_str, citations
