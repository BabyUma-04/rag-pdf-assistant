"""
rag_pipeline.py — End-to-end RAG pipeline orchestrator

Ties together: PDF processing → embedding → retrieval → LLM generation
This is the single entry point for the Streamlit UI to use.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import OLLAMA_MODEL, TOP_K_RESULTS
from llm_handler import (
    check_ollama_running,
    generate_answer,
    generate_answer_streaming,
    is_model_available,
    list_available_models,
)
from pdf_processor import process_multiple_pdfs, process_pdf
from vector_store import (
    format_retrieved_context,
    get_or_build_vectorstore,
    retrieve_relevant_chunks,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    role: str           # "user" or "assistant"
    content: str
    citations: List[dict] = field(default_factory=list)


@dataclass
class RAGResult:
    answer: str
    citations: List[dict]
    retrieved_chunks: List[Document]
    question: str
    model_used: str


# ─────────────────────────────────────────────────────────────
# RAG Pipeline Class
# ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Manages the full RAG lifecycle:
      - PDF ingestion and indexing
      - Query → retrieve → generate flow
      - Chat history tracking
    """

    def __init__(self):
        self.vectorstore: Optional[FAISS] = None
        self.session_id: Optional[str] = None
        self.pdf_stats: List[dict] = []
        self.chat_history: List[ChatMessage] = []
        self.current_model: str = OLLAMA_MODEL
        self._indexed_sources: List[str] = []

    # ── Indexing ─────────────────────────────────────────────

    def index_pdfs(
        self,
        pdf_paths: List[str],
        progress_callback=None,
    ) -> Tuple[bool, str]:
        """
        Process PDFs and build/update the FAISS index.

        Args:
            pdf_paths:          List of PDF file paths
            progress_callback:  Optional callable(step: str, pct: float)

        Returns:
            (success, message)
        """
        try:
            if progress_callback:
                progress_callback("Extracting text from PDFs...", 0.1)

            # Process all PDFs
            chunks, stats = process_multiple_pdfs(pdf_paths)
            self.pdf_stats = stats

            if not chunks:
                return False, "No text could be extracted from the provided PDFs."

            if progress_callback:
                progress_callback("Building embeddings and FAISS index...", 0.4)

            # Create a session ID from all PDF hashes combined
            combined_hash = hashlib.md5(
                "".join(s.get("pdf_hash", "") for s in stats if "pdf_hash" in s).encode()
            ).hexdigest()[:12]

            self.session_id = combined_hash

            # Build or load vector store
            self.vectorstore, was_cached = get_or_build_vectorstore(
                chunks, self.session_id
            )

            self._indexed_sources = [
                s["source"] for s in stats if "source" in s and "error" not in s
            ]

            if progress_callback:
                progress_callback("Ready!", 1.0)

            total_chunks = sum(s.get("total_chunks", 0) for s in stats if "total_chunks" in s)
            cache_note = " (loaded from cache)" if was_cached else " (freshly indexed)"
            msg = (
                f"✅ Indexed {len(pdf_paths)} PDF(s) into {total_chunks} chunks{cache_note}.\n"
                f"Sources: {', '.join(self._indexed_sources)}"
            )
            return True, msg

        except Exception as e:
            logger.error(f"Indexing failed: {e}", exc_info=True)
            return False, f"❌ Indexing failed: {e}"

    # ── Querying ─────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = TOP_K_RESULTS,
        model_name: Optional[str] = None,
        stream: bool = False,
    ):
        """
        Full RAG query: retrieve relevant chunks → generate answer.

        Args:
            question:   User's question string
            top_k:      Number of chunks to retrieve
            model_name: Override the default Ollama model
            stream:     If True, returns a generator; else returns RAGResult

        Returns:
            RAGResult (stream=False) or Generator[str] (stream=True)
        """
        if self.vectorstore is None:
            raise RuntimeError("No documents indexed. Please upload PDFs first.")

        model = model_name or self.current_model

        # 1. Retrieve relevant chunks
        docs = retrieve_relevant_chunks(self.vectorstore, question, top_k=top_k)

        # 2. Format context + citations
        context, citations = format_retrieved_context(docs)

        # 3. Generate answer
        if stream:
            return self._stream_answer(question, context, citations, docs, model)
        else:
            answer = generate_answer(context, question, model_name=model)
            result = RAGResult(
                answer=answer,
                citations=citations,
                retrieved_chunks=docs,
                question=question,
                model_used=model,
            )
            # Track in chat history
            self.chat_history.append(ChatMessage("user", question))
            self.chat_history.append(
                ChatMessage("assistant", answer, citations=citations)
            )
            return result

    def _stream_answer(
        self,
        question: str,
        context: str,
        citations: List[dict],
        docs: List[Document],
        model: str,
    ) -> Generator[str, None, None]:
        """Internal: stream answer and update chat history."""
        full_answer = []
        for chunk in generate_answer_streaming(context, question, model_name=model):
            full_answer.append(chunk)
            yield chunk

        # Store in history after streaming completes
        complete = "".join(full_answer)
        self.chat_history.append(ChatMessage("user", question))
        self.chat_history.append(
            ChatMessage("assistant", complete, citations=citations)
        )

    # ── Utilities ────────────────────────────────────────────

    def clear_history(self):
        """Clear chat history (keeps index intact)."""
        self.chat_history = []

    def reset(self):
        """Full reset — clears index and history."""
        self.vectorstore = None
        self.session_id = None
        self.pdf_stats = []
        self.chat_history = []
        self._indexed_sources = []

    @property
    def is_ready(self) -> bool:
        """True if documents are indexed and ready to query."""
        return self.vectorstore is not None

    @property
    def indexed_sources(self) -> List[str]:
        return self._indexed_sources

    # ── System Status ─────────────────────────────────────────

    @staticmethod
    def get_system_status() -> Dict:
        """Returns a dict with Ollama and model status info."""
        ollama_ok = check_ollama_running()
        available_models = list_available_models() if ollama_ok else []
        return {
            "ollama_running": ollama_ok,
            "available_models": available_models,
            "default_model": OLLAMA_MODEL,
            "model_ready": is_model_available(OLLAMA_MODEL) if ollama_ok else False,
        }
