"""
llm_handler.py — Ollama LLM integration for answer generation

Handles:
  - Connecting to local Ollama server
  - Checking model availability
  - Generating answers with RAG prompt template
  - Streaming support for real-time output
  - Disk-based response caching
"""

import hashlib
import json
import logging
from typing import Generator, List, Optional

import diskcache
import requests
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from config import CACHE_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL, RAG_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Response cache (persists across sessions)
_response_cache = diskcache.Cache(str(CACHE_DIR / "llm_responses"))


# ─────────────────────────────────────────────────────────────
# Ollama Health & Model Management
# ─────────────────────────────────────────────────────────────

def check_ollama_running() -> bool:
    """Returns True if the Ollama server is reachable."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def list_available_models() -> List[str]:
    """Returns list of model names available in the local Ollama instance."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        logger.error(f"Could not fetch Ollama models: {e}")
        return []


def is_model_available(model_name: str) -> bool:
    """Check if a specific model is pulled and ready."""
    models = list_available_models()
    # Match exact name or name without tag
    base_name = model_name.split(":")[0]
    return any(
        m == model_name or m.split(":")[0] == base_name
        for m in models
    )


def get_llm(model_name: str = OLLAMA_MODEL) -> OllamaLLM:
    """
    Initialize and return an Ollama LLM instance.
    Raises ConnectionError if Ollama is not running.
    """
    if not check_ollama_running():
        raise ConnectionError(
            f"Ollama is not running at {OLLAMA_BASE_URL}.\n"
            "Please start it with: `ollama serve`"
        )

    return OllamaLLM(
        base_url=OLLAMA_BASE_URL,
        model=model_name,
        temperature=0.1,       # Low temp for factual answers
        num_predict=1024,      # Max tokens in response
        top_p=0.9,
    )


# ─────────────────────────────────────────────────────────────
# RAG Chain
# ─────────────────────────────────────────────────────────────

def build_rag_prompt(context: str, question: str) -> str:
    """Fills the RAG prompt template with context and question."""
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    return prompt.format(context=context, question=question)


def generate_answer(
    context: str,
    question: str,
    model_name: str = OLLAMA_MODEL,
    use_cache: bool = True,
) -> str:
    """
    Generate an answer using Ollama LLM with the RAG prompt.

    Args:
        context:    Retrieved document context (formatted string)
        question:   User's question
        model_name: Ollama model to use
        use_cache:  Whether to use disk cache for identical queries

    Returns:
        Generated answer string
    """
    # Build cache key from context + question + model
    cache_key = hashlib.md5(
        f"{model_name}::{question}::{context[:500]}".encode()
    ).hexdigest()

    if use_cache and cache_key in _response_cache:
        logger.info("Cache hit — returning cached answer.")
        return _response_cache[cache_key]

    prompt = build_rag_prompt(context, question)

    try:
        llm = get_llm(model_name)
        answer = llm.invoke(prompt)

        # Cache the response
        if use_cache:
            _response_cache[cache_key] = answer

        return answer

    except ConnectionError as e:
        raise e
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise RuntimeError(f"Answer generation failed: {e}") from e


def generate_answer_streaming(
    context: str,
    question: str,
    model_name: str = OLLAMA_MODEL,
) -> Generator[str, None, None]:
    """
    Stream the LLM answer token by token.
    Yields string chunks as they arrive from Ollama.
    """
    prompt = build_rag_prompt(context, question)

    try:
        llm = get_llm(model_name)
        for chunk in llm.stream(prompt):
            yield chunk
    except ConnectionError as e:
        yield f"❌ Error: {e}"
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        yield f"❌ Error during generation: {e}"


def clear_cache():
    """Clear the LLM response cache."""
    _response_cache.clear()
    logger.info("LLM response cache cleared.")
