from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

_LOG = logging.getLogger(__name__)

try:
    import anthropic as _anthropic_lib
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> List[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def embed_chunks(chunks: List[str], search) -> "Optional[np.ndarray]":
    """Embed a list of text chunks using the SemanticSearch instance."""
    if not chunks or search is None:
        return None
    try:
        import numpy as np
        embs = search.embed(chunks)
        return embs
    except Exception as e:
        _LOG.warning("embed_chunks failed: %s", e)
        return None


def retrieve_chunks(
    query: str,
    chunks: List[str],
    chunk_embeddings,
    search,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Return the top-k most relevant chunks for a query."""
    if not chunks:
        return []
    candidates = [{"text": c, "chunk_index": i} for i, c in enumerate(chunks)]
    if search is None or chunk_embeddings is None:
        return candidates[:top_k]
    try:
        return search.search(query, candidates, chunk_embeddings, top_k=top_k)
    except Exception as e:
        _LOG.warning("retrieve_chunks failed: %s", e)
        return candidates[:top_k]


def answer_question(
    question: str,
    context_chunks: List[str],
    doc_titles: Optional[List[str]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Answer a question using retrieved context chunks.

    Uses Claude API when ANTHROPIC_API_KEY is set, otherwise falls back to
    a simple keyword-match excerpt.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if _HAS_ANTHROPIC and api_key:
        return _claude_answer(question, context_chunks, doc_titles, chat_history, api_key)
    return _fallback_answer(question, context_chunks)


def _claude_answer(
    question: str,
    context_chunks: List[str],
    doc_titles: Optional[List[str]],
    chat_history: Optional[List[Dict[str, str]]],
    api_key: str,
) -> str:
    title_str = ", ".join(doc_titles) if doc_titles else "the document"
    context = "\n\n---\n\n".join(
        f"[Excerpt {i + 1}]:\n{c}" for i, c in enumerate(context_chunks)
    )
    system_prompt = (
        f"You are an intelligent document assistant analyzing: {title_str}.\n"
        "Answer questions based ONLY on the provided document excerpts. "
        "If the answer is not in the context, clearly say so. "
        "Always cite which excerpt(s) support your answer (e.g. 'According to Excerpt 2...'). "
        "Be concise and precise."
    )
    messages: List[Dict[str, str]] = []
    if chat_history:
        for turn in chat_history:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {question}",
    })
    try:
        client = _anthropic_lib.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text.strip()
    except Exception as e:
        _LOG.error("Claude API call failed: %s", e)
        return f"_(Claude API error: {e})_\n\n" + _fallback_answer(question, context_chunks)


def _fallback_answer(question: str, context_chunks: List[str]) -> str:
    """Keyword-overlap fallback when Claude API is unavailable."""
    q_words = set(re.findall(r"\w+", question.lower()))
    best, best_score = "", -1
    for chunk in context_chunks:
        score = len(q_words & set(re.findall(r"\w+", chunk.lower())))
        if score > best_score:
            best_score = score
            best = chunk
    if best:
        return f"**Most relevant excerpt:**\n\n> {best[:700].strip()}…"
    return "_No relevant content found for this question._"
