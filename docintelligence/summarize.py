from __future__ import annotations

import logging
from typing import List, Optional

try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    pipeline = None  # type: ignore
    _HAS_TRANSFORMERS = False

_LOG = logging.getLogger(__name__)

_SUMMARIZER = None


def get_summarizer(model_name: str = "facebook/bart-large-cnn"):
    if not _HAS_TRANSFORMERS:
        raise ImportError(
            "transformers is not installed. Install it (e.g. `pip install transformers`) to use summarization."
        )

    global _SUMMARIZER
    if _SUMMARIZER is None:
        _SUMMARIZER = pipeline("summarization", model=model_name)
    return _SUMMARIZER


def summarize_text(text: str, max_length: int = 256, min_length: int = 60, num_sentences: int = 3) -> str:
    """Generate a short summary of the given text."""

    if not text or not text.strip():
        return ""

    summarizer = get_summarizer()
    try:
        result = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        summary = result[0]["summary_text"].strip()
        # Ensure roughly num_sentences (heuristic)
        sentences = [s.strip() for s in summary.replace("\n", " ").split(".") if s.strip()]
        return ". ".join(sentences[:num_sentences]).strip() + ("." if sentences else "")
    except Exception as e:
        _LOG.warning("Summarization failed: %s", e)
        return ""
