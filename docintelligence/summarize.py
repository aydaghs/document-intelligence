from __future__ import annotations

import logging
import re
from typing import List

_LOG = logging.getLogger(__name__)

try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    pipeline = None  # type: ignore
    _HAS_TRANSFORMERS = False

_SUMMARIZER = None


def _extractive_summarize(text: str, num_sentences: int = 3) -> str:
    """Lightweight extractive summarization using TF-IDF sentence scoring."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    # Split into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
    if not sentences:
        return text[:500].strip()
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    try:
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(sentences)
        scores = np.asarray(mat.sum(axis=1)).flatten()
        top_idx = sorted(np.argsort(scores)[-num_sentences:])
        return " ".join(sentences[i] for i in top_idx)
    except Exception:
        return " ".join(sentences[:num_sentences])


def get_summarizer(model_name: str = "facebook/bart-large-cnn"):
    if not _HAS_TRANSFORMERS:
        return None
    global _SUMMARIZER
    if _SUMMARIZER is None:
        try:
            _SUMMARIZER = pipeline("summarization", model=model_name)
        except Exception as exc:
            _LOG.warning("Failed to load summarization model: %s", exc)
            return None
    return _SUMMARIZER


def summarize_text(text: str, max_length: int = 256, min_length: int = 60, num_sentences: int = 3) -> str:
    """Summarize text. Uses BART if transformers is installed, extractive TF-IDF otherwise."""

    if not text or not text.strip():
        return ""

    summarizer = get_summarizer()
    if summarizer is not None:
        try:
            # BART has a ~1024 token limit; truncate input safely
            input_text = text[:3000]
            result = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
            summary = result[0]["summary_text"].strip()
            sentences = [s.strip() for s in summary.replace("\n", " ").split(".") if s.strip()]
            return ". ".join(sentences[:num_sentences]).strip() + ("." if sentences else "")
        except Exception as e:
            _LOG.warning("BART summarization failed: %s — falling back to extractive", e)

    return _extractive_summarize(text, num_sentences=num_sentences)
