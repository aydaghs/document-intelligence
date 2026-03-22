from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

_LOG = logging.getLogger(__name__)

try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    pipeline = None  # type: ignore
    _HAS_TRANSFORMERS = False

try:
    import anthropic as _anthropic_lib
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

_SUMMARIZER = None


def _is_table_row(sentence: str) -> bool:
    """Return True if the sentence looks like a table row or numeric data dump."""
    tokens = sentence.split()
    if not tokens:
        return True
    # High ratio of numeric/symbol tokens → likely a table row
    numeric = sum(1 for t in tokens if re.match(r'^[\d,.\-\(\)%/]+$', t))
    if len(tokens) > 3 and numeric / len(tokens) > 0.35:
        return True
    # Very short average word length → abbreviations / codes / column headers
    avg_len = sum(len(t) for t in tokens) / len(tokens)
    if avg_len < 4.5 and len(tokens) > 6:
        return True
    return False


def _extractive_summarize(text: str, num_sentences: int = 3) -> str:
    """Lightweight extractive summarization using TF-IDF sentence scoring."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    # Split on sentence boundaries
    raw = re.split(r'(?<=[.!?])\s+', text)

    # Keep only real sentences: long enough and not table/numeric rows
    sentences = [s.strip() for s in raw
                 if len(s.strip()) > 50 and not _is_table_row(s.strip())]

    if not sentences:
        # Fall back: pick first non-empty, non-table lines
        lines = [l.strip() for l in text.splitlines()
                 if len(l.strip()) > 50 and not _is_table_row(l.strip())]
        return " ".join(lines[:num_sentences]) if lines else text[:400].strip()

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    try:
        vec = TfidfVectorizer(stop_words="english", max_features=1000)
        mat = vec.fit_transform(sentences)
        scores = np.asarray(mat.sum(axis=1)).flatten()
        # Take top-scored sentences but preserve their original order
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


def _claude_summarize(text: str, api_key: str, num_sentences: int = 3) -> Optional[str]:
    """Use Claude API for high-quality abstractive summarization."""
    if not _HAS_ANTHROPIC:
        return None
    try:
        client = _anthropic_lib.Anthropic(api_key=api_key)
        prompt = (
            f"Summarize the following document in {num_sentences} concise sentences. "
            "Focus on the main topics, key findings, and important details. "
            "Return only the summary, no preamble.\n\n"
            f"{text[:4000]}"
        )
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        _LOG.warning("Claude summarization failed: %s", exc)
        return None


def summarize_text(
    text: str,
    max_length: int = 256,
    min_length: int = 60,
    num_sentences: int = 3,
    use_claude: bool = True,
) -> str:
    """Summarize text. Priority: Claude API → BART → extractive TF-IDF."""

    if not text or not text.strip():
        return ""

    # 1. Claude API (best quality, fast, low cost with Haiku)
    if use_claude:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if api_key:
            result = _claude_summarize(text, api_key, num_sentences=num_sentences)
            if result:
                return result

    # 2. BART (local transformer model)
    summarizer = get_summarizer()
    if summarizer is not None:
        try:
            input_text = text[:3000]
            result = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
            summary = result[0]["summary_text"].strip()
            sents = [s.strip() for s in summary.replace("\n", " ").split(".") if s.strip()]
            return ". ".join(sents[:num_sentences]).strip() + ("." if sents else "")
        except Exception as e:
            _LOG.warning("BART summarization failed: %s — falling back to extractive", e)

    # 3. Extractive TF-IDF (no dependencies)
    return _extractive_summarize(text, num_sentences=num_sentences)
