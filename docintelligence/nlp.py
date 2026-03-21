from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

_LOG = logging.getLogger(__name__)

try:
    import spacy
    from spacy.language import Language
    _HAS_SPACY = True
except ImportError:  # pragma: no cover
    spacy = None  # type: ignore
    Language = None  # type: ignore
    _HAS_SPACY = False

_NLP: Optional[object] = None

# Regex patterns for common entities — work with zero dependencies
_REGEX_PATTERNS: List[Dict[str, str]] = [
    {"label": "EMAIL",   "pattern": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"},
    {"label": "URL",     "pattern": r"https?://[^\s\"'<>]+"},
    {"label": "PHONE",   "pattern": r"\b(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"},
    {"label": "DATE",    "pattern": r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"},
    {"label": "MONEY",   "pattern": r"\$\s?\d[\d,]*(?:\.\d{1,2})?|\b\d[\d,]*(?:\.\d{1,2})?\s?(?:USD|EUR|GBP|dollars?|euros?|pounds?)\b"},
    {"label": "PERCENT", "pattern": r"\b\d+(?:\.\d+)?\s?%"},
    {"label": "TIME",    "pattern": r"\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b"},
]


def _regex_entities(text: str) -> List[Dict[str, str]]:
    """Extract entities using regex patterns — no external deps required."""
    found = []
    seen: set = set()
    for spec in _REGEX_PATTERNS:
        for m in re.finditer(spec["pattern"], text):
            val = m.group(0).strip()
            key = (val.lower(), spec["label"])
            if key not in seen:
                seen.add(key)
                found.append({"text": val, "label": spec["label"]})
    return found


def get_nlp(model: str = "en_core_web_sm") -> object:
    if not _HAS_SPACY:
        raise ImportError("spaCy is not installed.")
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load(model)
        except OSError:
            _LOG.warning("spaCy model %s not found. Run: python -m spacy download %s", model, model)
            raise
    return _NLP


def extract_entities(text: str, model: str = "en_core_web_sm") -> List[Dict[str, str]]:
    """Extract entities. Uses spaCy when available, regex patterns otherwise."""

    regex_results = _regex_entities(text)

    if not _HAS_SPACY:
        return regex_results

    try:
        nlp = get_nlp(model=model)
        doc = nlp(text)
        spacy_results = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        # Merge: spaCy first, then any regex hits not already covered
        spacy_texts = {e["text"].lower() for e in spacy_results}
        extra = [e for e in regex_results if e["text"].lower() not in spacy_texts]
        return spacy_results + extra
    except Exception as exc:
        _LOG.warning("spaCy extraction failed: %s — using regex fallback", exc)
        return regex_results


def extract_key_phrases(text: str, model: str = "en_core_web_sm") -> List[str]:
    """Extract key phrases. Uses spaCy noun chunks when available, else top TF-IDF terms."""

    if _HAS_SPACY:
        try:
            nlp = get_nlp(model=model)
            doc = nlp(text)
            phrases = sorted({chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2})
            if phrases:
                return phrases
        except Exception as exc:
            _LOG.warning("spaCy key phrase extraction failed: %s — using TF-IDF fallback", exc)

    # TF-IDF keyword fallback
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        if not sentences:
            return []
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=200)
        vec.fit_transform(sentences)
        scores = dict(zip(vec.get_feature_names_out(), vec.idf_))
        # Lower IDF = more common = more "key" in this doc
        sorted_terms = sorted(scores.items(), key=lambda x: x[1])
        return [t for t, _ in sorted_terms[:30]]
    except Exception:
        return []
