from __future__ import annotations

import logging
from typing import Dict, List, Optional

try:
    import spacy
    from spacy.language import Language
    _HAS_SPACY = True
except ImportError:  # pragma: no cover
    spacy = None  # type: ignore
    Language = None  # type: ignore
    _HAS_SPACY = False

_LOG = logging.getLogger(__name__)

_NLP: Optional["Language"] = None


def get_nlp(model: str = "en_core_web_sm") -> "Language":
    """Load a spaCy model (caches the model for reuse)."""

    if not _HAS_SPACY:
        raise ImportError(
            "spaCy is not installed. Install it (e.g. `pip install spacy`) to use NER/keyphrase extraction."
        )

    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load(model)
        except OSError:
            _LOG.warning(
                "spaCy model %s not found. Please run: python -m spacy download %s",
                model,
                model,
            )
            raise
    return _NLP


def extract_entities(text: str, model: str = "en_core_web_sm") -> List[Dict[str, str]]:
    """Extract entities from text using spaCy."""

    nlp = get_nlp(model=model)
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]


def extract_key_phrases(text: str, model: str = "en_core_web_sm") -> List[str]:
    """Extract simple noun chunks as key phrases."""

    nlp = get_nlp(model=model)
    doc = nlp(text)
    phrases = set([chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2])
    return sorted(phrases)
