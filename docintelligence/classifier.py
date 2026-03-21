from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

_LOG = logging.getLogger(__name__)

try:
    import anthropic as _anthropic_lib
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

CATEGORIES: List[str] = [
    "Invoice",
    "Contract",
    "Report",
    "Research Paper",
    "Resume / CV",
    "Legal Document",
    "Financial Statement",
    "Letter / Email",
    "Form",
    "Other",
]

# Keyword heuristics used when Claude API is unavailable
_KEYWORDS: Dict[str, List[str]] = {
    "Invoice": ["invoice", "bill to", "amount due", "payment", "subtotal", "qty", "total due", "purchase order"],
    "Contract": ["agreement", "party", "parties", "terms and conditions", "signed", "witness", "hereby agrees"],
    "Report": ["report", "analysis", "findings", "conclusion", "executive summary", "results", "recommendations"],
    "Research Paper": ["abstract", "introduction", "methodology", "references", "bibliography", "doi", "citation"],
    "Resume / CV": ["experience", "education", "skills", "resume", "curriculum vitae", "objective", "summary of qualifications"],
    "Legal Document": ["whereas", "hereby", "pursuant", "jurisdiction", "plaintiff", "defendant", "court", "statute"],
    "Financial Statement": ["balance sheet", "income statement", "cash flow", "assets", "liabilities", "equity", "revenue"],
    "Letter / Email": ["dear", "sincerely", "regards", "to whom it may concern", "re:", "yours faithfully"],
    "Form": ["please fill", "checkbox", "signature", "date of birth", "form number", "please complete"],
}

CATEGORY_COLORS: Dict[str, str] = {
    "Invoice": "#FF6B6B",
    "Contract": "#4ECDC4",
    "Report": "#45B7D1",
    "Research Paper": "#96CEB4",
    "Resume / CV": "#FFEAA7",
    "Legal Document": "#DDA0DD",
    "Financial Statement": "#98D8C8",
    "Letter / Email": "#F7DC6F",
    "Form": "#AED6F1",
    "Other": "#BDC3C7",
}


def classify_document(text: str, title: str = "") -> Dict[str, Any]:
    """Classify document type using Claude API when available, keyword heuristics otherwise.

    Returns a dict with:
      - category (str)
      - confidence (float 0–1)
      - language (str)
      - key_topics (list[str])
      - date_mentioned (str | None)
      - author (str | None)
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if _HAS_ANTHROPIC and api_key:
        result = _claude_classify(text[:3000], title, api_key)
        if result:
            return result
    return _heuristic_classify(text)


def _claude_classify(text: str, title: str, api_key: str) -> Dict[str, Any] | None:
    categories_str = ", ".join(CATEGORIES)
    prompt = (
        f"Analyze this document and respond with a single JSON object (no markdown) containing:\n"
        f'- "category": one of [{categories_str}]\n'
        f'- "confidence": float 0.0–1.0\n'
        f'- "language": detected language (e.g. "English")\n'
        f'- "key_topics": list of 3–5 main topics (short strings)\n'
        f'- "date_mentioned": any prominent date found in the document, or null\n'
        f'- "author": author or organization name if found, or null\n\n'
        f"Document title: {title or '(unknown)'}\n\n"
        f"Document text (truncated):\n{text}\n\n"
        f"Respond with ONLY the JSON object."
    )
    try:
        client = _anthropic_lib.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        result.setdefault("confidence", 0.85)
        result.setdefault("language", "English")
        result.setdefault("key_topics", [])
        result.setdefault("date_mentioned", None)
        result.setdefault("author", None)
        return result
    except Exception as e:
        _LOG.warning("Claude classification failed: %s — falling back to heuristic", e)
        return None


def _heuristic_classify(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    scores: Dict[str, int] = {cat: 0 for cat in _KEYWORDS}
    for category, keywords in _KEYWORDS.items():
        scores[category] = sum(1 for kw in keywords if kw in text_lower)

    best = max(scores, key=lambda k: scores[k])
    total_hits = sum(scores.values())
    if scores[best] == 0:
        return {
            "category": "Other",
            "confidence": 0.1,
            "language": "Unknown",
            "key_topics": [],
            "date_mentioned": None,
            "author": None,
        }

    confidence = round(min(scores[best] / max(total_hits * 0.4, 1), 0.85), 2)
    return {
        "category": best,
        "confidence": confidence,
        "language": "Unknown",
        "key_topics": [],
        "date_mentioned": None,
        "author": None,
    }
