"""Document Intelligence core package."""

# Core components — always available
from .ocr import ocr_image, ocr_pdf
from .parser import parse_layout
from .storage import DocumentStorage
from .utils import ensure_dir, file_hash
from .diff import diff_markdown, side_by_side_diff

# These modules now have built-in lightweight fallbacks and are always importable
from .nlp import extract_entities, extract_key_phrases
from .summarize import summarize_text
from .search import SemanticSearch

# Optional heavy features (require transformers)
try:
    from .donut import extract_with_donut
except ImportError:  # pragma: no cover
    extract_with_donut = None  # type: ignore

try:
    from .trocr import trocr_ocr
except ImportError:  # pragma: no cover
    trocr_ocr = None  # type: ignore

__all__ = [
    "ocr_image",
    "ocr_pdf",
    "parse_layout",
    "DocumentStorage",
    "ensure_dir",
    "file_hash",
    "diff_markdown",
    "side_by_side_diff",
    "extract_entities",
    "extract_key_phrases",
    "summarize_text",
    "SemanticSearch",
    "extract_with_donut",
    "trocr_ocr",
]
