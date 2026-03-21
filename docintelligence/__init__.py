"""Document Intelligence core package."""

# Core (lightweight) components
from .ocr import ocr_image, ocr_pdf
from .parser import parse_layout
from .storage import DocumentStorage
from .utils import ensure_dir, file_hash
from .diff import diff_markdown, side_by_side_diff

# Optional/extra features (may be unavailable if dependencies are missing)
try:
    from .nlp import extract_entities, extract_key_phrases
except ImportError:  # pragma: no cover
    extract_entities = None  # type: ignore
    extract_key_phrases = None  # type: ignore

try:
    from .summarize import summarize_text
except ImportError:  # pragma: no cover
    summarize_text = None  # type: ignore

try:
    from .donut import extract_with_donut
except ImportError:  # pragma: no cover
    extract_with_donut = None  # type: ignore

try:
    from .trocr import trocr_ocr
except ImportError:  # pragma: no cover
    trocr_ocr = None  # type: ignore

try:
    from .search import SemanticSearch
except ImportError:  # pragma: no cover
    SemanticSearch = None  # type: ignore

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
    "extract_with_donut",
    "trocr_ocr",
    "SemanticSearch",
]
