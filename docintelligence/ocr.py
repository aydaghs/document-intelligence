import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional dependencies are imported lazily to make the module usable even when some
# libraries are missing (e.g., Streamlit Cloud environments with limited install budgets).
try:
    import easyocr
    _HAS_EASYOCR = True
except ImportError:  # pragma: no cover
    easyocr = None  # type: ignore
    _HAS_EASYOCR = False

try:
    import pytesseract
    _HAS_PYTESSERACT = True
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore
    _HAS_PYTESSERACT = False

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:
    from pdf2image import convert_from_path
    _HAS_PDF2IMAGE = True
except ImportError:  # pragma: no cover
    convert_from_path = None  # type: ignore
    _HAS_PDF2IMAGE = False

_LOG = logging.getLogger(__name__)

# lazy init for performance
_EASYOCR_READER: Optional["easyocr.Reader"] = None


def _ensure_reader(lang_list: List[str] = ["en"]) -> "easyocr.Reader":
    global _EASYOCR_READER
    if not _HAS_EASYOCR:
        raise ImportError(
            "easyocr is not installed. Install it (e.g. `pip install easyocr`) to use OCR features, or use pytesseract instead."
        )
    if _EASYOCR_READER is None:
        _EASYOCR_READER = easyocr.Reader(lang_list, gpu=False)
    return _EASYOCR_READER


def ocr_image(image: "Image.Image", lang_list: List[str] = ["en"]) -> List[Dict[str, Any]]:
    """Run OCR on a PIL image and return structured results."""

    if _HAS_EASYOCR:
        reader = _ensure_reader(lang_list)
        raw = reader.readtext(image, detail=1)

        # easyocr returns (bbox, text, confidence)
        results: List[Dict[str, Any]] = []
        for bbox, text, conf in raw:
            results.append({
                "text": text,
                "confidence": float(conf),
                "bbox": [list(map(float, pt)) for pt in bbox],
            })
        return results

    if _HAS_PYTESSERACT:
        # Fallback to pytesseract if easyocr isn't available.
        return ocr_image_with_tesseract(image, lang=lang_list[0] if lang_list else "eng")

    raise ImportError(
        "No OCR backend is installed. Install easyocr or pytesseract to enable OCR functionality."
    )


def ocr_pdf(pdf_path: str, dpi: int = 300, lang_list: List[str] = ["en"]) -> List[Dict[str, Any]]:
    """Convert a PDF into images and run OCR per page."""

    if not _HAS_PDF2IMAGE:
        raise ImportError(
            "pdf2image is not installed. Install it to run OCR on PDFs (e.g. `pip install pdf2image`)."
        )

    pages = convert_from_path(pdf_path, dpi=dpi)
    doc_results: List[Dict[str, Any]] = []

    for page_idx, page in enumerate(pages, start=1):
        _LOG.info("OCR: page %s", page_idx)
        page_data = ocr_image(page, lang_list=lang_list)
        doc_results.append({"page": page_idx, "blocks": page_data})

    return doc_results


def ocr_image_with_tesseract(image: "Image.Image", lang: str = "eng") -> List[Dict[str, Any]]:
    """Fallback OCR using pytesseract; returns word-level boxes."""

    if not _HAS_PYTESSERACT:
        raise ImportError(
            "pytesseract is not installed. Install it (e.g. `pip install pytesseract`) to use this fallback OCR backend."
        )

    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    results: List[Dict[str, Any]] = []
    n_boxes = len(data["level"])
    for i in range(n_boxes):
        text = data["text"][i].strip()
        if not text:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        results.append(
            {
                "text": text,
                "confidence": float(data["conf"][i]) if data["conf"][i] != "-1" else None,
                "bbox": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            }
        )
    return results


def text_from_ocr_blocks(ocr_blocks: List[Dict[str, Any]]) -> str:
    """Flatten OCR blocks into a single string."""

    lines: List[str] = []
    for block in ocr_blocks:
        if "text" in block and block["text"]:
            lines.append(block["text"].strip())
    return "\n".join(lines)
