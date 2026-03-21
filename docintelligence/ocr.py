from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

_LOG = logging.getLogger(__name__)

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
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    Image = None  # type: ignore
    _HAS_PIL = False

try:
    from pdf2image import convert_from_path
    _HAS_PDF2IMAGE = True
except ImportError:  # pragma: no cover
    convert_from_path = None  # type: ignore
    _HAS_PDF2IMAGE = False

try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:  # pragma: no cover
    pdfplumber = None  # type: ignore
    _HAS_PDFPLUMBER = False

_EASYOCR_READER: Optional[Any] = None


def _ensure_reader(lang_list: List[str] = ["en"]) -> Any:
    global _EASYOCR_READER
    if not _HAS_EASYOCR:
        raise ImportError("easyocr is not installed.")
    if _EASYOCR_READER is None:
        _EASYOCR_READER = easyocr.Reader(lang_list, gpu=False)
    return _EASYOCR_READER


def ocr_image(image: Any, lang_list: List[str] = ["en"]) -> List[Dict[str, Any]]:
    """Run OCR on a PIL image. Uses EasyOCR → pytesseract → raises if neither available."""

    if _HAS_EASYOCR:
        reader = _ensure_reader(lang_list)
        raw = reader.readtext(image, detail=1)
        return [
            {"text": text, "confidence": float(conf),
             "bbox": [list(map(float, pt)) for pt in bbox]}
            for bbox, text, conf in raw
        ]

    if _HAS_PYTESSERACT:
        return ocr_image_with_tesseract(image, lang=lang_list[0] if lang_list else "eng")

    raise ImportError(
        "No OCR backend available. Install easyocr or pytesseract for image/scanned-PDF OCR."
    )


def ocr_image_with_tesseract(image: Any, lang: str = "eng") -> List[Dict[str, Any]]:
    if not _HAS_PYTESSERACT:
        raise ImportError("pytesseract is not installed.")
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    results = []
    for i in range(len(data["level"])):
        text = data["text"][i].strip()
        if not text:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        conf = data["conf"][i]
        results.append({
            "text": text,
            "confidence": float(conf) if conf != "-1" else None,
            "bbox": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        })
    return results


def ocr_pdf(pdf_path: str, dpi: int = 300, lang_list: List[str] = ["en"]) -> List[Dict[str, Any]]:
    """Extract text from a PDF.

    Priority:
      1. pdfplumber  — pure-Python, works everywhere, good for text PDFs
      2. pdf2image + OCR — for scanned PDFs (requires Poppler + OCR backend)
    """

    # Always try pdfplumber first (no system deps)
    if _HAS_PDFPLUMBER:
        try:
            return _ocr_pdf_pdfplumber(pdf_path)
        except Exception as exc:
            _LOG.warning("pdfplumber extraction failed: %s — trying pdf2image", exc)

    # Fallback: rasterize pages with pdf2image and run OCR
    if _HAS_PDF2IMAGE:
        pages_imgs = convert_from_path(pdf_path, dpi=dpi)
        return [
            {"page": i + 1, "blocks": ocr_image(img, lang_list=lang_list)}
            for i, img in enumerate(pages_imgs)
        ]

    raise ImportError(
        "Neither pdfplumber nor pdf2image is installed. Install pdfplumber to process PDFs."
    )


def _ocr_pdf_pdfplumber(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text and layout blocks from a PDF using pdfplumber."""
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            blocks = []
            # Word-level extraction preserves bounding boxes
            words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            if words:
                for w in words:
                    x0, y0, x1, y1 = w["x0"], w["top"], w["x1"], w["bottom"]
                    blocks.append({
                        "text": w["text"],
                        "confidence": None,
                        "bbox": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                    })
            else:
                # No word-level data — fall back to full-page text
                text = page.extract_text() or ""
                if text.strip():
                    blocks.append({
                        "text": text,
                        "confidence": None,
                        "bbox": [[0, 0], [page.width, 0], [page.width, page.height], [0, page.height]],
                    })
            results.append({"page": page_idx, "blocks": blocks})
    return results


def text_from_ocr_blocks(ocr_blocks: List[Dict[str, Any]]) -> str:
    return "\n".join(b["text"].strip() for b in ocr_blocks if b.get("text"))
