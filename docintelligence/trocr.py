from __future__ import annotations

from typing import Any, List

from PIL import Image

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    TrOCRProcessor = None  # type: ignore
    VisionEncoderDecoderModel = None  # type: ignore
    _HAS_TRANSFORMERS = False


def trocr_ocr(image: Image.Image, model_name: str = "microsoft/trocr-base-handwritten") -> List[str]:
    """Run TrOCR on an image (handwriting) and return a list of recognized lines."""

    if not _HAS_TRANSFORMERS:
        raise ImportError(
            "transformers is not installed. Install it (e.g. `pip install transformers`) to use TrOCR handwriting OCR."
        )

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
    # TrOCR returns a list of strings, typically one per image.
    return [p.strip() for p in preds if p and p.strip()]
