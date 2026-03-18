from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PIL import Image

_LOG = logging.getLogger(__name__)


def _load_donut(model_name: str = "naver-clova-ix/donut-base") -> Any:
    """Lazy-load Donut processor and model."""

    from transformers import DonutProcessor, VisionEncoderDecoderModel

    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model


def extract_with_donut(
    image: Image.Image,
    prompt: str = "",
    model_name: str = "naver-clova-ix/donut-base",
    max_length: int = 512,
) -> Dict[str, Any]:
    """Run Donut (layout-aware OCR + parsing) on an image.

    Note: This requires a large model download (~1GB)."""

    try:
        processor, model = _load_donut(model_name=model_name)
    except ImportError as exc:
        raise RuntimeError("transformers is required for Donut extraction") from exc

    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=max_length, decoder_start_token_id=processor.tokenizer.bos_token_id)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # The model typically returns a JSON-like string
    try:
        import json

        parsed = json.loads(generated_text)
    except Exception:
        parsed = {"text": generated_text}

    return {"raw": generated_text, "parsed": parsed}
