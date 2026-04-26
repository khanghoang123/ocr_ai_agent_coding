"""HybridRectifier: DocTr++ primary + OpenCV fallback.

Selected by ``backend="hybrid"`` (the default). The fallback chain is:

    1. Try DocTr++. If `applied=True` AND `confidence >= min_confidence`,
       accept the result.
    2. Otherwise (weights missing, inference error, or low-confidence
       output), fall back to the OpenCV 4-corner rectifier.
    3. If OpenCV also returns `applied=False`, return the original image
       with full diagnostic context.
"""
from __future__ import annotations

import logging
from PIL import Image

from ocr_pipeline.rectifier.base import Rectifier, RectifierResult

logger = logging.getLogger(__name__)


class HybridRectifier(Rectifier):
    name = "hybrid"

    def __init__(
        self,
        primary: Rectifier,
        fallback: Rectifier,
        fallback_on_low_confidence: bool = True,
        min_confidence: float = 0.5,
    ):
        self.primary = primary
        self.fallback = fallback
        self.fallback_on_low_confidence = fallback_on_low_confidence
        self.min_confidence = float(min_confidence)

    def rectify(self, image: Image.Image) -> RectifierResult:
        primary = self.primary.rectify(image)
        accept_primary = (
            primary.applied
            and (
                not self.fallback_on_low_confidence
                or primary.confidence >= self.min_confidence
            )
        )
        if accept_primary:
            return RectifierResult(
                image=primary.image,
                backend_used=f"hybrid:{self.primary.name}",
                applied=True,
                confidence=primary.confidence,
                diagnostics={
                    "primary": primary.diagnostics,
                    "fallback_used": False,
                },
            )

        fallback = self.fallback.rectify(image)
        return RectifierResult(
            image=fallback.image,
            backend_used=f"hybrid:{self.fallback.name}" if fallback.applied else "hybrid:identity",
            applied=fallback.applied,
            confidence=fallback.confidence,
            diagnostics={
                "primary": primary.diagnostics,
                "primary_applied": primary.applied,
                "primary_confidence": primary.confidence,
                "fallback": fallback.diagnostics,
                "fallback_used": True,
                "reason_for_fallback": (
                    "primary_unavailable"
                    if not primary.applied
                    else "primary_low_confidence"
                ),
            },
        )
