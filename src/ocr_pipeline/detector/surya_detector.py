"""Surya document detector backend (Tier-1 candidate E1).

Surya (https://github.com/VikParuchuri/surya) is a modern document OCR
toolkit. We use only its detection component — a small EfficientViT-style
segmenter trained on a large multilingual document corpus that includes
handwriting and Vietnamese. It outputs **per-line** polygons natively,
which is exactly what VietOCR's cropper expects.

Implementation notes:
  * We pin to surya-ocr 0.6.x (`batch_text_detection` API). Newer Surya
    (0.17+) couples the detection model to a quantised torch runtime that
    is incompatible with the project's torch 2.11 build; the heatmap goes
    flat and every page collapses to a single full-image box. The old
    API still ships the same DBNet-style backbone and is stable.
  * No silent grid fallback. If Surya returns zero boxes, we report
    ``low_detector_recall=True`` and let the pipeline drop the page.
  * Confidence per polygon is preserved.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from ocr_pipeline.detector.base import empty_detection_result
from ocr_pipeline.detector.paddle_detector import DetectionResult

logger = logging.getLogger(__name__)


class SuryaDetector:
    """Wrap Surya's detection model behind the project ``LineDetector`` API."""

    BACKEND_NAME = "surya"

    def __init__(
        self,
        device: str = "cpu",
        min_confidence: float = 0.4,
        **_: Any,
    ):
        self.device = device
        self.min_confidence = float(min_confidence)
        self._model = None
        self._processor = None
        self._batch_text_detection = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from surya.detection import batch_text_detection
            from surya.model.detection.model import load_model, load_processor
        except ImportError as exc:
            raise RuntimeError(
                "Surya is not installed. Run `pip install 'surya-ocr<0.7'` to "
                "use the surya detector backend (project-tested at 0.6.13)."
            ) from exc

        logger.info("Initializing Surya detector (device=%s) ...", self.device)
        self._model = load_model()
        self._processor = load_processor()
        self._batch_text_detection = batch_text_detection
        logger.info("Surya detector ready.")

    # ── Public API ──────────────────────────────────────────────────────────

    def detect(self, image: Image.Image) -> DetectionResult:
        return self.detect_with_notebook_fallback(image)

    def detect_with_notebook_fallback(self, image: Image.Image) -> DetectionResult:
        try:
            self._ensure_loaded()
        except Exception as exc:
            logger.warning("Surya init failed: %s", exc)
            return empty_detection_result(
                image,
                backend=self.BACKEND_NAME,
                reason=f"surya_init_error:{type(exc).__name__}",
            )

        try:
            results = self._batch_text_detection([image], self._model, self._processor)
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.warning("Surya detection raised: %s", exc)
            return empty_detection_result(
                image,
                backend=self.BACKEND_NAME,
                reason=f"surya_runtime_error:{type(exc).__name__}",
            )

        if not results:
            return empty_detection_result(
                image,
                backend=self.BACKEND_NAME,
                reason="surya_returned_empty",
            )

        page = results[0]
        polygons: list[np.ndarray] = []
        confidences: list[float] = []
        sources: list[str] = []
        dropped_low_conf = 0

        for box in page.bboxes:
            conf = float(getattr(box, "confidence", 0.0) or 0.0)
            if conf < self.min_confidence:
                dropped_low_conf += 1
                continue
            poly = np.asarray(box.polygon, dtype=np.float32).reshape(-1, 2)
            if poly.shape[0] < 4:
                continue
            polygons.append(poly)
            confidences.append(conf)
            sources.append("surya_full_page")

        if not polygons:
            return empty_detection_result(
                image,
                backend=self.BACKEND_NAME,
                reason="surya_all_below_confidence",
                extra_diagnostics={"dropped_low_conf": dropped_low_conf},
            )

        diagnostics = _diagnose_polygons(polygons, image.height)
        diagnostics.update(
            {
                "backend": self.BACKEND_NAME,
                "low_detector_recall": False,
                "min_confidence": self.min_confidence,
                "dropped_low_conf": dropped_low_conf,
                "patchwise_used": False,
            }
        )
        return DetectionResult(
            polygons=polygons,
            confidences=confidences,
            sources=sources,
            diagnostics=diagnostics,
        )


def _diagnose_polygons(polygons: list[np.ndarray], image_height: int) -> dict:
    """Compute the standard diagnostic block (median height / coverage)."""
    if not polygons:
        return {"median_height": 0.0, "estimated_rows": 0.0, "coverage_ratio": 0.0}
    heights = []
    for poly in polygons:
        pts = poly.reshape(-1, 2)
        h = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        heights.append(max(h, 1.0))
    median_height = float(np.median(heights))
    estimated_rows = float(image_height / max(median_height, 1.0))
    coverage_ratio = float(len(polygons) / max(estimated_rows, 1.0))
    return {
        "median_height": median_height,
        "estimated_rows": estimated_rows,
        "coverage_ratio": coverage_ratio,
    }
