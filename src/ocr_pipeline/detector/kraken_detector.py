"""Kraken BLLA baseline-aware line segmenter (Tier-1 candidate E3).

Kraken (https://kraken.re) is the segmentation backbone behind
eScriptorium and a number of historical-document HTR projects. Its BLLA
("Baseline Layout Analysis") model is **explicitly trained on
handwriting** and emits per-line polygons + baselines, which fits
VietOCR's per-line cropper exactly.

This backend uses Kraken's bundled default model (``blla.mlmodel``) and
returns each line's `boundary` polygon. Empty results are reported via
the standard ``low_detector_recall`` flag — no silent grid fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from ocr_pipeline.detector.base import empty_detection_result
from ocr_pipeline.detector.paddle_detector import DetectionResult

logger = logging.getLogger(__name__)


class KrakenBllaDetector:
    """Wrap kraken's BLLA segmenter behind the project ``LineDetector`` API."""

    BACKEND_NAME = "kraken_blla"

    def __init__(
        self,
        device: str = "cpu",
        text_direction: str = "horizontal-lr",
        min_polygon_points: int = 4,
        **_: Any,
    ):
        self.device = device
        self.text_direction = text_direction
        self.min_polygon_points = int(min_polygon_points)
        self._segment = None  # callable kraken.blla.segment

    def _ensure_loaded(self) -> None:
        if self._segment is not None:
            return
        try:
            from kraken import blla
        except ImportError as exc:
            raise RuntimeError(
                "Kraken is not installed. Run `pip install kraken` to use the "
                "kraken_blla detector backend."
            ) from exc
        self._segment = blla.segment
        logger.info("Kraken BLLA detector ready (device=%s).", self.device)

    # ── Public API ──────────────────────────────────────────────────────────

    def detect(self, image: Image.Image) -> DetectionResult:
        return self.detect_with_notebook_fallback(image)

    def detect_with_notebook_fallback(self, image: Image.Image) -> DetectionResult:
        try:
            self._ensure_loaded()
        except Exception as exc:
            return empty_detection_result(
                image,
                backend=self.BACKEND_NAME,
                reason=f"kraken_init_error:{type(exc).__name__}",
            )

        try:
            seg = self._segment(  # type: ignore[misc]
                image,
                text_direction=self.text_direction,
                device=self.device,
            )
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.warning("Kraken BLLA raised: %s", exc)
            return empty_detection_result(
                image,
                backend=self.BACKEND_NAME,
                reason=f"kraken_runtime_error:{type(exc).__name__}",
            )

        polygons: list[np.ndarray] = []
        confidences: list[float] = []
        sources: list[str] = []
        skipped_short_polygon = 0

        for line in getattr(seg, "lines", []) or []:
            boundary = getattr(line, "boundary", None)
            if not boundary:
                continue
            poly = np.asarray(boundary, dtype=np.float32).reshape(-1, 2)
            if poly.shape[0] < self.min_polygon_points:
                skipped_short_polygon += 1
                continue
            polygons.append(poly)
            # Kraken does not emit per-line confidences for BLLA. Use 1.0 so
            # we don't accidentally drop lines downstream; the leaderboard
            # uses geometry metrics rather than confidence.
            confidences.append(1.0)
            sources.append("kraken_blla")

        if not polygons:
            return empty_detection_result(
                image,
                backend=self.BACKEND_NAME,
                reason="kraken_returned_no_lines",
                extra_diagnostics={"skipped_short_polygon": skipped_short_polygon},
            )

        diagnostics = _diagnose_polygons(polygons, image.height)
        diagnostics.update(
            {
                "backend": self.BACKEND_NAME,
                "low_detector_recall": False,
                "skipped_short_polygon": skipped_short_polygon,
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
