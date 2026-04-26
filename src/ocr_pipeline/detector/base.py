"""Detector backend abstraction.

A `LineDetector` is anything that can take a PIL image and return a
`DetectionResult` containing per-line polygons. The pipeline does not care
which model produced them, so backends are interchangeable.

The factory `build_detector(name, **kwargs)` constructs a backend by name
without hard-coding the choice in the pipeline.

Backends MUST set `diagnostics["low_detector_recall"] = True` whenever
they return zero polygons. The pipeline uses that flag instead of silently
fabricating fallback bands.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

from ocr_pipeline.detector.paddle_detector import DetectionResult, PaddleDetector

logger = logging.getLogger(__name__)


@runtime_checkable
class LineDetector(Protocol):
    """Backend-neutral line detector interface.

    `detect_with_notebook_fallback(image)` must:
      * return a `DetectionResult` with axis-aligned-or-quadrilateral
        polygons in image (rectified) coordinates,
      * populate `diagnostics["low_detector_recall"] = True` when no
        text was found, instead of returning fabricated bands,
      * populate `diagnostics["backend"]` with a short backend identifier
        for debugging / leaderboard tagging.
    """

    def detect_with_notebook_fallback(self, image: Image.Image) -> DetectionResult: ...


def empty_detection_result(
    image: Image.Image,
    backend: str,
    reason: str,
    extra_diagnostics: dict | None = None,
) -> DetectionResult:
    """Helper: construct a detection result that explicitly says 'no boxes here'.

    The rule: if a backend cannot produce text-line polygons, it must
    return zero polygons + `low_detector_recall=True`. NEVER fabricate
    full-image-wide bands as a stand-in.
    """
    diagnostics: dict = {
        "backend": backend,
        "low_detector_recall": True,
        "reason": reason,
        "median_height": 0.0,
        "estimated_rows": 0.0,
        "coverage_ratio": 0.0,
    }
    if extra_diagnostics:
        diagnostics.update(extra_diagnostics)
    return DetectionResult(
        polygons=[],
        confidences=[],
        sources=[],
        diagnostics=diagnostics,
    )


def build_detector(name: str, **kwargs) -> LineDetector:
    """Construct a detector backend by name.

    Supported names (case-insensitive):
      * ``paddle`` — current PaddleOCR DBNet (with grid fallback now disabled
        by default; pass ``allow_grid_fallback=True`` to opt in for debug).
      * ``surya``  — Surya document detector (line-level polygons).
      * ``craft``  — CRAFT word detector + row clustering into line polygons.
      * ``kraken`` — Kraken BLLA baseline-aware line segmenter.

    Unknown names raise ``ValueError`` so misconfigured experiments fail
    loudly rather than silently falling back to PaddleOCR.
    """
    backend = (name or "paddle").strip().lower()

    if backend in ("paddle", "paddleocr", "dbnet"):
        return PaddleDetector(**kwargs)

    if backend == "surya":
        from ocr_pipeline.detector.surya_detector import SuryaDetector
        return SuryaDetector(**kwargs)

    if backend == "craft":
        from ocr_pipeline.detector.craft_detector import CraftDetector
        return CraftDetector(**kwargs)

    if backend in ("kraken", "kraken_blla", "blla"):
        from ocr_pipeline.detector.kraken_detector import KrakenBllaDetector
        return KrakenBllaDetector(**kwargs)

    raise ValueError(
        f"Unknown detector backend: {name!r}. "
        f"Supported: paddle, surya, craft, kraken."
    )


def polygon_from_quad(quad: list[list[float]]) -> np.ndarray:
    """Coerce a 4-point list into an (N, 2) float32 numpy array.

    Used by every backend to keep the on-the-wire polygon format consistent.
    """
    arr = np.asarray(quad, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) polygon, got shape {arr.shape}")
    return arr
