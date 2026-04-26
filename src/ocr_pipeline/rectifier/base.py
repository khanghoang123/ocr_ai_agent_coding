"""Rectifier interface + factory."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class RectifierResult:
    """Output of a rectification step.

    Always returns a *valid* PIL image — even when rectification is skipped
    or fails — so the rest of the pipeline never has to special-case it.
    """

    image: Image.Image
    backend_used: str
    applied: bool
    confidence: float = 1.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Rectifier(ABC):
    """Abstract base class for page-level dewarping backends."""

    name: str = "abstract"

    @abstractmethod
    def rectify(self, image: Image.Image) -> RectifierResult:
        """Rectify a single PIL RGB image."""
        raise NotImplementedError


class IdentityRectifier(Rectifier):
    """No-op rectifier — returns the input unchanged."""

    name = "identity"

    def rectify(self, image: Image.Image) -> RectifierResult:
        return RectifierResult(image=image, backend_used=self.name, applied=False)


def build_rectifier(
    backend: str = "hybrid",
    *,
    enabled: bool = True,
    weights_path: str | None = None,
    device: str = "cpu",
    min_quad_area_ratio: float = 0.25,
    min_confidence: float = 0.5,
    fallback_on_low_confidence: bool = True,
    **kwargs: Any,
) -> Rectifier:
    """Factory: build a Rectifier given a backend name.

    backend ∈ {"identity", "opencv", "doctrpp", "hybrid"}.

    `enabled=False` always returns an IdentityRectifier so callers can wire
    this through a single config flag without branching.

    `min_confidence` is only consumed by ``HybridRectifier`` (the threshold
    above which the primary backend's output is accepted). It is not
    forwarded to the per-backend constructors, which makes the wiring
    explicit and avoids the kwarg being silently dropped by the leaf
    rectifiers.
    """
    if not enabled:
        return IdentityRectifier()

    backend = (backend or "hybrid").lower().strip()

    if backend == "identity":
        return IdentityRectifier()

    # Imported lazily to avoid circular imports at module load.
    from ocr_pipeline.rectifier.opencv_rectifier import OpenCVRectifier
    from ocr_pipeline.rectifier.doctrpp_rectifier import DocTrPlusRectifier
    from ocr_pipeline.rectifier.hybrid import HybridRectifier

    if backend == "opencv":
        return OpenCVRectifier(min_quad_area_ratio=min_quad_area_ratio, **kwargs)

    if backend == "doctrpp":
        return DocTrPlusRectifier(weights_path=weights_path, device=device, **kwargs)

    if backend == "hybrid":
        primary = DocTrPlusRectifier(weights_path=weights_path, device=device, **kwargs)
        fallback = OpenCVRectifier(min_quad_area_ratio=min_quad_area_ratio, **kwargs)
        return HybridRectifier(
            primary=primary,
            fallback=fallback,
            fallback_on_low_confidence=fallback_on_low_confidence,
            min_confidence=min_confidence,
        )

    raise ValueError(f"Unknown rectifier backend: {backend!r}")


def to_numpy_rgb(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGB uint8 numpy array."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def from_numpy_rgb(arr: np.ndarray) -> Image.Image:
    """Convert an RGB uint8 numpy array back to a PIL image."""
    if arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")
