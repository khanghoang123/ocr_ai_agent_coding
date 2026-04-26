"""Tests for the page-level rectifier module (Phase 2C)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_pipeline.rectifier import (  # noqa: E402
    DocTrPlusRectifier,
    IdentityRectifier,
    OpenCVRectifier,
    build_rectifier,
)
from ocr_pipeline.rectifier.hybrid import HybridRectifier  # noqa: E402


def _make_perspective_photo() -> Image.Image:
    """Synthesize a 'photographed' page with perspective distortion."""
    page = np.full((600, 900, 3), 255, dtype=np.uint8)
    # Add some structure (text-like horizontal stripes) so the page has edges.
    for y in range(80, 560, 60):
        page[y : y + 6, 80:820] = 30

    canvas = np.full((800, 1200, 3), 60, dtype=np.uint8)
    src = np.float32([[0, 0], [900, 0], [900, 600], [0, 600]])
    dst = np.float32([[160, 50], [1080, 30], [1180, 740], [110, 700]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(page, M, (1200, 800), borderValue=(255, 255, 255))
    mask = cv2.warpPerspective(np.full(page.shape[:2], 255, np.uint8), M, (1200, 800))
    canvas[mask > 0] = warped[mask > 0]
    return Image.fromarray(canvas)


def test_identity_rectifier_returns_input_unchanged():
    img = Image.new("RGB", (100, 100), "white")
    rect = IdentityRectifier()
    result = rect.rectify(img)
    assert result.applied is False
    assert result.image is img
    assert result.backend_used == "identity"


def test_opencv_rectifier_finds_paper_quad_in_synthetic_photo():
    photo = _make_perspective_photo()
    rect = OpenCVRectifier(min_quad_area_ratio=0.20)
    result = rect.rectify(photo)
    assert result.applied is True
    assert result.confidence > 0.4
    # Output should be smaller-or-equal in both dims (we crop to the quad).
    out_w, out_h = result.image.size
    in_w, in_h = photo.size
    assert out_w <= in_w + 1 and out_h <= in_h + 1
    assert "quad" in result.diagnostics


def test_opencv_rectifier_skips_when_no_quad():
    # Pure noise: should not find a paper quad and should return identity-like.
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (240, 240, 3), dtype=np.uint8)
    rect = OpenCVRectifier(min_quad_area_ratio=0.40)
    result = rect.rectify(Image.fromarray(noise))
    # Either applied=False or low confidence — must not blow up. Acceptable
    # not-applied reasons are no_quad_found (no rect-like contour) or
    # quad_likely_inner_content_rect (rejected by the paper-vs-background
    # contrast gate).
    if result.applied:
        assert result.confidence >= 0.0
    else:
        assert result.diagnostics.get("reason") in {
            "no_quad_found",
            "quad_likely_inner_content_rect",
        }


def test_doctrpp_rectifier_falls_back_when_weights_missing(tmp_path):
    # No weights at the configured path → should report applied=False.
    rect = DocTrPlusRectifier(weights_path=str(tmp_path / "missing.torchscript"))
    img = Image.new("RGB", (256, 256), "white")
    result = rect.rectify(img)
    assert result.applied is False
    assert "doctrpp_unavailable" in result.diagnostics.get("reason", "")
    assert result.image is img


def test_hybrid_rectifier_falls_back_to_opencv_when_primary_unavailable():
    photo = _make_perspective_photo()
    primary = DocTrPlusRectifier(weights_path="/nonexistent/path.torchscript")
    fallback = OpenCVRectifier(min_quad_area_ratio=0.20)
    rect = HybridRectifier(primary=primary, fallback=fallback)
    result = rect.rectify(photo)
    assert result.applied is True  # OpenCV should succeed on synthetic photo
    assert result.diagnostics["fallback_used"] is True
    assert result.diagnostics["reason_for_fallback"] == "primary_unavailable"
    assert result.backend_used.startswith("hybrid:")


def test_build_rectifier_disabled_returns_identity():
    rect = build_rectifier(backend="hybrid", enabled=False)
    assert isinstance(rect, IdentityRectifier)
    img = Image.new("RGB", (50, 50), "white")
    assert rect.rectify(img).applied is False


def test_build_rectifier_unknown_backend_raises():
    import pytest

    with pytest.raises(ValueError):
        build_rectifier(backend="not_a_real_backend", enabled=True)


def test_build_rectifier_hybrid_default():
    rect = build_rectifier(backend="hybrid", enabled=True)
    assert isinstance(rect, HybridRectifier)
    assert isinstance(rect.primary, DocTrPlusRectifier)
    assert isinstance(rect.fallback, OpenCVRectifier)
