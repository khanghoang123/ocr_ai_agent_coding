"""Regression tests for LineCropper polygon handling.

The Kraken BLLA detector returns *closed-boundary* polygons (the poly
traces the whole line perimeter). The original LineCropper assumed the
first-half-by-index of a polygon is always the top edge, which is true
for Paddle / CRAFT but false for closed boundaries — both halves contain
points on the top *and* bottom edges. The consequence was that
``_calculate_curvature()`` reported an inflated value, which routed the
polygon into ``_crop_unwarp()`` where the broken split produced
``height < min_height`` and silently dropped the line.

These tests lock in the fix:

* closed-boundary polygons are split by y-median and no longer produce
  a falsely-high curvature score;
* ``warp_polygon`` returns a valid crop on a closed-boundary polygon;
* the Paddle/CRAFT-style index split is still used when it is valid.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ocr_pipeline.cropper.line_cropper import LineCropper


def _make_image(width: int = 1200, height: int = 200) -> Image.Image:
    """Blank white page, large enough for the test polygons."""
    return Image.fromarray(np.full((height, width, 3), 255, dtype=np.uint8))


def _paddle_style_quad() -> np.ndarray:
    """Paddle 4-point poly: top-left, top-right, bottom-right, bottom-left."""
    return np.array(
        [[100, 50], [700, 50], [700, 90], [100, 90]],
        dtype=np.float32,
    )


def _kraken_style_closed_boundary() -> np.ndarray:
    """Realistic 26-point closed-boundary polygon (baseline-aware segmenter).

    Shape mirrors the real Kraken output on
    ``tests/test/thumb_1200_1698.png``: first-half-by-index and
    second-half-by-index both contain a mix of top-edge and bottom-edge
    points, and the index-based split fails the "top above bottom" check.
    """
    # Hand-crafted to reproduce Kraken's "wrap-around" ordering: a handful
    # of top-edge points, then the right cap, then the bottom edge, then
    # the left cap back up to the start.
    pts = [
        (364, 172), (362, 172), (353, 177), (277, 177), (266, 171),
        (260, 171), (250, 172), (240, 174), (230, 175), (220, 176),
        (200, 178), (180, 180), (150, 195), (138, 200), (140, 200),
        (180, 200), (220, 199), (260, 198), (300, 198), (340, 198),
        (376, 199), (376, 193), (375, 177), (370, 172), (368, 171),
        (366, 171),
    ]
    return np.array(pts, dtype=np.float32)


def test_split_top_bottom_index_split_used_for_paddle_quad():
    poly = _paddle_style_quad()
    top, bot = LineCropper._split_top_bottom(poly)
    assert top is not None and bot is not None
    # Paddle-style: top points all above bottom points.
    assert float(np.max(top[:, 1])) < float(np.min(bot[:, 1]))


def test_split_top_bottom_falls_back_to_y_median_for_closed_boundary():
    poly = _kraken_style_closed_boundary()
    top, bot = LineCropper._split_top_bottom(poly)
    assert top is not None and bot is not None
    # y-median split guarantees top points are not mixed with bottom
    # points — median of the poly sits roughly between the two edges.
    assert float(np.max(top[:, 1])) <= float(np.min(bot[:, 1])) + 1e-6


def test_calculate_curvature_uses_y_median_top_for_closed_boundary():
    """Fix routes the curvature fit through the actual top edge.

    Before the fix, the top-by-index of a closed-boundary polygon mixed
    bottom-edge points in, so ``_calculate_curvature`` fitted a line
    through a zig-zag of points at wildly different y values. After the
    fix the curvature score is bounded by the actual deviation of the
    real top edge from its linear fit, which is at most the line's own
    thickness relative to its width — well below 1.0 for any real line.
    """
    cropper = LineCropper(min_height=8, min_width=20)
    poly = _kraken_style_closed_boundary()
    score = cropper._calculate_curvature(poly)
    # The real top edge spans y∈[171, 180] across x∈[180, 376]; the
    # linear fit MAE divided by the height range must be finite and
    # well below 1.0 on any sane polygon.
    assert 0.0 <= score < 1.0, f"curvature score out of range: {score:.3f}"


def test_warp_polygon_returns_crop_for_closed_boundary():
    """Regression: closed-boundary polygons must not be silently dropped."""
    cropper = LineCropper(min_height=8, min_width=20)
    img = _make_image()
    poly = _kraken_style_closed_boundary()
    warped = cropper.warp_polygon(img, poly)
    assert warped is not None, (
        "warp_polygon should return a crop for a closed-boundary polygon; "
        "returning None drops the line from the final OCR output."
    )
    assert warped.image.shape[0] >= cropper.min_height
    assert warped.image.shape[1] >= cropper.min_width


def test_warp_polygon_still_works_for_paddle_quad():
    cropper = LineCropper(min_height=8, min_width=20)
    img = _make_image()
    poly = _paddle_style_quad()
    warped = cropper.warp_polygon(img, poly)
    assert warped is not None
    assert warped.image.shape[0] >= cropper.min_height
