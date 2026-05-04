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


def test_polygon_padding_inflates_quad_crop_dimensions():
    """Polygon-level padding must enlarge the output crop versus no padding.

    The recogniser was getting visibly tight crops on Kraken polygons
    (ascenders/descenders clipped). The fix adds polygon-level padding
    *before* the warp so the crop captures pixels around the polygon.
    """
    img = _make_image()
    poly = _paddle_style_quad()
    base = LineCropper(
        min_height=8, min_width=20,
        polygon_pad_v_ratio=0.0, polygon_pad_h_ratio=0.0,
    )
    padded = LineCropper(
        min_height=8, min_width=20,
        polygon_pad_v_ratio=0.4, polygon_pad_h_ratio=0.1,
    )
    base_crop = base.warp_polygon(img, poly)
    pad_crop = padded.warp_polygon(img, poly)
    assert base_crop is not None and pad_crop is not None
    # height grows by ~80% of the line height, width by ~20%
    assert pad_crop.image.shape[0] > base_crop.image.shape[0]
    assert pad_crop.image.shape[1] > base_crop.image.shape[1]


def test_split_top_bottom_typed_returns_kind_label():
    """warp_polygon relies on the typed split to skip the polynomial
    unwarp on closed-boundary polygons; the kind label must distinguish
    Paddle/CRAFT-style polys from Kraken-style closed boundaries."""
    # Paddle/CRAFT polygons walk the top edge left→right then the bottom
    # edge right→left, so the index split puts every top point in the
    # first half and every bottom point in the second half.
    paddle = np.array(
        [
            [10, 10], [40, 10], [70, 10], [110, 10],   # top edge
            [110, 40], [70, 40], [40, 40], [10, 40],   # bottom edge
        ],
        dtype=np.float32,
    )
    _, _, kind_idx = LineCropper._split_top_bottom_typed(paddle)
    assert kind_idx == "index"

    closed = _kraken_style_closed_boundary()
    _, _, kind_closed = LineCropper._split_top_bottom_typed(closed)
    assert kind_closed == "closed"


def test_polygon_background_mask_replaces_outside_pixels():
    """Pixels outside the polygon must be replaced with paper colour
    so neighbour-line bleed-in does not contaminate the recogniser.

    Setup: a page with the *target* line on a white strip and a black
    bar that crosses the warp's source quad just above the polygon.
    Without masking, the rotated-rect crop will include the black bar
    pixels (they sit inside the inflated quad). With masking, those
    pixels are replaced with the polygon's interior median colour.
    """
    arr = np.full((100, 400, 3), 240, dtype=np.uint8)
    # Black bar that fills the polygon padding region above the line.
    arr[20:39, 50:380] = 0
    # Light-grey "ink" inside the polygon at y=40..70 to give the mask
    # something paper-coloured to extrapolate.
    arr[40:70, 60:340] = 240
    img = Image.fromarray(arr)
    poly = np.array(
        [[60, 40], [340, 40], [340, 70], [60, 70]], dtype=np.float32
    )
    masked = LineCropper(
        min_height=8, min_width=20,
        mask_polygon_background=True,
        polygon_pad_v_ratio=0.6, polygon_pad_h_ratio=0.0,
        mask_dilation_px=0,
    ).warp_polygon(img, poly)
    unmasked = LineCropper(
        min_height=8, min_width=20,
        mask_polygon_background=False,
        polygon_pad_v_ratio=0.6, polygon_pad_h_ratio=0.0,
    ).warp_polygon(img, poly)
    assert masked is not None and unmasked is not None
    # Padding extends the crop above the polygon, so the unmasked crop
    # captures the black bar pixels and is appreciably darker than the
    # masked crop, which should be near the paper colour.
    assert masked.image.mean() > unmasked.image.mean() + 15, (
        f"mask did not lift mean brightness enough: "
        f"masked={masked.image.mean():.1f} unmasked={unmasked.image.mean():.1f}"
    )
