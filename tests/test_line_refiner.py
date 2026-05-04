from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ocr_pipeline.cropper.line_cropper import LineCropper
from ocr_pipeline.detector.paddle_detector import DetectionResult
from ocr_pipeline.refiner.line_refiner import LineRefiner, _PageEntry


def _polygon(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _make_line_image(two_lines: bool = False) -> Image.Image:
    image = Image.new("RGB", (420, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 40, 360, 58], fill="black")
    if two_lines:
        draw.rectangle([40, 95, 360, 113], fill="black")
    return image


class _RecognizerStub:
    def __init__(self, prob: float, text: str = "stub"):
        self.prob = float(prob)
        self.text = text

    def recognize_batch(self, images, return_prob=False):
        assert return_prob is True
        return [(self.text, self.prob) for _ in images]


class _SegmentAwareRecognizerStub:
    """Return low confidence for multi-line crops, high for single-line crops."""

    def recognize_batch(self, images, return_prob=False):
        assert return_prob is True
        out = []
        for img in images:
            arr = np.array(img.convert("L"))
            ink = (arr < 128).astype(np.uint8)
            projection = ink.sum(axis=1).astype(np.float32)
            if projection.max() <= 0:
                out.append(("", 0.0))
                continue

            active = projection > (projection.max() * 0.35)
            segments = 0
            in_seg = False
            for is_active in active.tolist():
                if is_active and not in_seg:
                    in_seg = True
                    segments += 1
                elif not is_active and in_seg:
                    in_seg = False

            if segments >= 2:
                out.append(("merged unreadable text", 0.20))
            else:
                out.append(("single line", 0.82))
        return out


def test_line_refiner_keeps_single_line_boxes():
    image = _make_line_image(two_lines=False)
    detection = DetectionResult(
        polygons=[_polygon(35, 30, 370, 70)],
        confidences=[0.9],
    )

    refiner = LineRefiner()
    result = refiner.refine(image, detection, LineCropper())

    assert len(result.polygons) == 1
    assert result.decisions[0].action == "kept"
    assert result.decisions[0].tight_bbox is not None


def test_line_refiner_splits_merged_boxes_when_split_scores_higher():
    image = _make_line_image(two_lines=True)
    detection = DetectionResult(
        polygons=[_polygon(30, 25, 375, 125)],
        confidences=[0.9],
    )
    recognizer = _SegmentAwareRecognizerStub()

    refiner = LineRefiner()
    result = refiner.refine(image, detection, LineCropper(), recognizer=recognizer)

    assert len(result.polygons) == 2
    assert result.decisions[0].action == "split"


def test_line_refiner_falls_back_when_original_scores_higher():
    image = _make_line_image(two_lines=True)
    detection = DetectionResult(
        polygons=[_polygon(30, 25, 375, 125)],
        confidences=[0.9],
    )
    recognizer = _RecognizerStub(prob=0.99, text="strong original candidate")

    refiner = LineRefiner(split_score_margin=0.35)
    result = refiner.refine(image, detection, LineCropper(), recognizer=recognizer)

    assert len(result.polygons) == 1
    assert result.decisions[0].action == "kept"
    assert result.decisions[0].fallback_reason == "split_score_lower"


def test_line_refiner_neighbor_fallback_clamps_non_overlapping_lines():
    image = Image.new("RGB", (420, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([30, 34, 170, 52], fill="black")
    draw.rectangle([210, 86, 390, 106], fill="black")
    detection = DetectionResult(
        polygons=[
            _polygon(25, 28, 175, 58),
            _polygon(200, 54, 398, 118),
        ],
        confidences=[0.9, 0.88],
    )

    refiner = LineRefiner()
    result = refiner.refine(image, detection, LineCropper())

    assert len(result.polygons) == 2
    assert result.decisions[1].neighbor_strategy == "overlap_or_distance"
    assert result.decisions[1].clamped_bbox is not None
    assert result.decisions[1].clamped_bbox[1] >= result.decisions[0].raw_polygon[:, 1].max()


def _entry(x1: float, y1: float, x2: float, y2: float, idx: int = 0) -> _PageEntry:
    bbox = (float(x1), float(y1), float(x2), float(y2))
    return _PageEntry(
        index=idx,
        polygon=_polygon(x1, y1, x2, y2),
        bbox=bbox,
        center_y=(y1 + y2) / 2.0,
        height=max(y2 - y1, 1.0),
    )


def test_merge_near_duplicate_bands_does_not_cascade_for_overlapping_lines():
    """Regression: ascender/descender-rich polygons must not cascade-merge.

    Detectors that emit closed-boundary polygons (Kraken BLLA, Surya)
    routinely produce neighbouring line bboxes that overlap by 30-40%
    of their height because each polygon spans both ascenders and
    descenders. The previous merge logic treated any pair with
    ``overlap_ratio >= 0.35`` or strongly-negative gap as duplicates
    *regardless of resulting band height*, causing the merged band to
    grow tall enough to overlap the next polygon, which collapsed the
    entire page into a single band.
    """
    refiner = LineRefiner()
    page_median = 41.0
    # Eight neighbouring lines, each ~41px tall, line-to-line stride of
    # 28px → adjacent bboxes overlap by ~13/40 = 0.33 (just below the
    # old 0.35 threshold) but the merged pair would still cascade in
    # the strongly-negative-gap branch.
    entries = [_entry(40, 145 + 28 * i, 640, 186 + 28 * i, idx=i) for i in range(8)]
    merged = refiner._merge_near_duplicate_bands(entries, page_median)
    assert len(merged) == 8, (
        f"merge cascaded over ascender/descender overlap: got {len(merged)} bands "
        "for 8 distinct lines"
    )


def test_merge_near_duplicate_bands_still_merges_true_duplicates():
    """The merge must still collapse genuinely-duplicate bands."""
    refiner = LineRefiner()
    page_median = 40.0
    # Two near-identical bands at the same y range — combined height
    # stays inside the envelope (≤ 1.45 × median = 58px).
    entries = [
        _entry(40, 100, 640, 140, idx=0),
        _entry(45, 102, 638, 142, idx=1),
    ]
    merged = refiner._merge_near_duplicate_bands(entries, page_median)
    assert len(merged) == 1
