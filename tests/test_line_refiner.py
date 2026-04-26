from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ocr_pipeline.cropper.line_cropper import LineCropper
from ocr_pipeline.detector.paddle_detector import DetectionResult
from ocr_pipeline.refiner.line_refiner import LineRefiner


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
