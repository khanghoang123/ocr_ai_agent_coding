from __future__ import annotations

import numpy as np

from ocr_pipeline.detector.paddle_detector import DetectionResult, PaddleDetector


def _polygon(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def test_patchwise_merge_deduplicates_same_text_line():
    detector = PaddleDetector()
    primary = DetectionResult(
        polygons=[_polygon(20, 100, 220, 132)],
        confidences=[0.72],
        sources=["full_page"],
    )
    secondary = DetectionResult(
        polygons=[_polygon(18, 98, 222, 130)],
        confidences=[0.85],
        sources=["patch_0"],
    )

    merged = detector._merge_results(primary, secondary)

    assert merged.num_boxes == 1
    assert merged.sources == ["patch_0"]
    assert merged.confidences == [0.85]


def test_undersegmentation_heuristic_triggers_on_tall_notebook_page():
    detector = PaddleDetector()
    result = DetectionResult(
        polygons=[_polygon(20, 40 + idx * 120, 720, 88 + idx * 120) for idx in range(8)],
        confidences=[0.8] * 8,
        diagnostics={"median_height": 48.0, "estimated_rows": 28.0, "coverage_ratio": 0.28},
    )

    assert detector._looks_undersegmented(result, image_height=1400) is True
