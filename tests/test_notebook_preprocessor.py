from __future__ import annotations

import cv2
import numpy as np

from ocr_pipeline.preprocess.notebook_preprocessor import NotebookPreprocessor


def _make_notebook_crop() -> np.ndarray:
    image = np.full((80, 240, 3), 255, dtype=np.uint8)
    cv2.line(image, (0, 54), (239, 54), (0, 0, 0), 2)
    cv2.rectangle(image, (40, 18), (92, 34), (0, 0, 0), -1)
    cv2.rectangle(image, (122, 24), (194, 40), (0, 0, 0), -1)
    cv2.rectangle(image, (64, 60), (150, 72), (0, 0, 0), -1)
    return image


def _make_curved_crop() -> np.ndarray:
    image = np.full((90, 280, 3), 255, dtype=np.uint8)
    xs = np.arange(20, 260)
    centers = 42 + (np.sin(np.linspace(0, np.pi, len(xs))) * 8).astype(np.int32)
    for x, center in zip(xs, centers):
        cv2.line(image, (int(x), int(center - 8)), (int(x), int(center + 8)), (0, 0, 0), 1)
    return image


def _make_clean_printed_crop() -> np.ndarray:
    image = np.full((70, 240, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (36, 24), (204, 38), (0, 0, 0), -1)
    return image


def _center_variation(mask: np.ndarray) -> float:
    centers = []
    for col_idx in range(mask.shape[1]):
        ys = np.where(mask[:, col_idx] > 0)[0]
        if ys.size < 3:
            continue
        centers.append(float((ys[0] + ys[-1]) / 2.0))
    return float(np.std(centers)) if centers else 0.0


def test_tight_bounds_trim_whitespace():
    preprocessor = NotebookPreprocessor()
    mask = np.zeros((80, 240), dtype=np.uint8)
    mask[18:60, 42:190] = 255
    bounds = preprocessor.compute_tight_bounds(mask)

    assert bounds is not None
    x0, y0, x1, y1 = bounds
    assert x0 > 0
    assert y0 > 0
    assert x1 < mask.shape[1]
    assert y1 < mask.shape[0]


def test_mask_building_preserves_text_strokes_with_notebook_line():
    preprocessor = NotebookPreprocessor()
    crop = _make_notebook_crop()
    artifacts = preprocessor.build_mask(crop)
    bounds = preprocessor.compute_tight_bounds(artifacts.text_mask)

    assert bounds is not None
    assert int(np.count_nonzero(artifacts.text_mask[:, 40:190])) > 0
    assert bounds[3] - bounds[1] < crop.shape[0]


def test_curve_rectification_reduces_curve_score():
    preprocessor = NotebookPreprocessor(curve_threshold=0.01)
    crop = _make_curved_crop()
    mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
    xs = np.arange(20, 260)
    centers = 42 + (np.sin(np.linspace(0, np.pi, len(xs))) * 10).astype(np.int32)
    for x, center in zip(xs, centers):
        cv2.line(mask, (int(x), int(center - 10)), (int(x), int(center + 10)), 255, 2)

    before = _center_variation(mask)
    rectified_crop, rectified_mask = preprocessor.rectify_curved_crop(crop, mask)
    after = _center_variation(rectified_mask)

    assert rectified_crop.shape[1] == crop.shape[1]
    assert before > 0
    assert after < before


def test_projection_threshold_handles_noise_spike_and_faint_rows():
    preprocessor = NotebookPreprocessor()
    mask = np.zeros((72, 180), dtype=np.uint8)
    mask[28:34, 20:160] = 90
    mask[32, 88] = 255

    projection, smooth, threshold = preprocessor.compute_projection_profile(mask)

    assert projection.max() > 0
    assert threshold > 0
    active_rows = np.where(smooth > threshold)[0]
    assert active_rows.size > 0
    assert active_rows.min() <= 30
    assert active_rows.max() >= 31


def test_clean_printed_crop_does_not_trigger_notebook_mode():
    preprocessor = NotebookPreprocessor()
    artifacts = preprocessor.build_mask(_make_clean_printed_crop())

    assert artifacts.notebook_mode is False
    assert artifacts.ruled_line_mode is False
