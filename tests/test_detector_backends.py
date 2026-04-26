"""Unit tests for the Phase 3 detector backend abstraction.

These tests verify the *contract* every backend must satisfy:
  * the factory ``build_detector`` returns the correct class for each name,
  * the silent OpenCV grid fallback in PaddleDetector is OFF by default,
  * empty detection results carry ``low_detector_recall=True`` and
    ``backend=<name>`` diagnostics,
  * the new detector-level metrics (full_width_band_rate, mean_box_aspect_ratio,
    mean_distinct_x1_per_page) behave correctly on synthetic inputs.

We do NOT load Surya/CRAFT/Kraken weights here — those run in the offline
experiment harness. The unit tests stay fast and CI-portable.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ocr_pipeline.detector import build_detector, empty_detection_result
from ocr_pipeline.detector.craft_detector import (
    CraftDetector,
    cluster_words_into_lines,
)
from ocr_pipeline.detector.kraken_detector import KrakenBllaDetector
from ocr_pipeline.detector.paddle_detector import PaddleDetector
from ocr_pipeline.detector.surya_detector import SuryaDetector
from ocr_pipeline.validation.metrics import (
    detector_metrics,
    full_width_band_rate,
    mean_box_aspect_ratio,
    mean_distinct_x1_per_page,
)


# ── Factory ──────────────────────────────────────────────────────────────────


def test_build_detector_paddle():
    det = build_detector("paddle")
    assert isinstance(det, PaddleDetector)


def test_build_detector_craft():
    det = build_detector("craft")
    assert isinstance(det, CraftDetector)


def test_build_detector_surya():
    det = build_detector("surya")
    assert isinstance(det, SuryaDetector)


def test_build_detector_kraken():
    det = build_detector("kraken")
    assert isinstance(det, KrakenBllaDetector)


def test_build_detector_aliases():
    assert isinstance(build_detector("dbnet"), PaddleDetector)
    assert isinstance(build_detector("kraken_blla"), KrakenBllaDetector)
    assert isinstance(build_detector("blla"), KrakenBllaDetector)


def test_build_detector_unknown_raises():
    with pytest.raises(ValueError, match="Unknown detector backend"):
        build_detector("nonexistent")


# ── Empty detection contract ────────────────────────────────────────────────


def test_empty_detection_result_carries_low_recall_flag():
    img = Image.new("RGB", (100, 50), color="white")
    res = empty_detection_result(img, backend="surya", reason="test")
    assert res.num_boxes == 0
    assert res.diagnostics["backend"] == "surya"
    assert res.diagnostics["low_detector_recall"] is True
    assert res.diagnostics["reason"] == "test"
    # Must NOT contain any fabricated polygons.
    assert res.polygons == []
    assert res.confidences == []


def test_paddle_grid_fallback_off_by_default():
    """The silent OpenCV grid fallback must be OFF in fresh detectors.

    Phase 3 disables the silent fallback that fabricates full-width bands
    when DBNet returns nothing. The audit traced ~30% of hallucinated-line
    output to that fallback. We verify the constructor default explicitly,
    NOT through expensive end-to-end inference.
    """
    det = PaddleDetector(use_gpu=False)
    assert det.allow_grid_fallback is False


def test_paddle_grid_fallback_can_be_opt_in():
    det = PaddleDetector(use_gpu=False, allow_grid_fallback=True)
    assert det.allow_grid_fallback is True


# ── CRAFT row clustering ─────────────────────────────────────────────────────


def test_cluster_words_into_lines_two_rows():
    """Words spread across two y-bands should produce two line polygons."""
    rows = []
    # Row 1 at y ~ 100, three words.
    for x in (10, 100, 200):
        rows.append(np.array([[x, 100], [x + 60, 100], [x + 60, 130], [x, 130]], dtype=float))
    # Row 2 at y ~ 200, two words.
    for x in (20, 180):
        rows.append(np.array([[x, 200], [x + 80, 200], [x + 80, 230], [x, 230]], dtype=float))
    lines = cluster_words_into_lines(rows, image_width=400, image_height=400)
    assert len(lines) == 2
    # First line should span x ~ 10..260, second ~ 20..260.
    poly0, _ = lines[0]
    assert poly0[:, 0].min() < 15
    assert poly0[:, 0].max() > 250


def test_cluster_words_into_lines_empty():
    assert cluster_words_into_lines([]) == []


def test_cluster_words_into_lines_single_word():
    word = [np.array([[10, 50], [60, 50], [60, 80], [10, 80]], dtype=float)]
    lines = cluster_words_into_lines(word, image_width=200, image_height=200)
    assert len(lines) == 1


# ── Detector-level metrics ──────────────────────────────────────────────────


def test_full_width_band_rate_full_band():
    band = [np.array([[0, 0], [600, 0], [600, 30], [0, 30]], dtype=float)]
    assert full_width_band_rate(band, image_width=600) == pytest.approx(1.0)


def test_full_width_band_rate_clean_lines():
    polys = [
        np.array([[10, 10], [200, 10], [200, 30], [10, 30]], dtype=float),
        np.array([[10, 50], [180, 50], [180, 70], [10, 70]], dtype=float),
    ]
    assert full_width_band_rate(polys, image_width=600) == 0.0


def test_full_width_band_rate_empty():
    assert full_width_band_rate([], image_width=600) == 0.0


def test_mean_box_aspect_ratio():
    polys = [
        np.array([[0, 0], [100, 0], [100, 10], [0, 10]], dtype=float),  # ratio 10
        np.array([[0, 0], [200, 0], [200, 10], [0, 10]], dtype=float),  # ratio 20
    ]
    assert mean_box_aspect_ratio(polys) == pytest.approx(15.0)


def test_mean_distinct_x1_per_page():
    polys = [
        np.array([[10, 0], [100, 0], [100, 10], [10, 10]], dtype=float),
        # Same left edge as row above — should NOT add a distinct bin.
        np.array([[10, 50], [120, 50], [120, 60], [10, 60]], dtype=float),
        # Different left edge.
        np.array([[80, 100], [200, 100], [200, 110], [80, 110]], dtype=float),
    ]
    assert mean_distinct_x1_per_page(polys) == 2.0


def test_detector_metrics_bundles_all():
    polys = [
        np.array([[10, 10], [100, 10], [100, 30], [10, 30]], dtype=float),
        np.array([[10, 50], [200, 50], [200, 80], [10, 80]], dtype=float),
    ]
    m = detector_metrics(polys, image_width=600, image_height=400)
    assert m["detection_count"] == 2
    assert m["full_width_band_rate"] == 0.0
    assert m["mean_distinct_x1_per_page"] == 1.0
    assert m["median_box_height"] in (20.0, 25.0, 30.0)


# ── Mock backend for end-to-end empty-result wiring ─────────────────────────


def test_surya_handles_init_failure_gracefully(monkeypatch):
    """If Surya cannot import, the backend reports low_detector_recall instead
    of crashing the run."""
    det = SuryaDetector()

    def _broken_loader():
        raise RuntimeError("simulated import failure")

    monkeypatch.setattr(det, "_ensure_loaded", _broken_loader)
    img = Image.new("RGB", (200, 100), color="white")
    res = det.detect_with_notebook_fallback(img)
    assert res.num_boxes == 0
    assert res.diagnostics["low_detector_recall"] is True
    assert res.diagnostics["backend"] == "surya"


def test_kraken_handles_init_failure_gracefully(monkeypatch):
    det = KrakenBllaDetector()

    def _broken_loader():
        raise RuntimeError("simulated import failure")

    monkeypatch.setattr(det, "_ensure_loaded", _broken_loader)
    img = Image.new("RGB", (200, 100), color="white")
    res = det.detect_with_notebook_fallback(img)
    assert res.num_boxes == 0
    assert res.diagnostics["low_detector_recall"] is True
    assert res.diagnostics["backend"] == "kraken_blla"


def test_craft_handles_init_failure_gracefully(monkeypatch):
    det = CraftDetector(weights_path="/nonexistent/path/to/weights.pth")
    img = Image.new("RGB", (200, 100), color="white")
    res = det.detect_with_notebook_fallback(img)
    assert res.num_boxes == 0
    assert res.diagnostics["low_detector_recall"] is True
    assert res.diagnostics["backend"] == "craft"
