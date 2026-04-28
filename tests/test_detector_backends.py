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


def test_paddle_diagnose_detection_always_carries_backend_keys():
    """The diagnostics dict must always include ``backend`` and
    ``low_detector_recall`` regardless of which Paddle return path produces
    it. This is the contract every downstream consumer relies on
    (``DebugPageResult``, ``run_detector_experiment.py``)."""
    det = PaddleDetector(use_gpu=False)

    # Empty path.
    diag_empty = det._diagnose_detection([], image_height=800)
    assert diag_empty["backend"] == "paddle"
    assert diag_empty["low_detector_recall"] is True

    # Non-empty path.
    polys = [
        np.array([[10, 10], [200, 10], [200, 30], [10, 30]], dtype=float),
        np.array([[10, 50], [180, 50], [180, 70], [10, 70]], dtype=float),
    ]
    diag_full = det._diagnose_detection(polys, image_height=800)
    assert diag_full["backend"] == "paddle"
    assert diag_full["low_detector_recall"] is False
    assert "median_height" in diag_full
    assert "estimated_rows" in diag_full
    assert "coverage_ratio" in diag_full


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


def test_cluster_words_into_lines_no_snowballing_across_overlapping_rows():
    """Phase 3 retune: words on stacked rows whose vertical extents partially
    overlap (because a descender from row N reaches into row N+1) must NOT
    collapse into a single band. The old extent-based clustering snowballed
    here; the new y-centroid clustering should keep them separate.
    """
    rows: list[np.ndarray] = []
    # Row 1 at y_center=100, height=30 (extent 85..115).
    for x in (10, 100, 200, 300):
        rows.append(
            np.array([[x, 85], [x + 60, 85], [x + 60, 115], [x, 115]], dtype=float)
        )
    # Row 2 at y_center=130, height=30 (extent 115..145). Row 1 and Row 2
    # vertical extents touch (115); a descender word would spill across.
    for x in (10, 100, 200, 300):
        rows.append(
            np.array([[x, 115], [x + 60, 115], [x + 60, 145], [x, 145]], dtype=float)
        )
    # Row 3 at y_center=160, height=30 (extent 145..175).
    for x in (10, 100, 200, 300):
        rows.append(
            np.array([[x, 145], [x + 60, 145], [x + 60, 175], [x, 175]], dtype=float)
        )
    lines = cluster_words_into_lines(rows, image_width=400, image_height=400)
    # Three distinct rows must remain three lines, not a single fat band.
    assert len(lines) == 3


def test_cluster_words_into_lines_drops_full_width_bands():
    """The max_line_width_ratio cap should reject misclustered rows whose
    horizontal extent spans almost the entire page width \u2014 those are the
    ``soan-bai\u2026`` style paragraph bands the old algorithm produced.
    """
    # Build one legitimate narrow row, and one runaway cluster that spans
    # 95%+ of the page even though the words are at very different y\u2010centers.
    legit = [
        np.array([[10, 100], [80, 100], [80, 130], [10, 130]], dtype=float),
        np.array([[100, 100], [180, 100], [180, 130], [100, 130]], dtype=float),
    ]
    # Same y\u2010center band, but spanning x=10..980 \u2014 simulates a misclustered
    # paragraph block. Both words at y_center=300.
    runaway = [
        np.array([[10, 290], [80, 290], [80, 310], [10, 310]], dtype=float),
        np.array([[900, 290], [980, 290], [980, 310], [900, 310]], dtype=float),
    ]
    out = cluster_words_into_lines(
        legit + runaway,
        image_width=1000,
        image_height=1000,
        max_line_width_ratio=0.5,
    )
    # The runaway row must be dropped; only the legit line survives.
    assert len(out) == 1
    poly, _ = out[0]
    assert poly[:, 0].max() < 200


def test_cluster_words_into_lines_drops_band_via_height_cap():
    """The ``max_line_height_word_ratio`` cap rejects a row whose merged
    vertical extent vastly exceeds a typical word-height \u2014 the signature
    of a paragraph block masquerading as a single line. Width caps don't
    work for printed textbook lines because real lines naturally span the
    whole page; height caps do.
    """
    # One legit line: 6 words at y_center=100, height=20.
    legit = [
        np.array([[x, 90], [x + 25, 90], [x + 25, 110], [x, 110]], dtype=float)
        for x in range(10, 180, 30)
    ]
    # Stacked over-tall \"band\" that the clustering will join because we
    # set tolerance high enough: words at y=200, 240, 280, 320, 360 (5
    # rows). With tolerance=10*median_h and median_h=20, all of these
    # cluster into one row whose height = 360+20 - 200 = 180 (= 9x word_h).
    band: list[np.ndarray] = []
    for dy in range(200, 380, 40):
        for x in range(10, 180, 30):
            band.append(
                np.array(
                    [[x, dy], [x + 25, dy], [x + 25, dy + 20], [x, dy + 20]],
                    dtype=float,
                )
            )
    out = cluster_words_into_lines(
        legit + band,
        image_width=1000,
        image_height=1000,
        # 4*median_h = 80px tolerance — large enough to merge band rows
        # 40px apart, small enough to keep legit (y=100) apart from band
        # (y=210+).
        row_y_center_tolerance=4.0,
        max_line_height_word_ratio=2.5,
    )
    # The over-tall band is dropped; only the legit line (height ~22 after
    # padding) survives.
    assert len(out) >= 1
    for poly, _ in out:
        height = float(poly[:, 1].max() - poly[:, 1].min())
        # Surviving lines must be near-word-height; bands are dropped.
        assert height < 60


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
