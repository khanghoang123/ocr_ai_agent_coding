"""
Comprehensive test suite for the OCR pipeline.

Covers:
  - Layout validation (reading order, left-to-right)
  - Edge cases (noise, skewed text, long/short lines)
  - Performance benchmarks (latency per image)
  - Output validation (bbox alignment, JSON/TXT formats)
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest
from PIL import Image

from ocr_pipeline.cropper.line_cropper import CropResult, LineCropper
from ocr_pipeline.exporters import export_result
from ocr_pipeline.layout.reconstructor import LayoutReconstructor, polygon_to_bbox
from ocr_pipeline.schemas import (
    BoundingBox,
    ExportFormat,
    OCRResult,
    PageResult,
    TextLine,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

def make_polygon(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Create a rectangular 4-point polygon array."""
    return np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ], dtype=np.float32)


def make_crop(
    polygon: np.ndarray,
    confidence: float = 0.9,
    line_index: int = 0,
    width: int = 200,
    height: int = 32,
) -> CropResult:
    """Create a CropResult with a blank image of given size."""
    return CropResult(
        image=Image.new("RGB", (width, height), color=(255, 255, 255)),
        polygon=polygon,
        confidence=confidence,
        line_index=line_index,
    )


def make_ocr_result(
    filename: str = "test.jpg",
    lines: list[tuple[str, float, float, float, float]] | None = None,
) -> OCRResult:
    """Create a minimal OCRResult for export tests."""
    if lines is None:
        lines = [("Xin chào", 0, 0, 100, 30), ("Việt Nam", 0, 40, 100, 70)]
    text_lines = [
        TextLine(
            line_index=i,
            text=line[0],
            confidence=0.9,
            bbox=BoundingBox(x1=line[1], y1=line[2], x2=line[3], y2=line[4]),
        )
        for i, line in enumerate(lines)
    ]
    page = PageResult(page_number=1, width=200, height=300, lines=text_lines)
    return OCRResult(
        filename=filename,
        file_type="image",
        model_used="experiment_B",
        total_pages=1,
        pages=[page],
        processing_time_ms=42.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layout Validation Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLayoutReconstruction:
    """Validate that reconstructed text preserves correct reading order."""

    def setup_method(self):
        self.reconstructor = LayoutReconstructor(row_tolerance_ratio=0.5)

    def _build_page(
        self,
        polygons: list[np.ndarray],
        texts: list[str],
    ) -> PageResult:
        crops = [make_crop(p, line_index=i) for i, p in enumerate(polygons)]
        return self.reconstructor.reconstruct(crops, texts, 800, 1200, 1)

    # ── Top-to-bottom ordering ────────────────────────────────────────────────

    def test_top_to_bottom_order_simple(self):
        """Lines should be sorted top-to-bottom."""
        # Line 3 given first, line 1 given last
        polygons = [
            make_polygon(10, 200, 400, 230),   # bottom
            make_polygon(10,  50, 400,  80),   # top
            make_polygon(10, 120, 400, 150),   # middle
        ]
        texts = ["bottom line", "top line", "middle line"]
        page = self._build_page(polygons, texts)

        ordered = [tl.text for tl in page.lines]
        assert ordered == ["top line", "middle line", "bottom line"], (
            f"Expected top→middle→bottom, got: {ordered}"
        )

    def test_left_to_right_within_row(self):
        """Lines on the same row should be sorted left-to-right."""
        # Two columns on the same row (y close)
        polygons = [
            make_polygon(400, 100, 780, 130),   # right column (given first)
            make_polygon( 10, 100, 380, 130),   # left column
        ]
        texts = ["right column", "left column"]
        page = self._build_page(polygons, texts)

        ordered = [tl.text for tl in page.lines]
        assert ordered[0] == "left column", (
            f"Expected left column first, got: {ordered}"
        )
        assert ordered[1] == "right column"

    def test_multi_row_multi_column(self):
        """3-row document with 2 columns per row."""
        # Row 1: y≈50; Row 2: y≈150; Row 3: y≈250
        polygons = [
            make_polygon(400,  50, 780,  80),  # row1-right
            make_polygon( 10,  50, 380,  80),  # row1-left
            make_polygon( 10, 150, 380, 180),  # row2-left
            make_polygon(400, 250, 780, 280),  # row3-right
            make_polygon( 10, 250, 380, 280),  # row3-left
            make_polygon(400, 150, 780, 180),  # row2-right
        ]
        texts = ["r1-right", "r1-left", "r2-left", "r3-right", "r3-left", "r2-right"]
        page = self._build_page(polygons, texts)

        expected = ["r1-left", "r1-right", "r2-left", "r2-right", "r3-left", "r3-right"]
        actual = [tl.text for tl in page.lines]
        assert actual == expected, f"Expected {expected}, got {actual}"

    def test_line_indices_are_sequential(self):
        """line_index must be sequential (0, 1, 2, ...) after sorting."""
        polygons = [make_polygon(0, i * 40, 200, i * 40 + 30) for i in range(5)]
        texts = [f"line{i}" for i in range(5)]
        page = self._build_page(polygons, texts)
        indices = [tl.line_index for tl in page.lines]
        assert indices == list(range(len(page.lines))), (
            f"Non-sequential indices: {indices}"
        )

    def test_single_line_document(self):
        """Single line document should work without errors."""
        polygons = [make_polygon(10, 10, 400, 40)]
        texts = ["Chỉ có một dòng"]
        page = self._build_page(polygons, texts)
        assert len(page.lines) == 1
        assert page.lines[0].text == "Chỉ có một dòng"

    def test_empty_page(self):
        """Empty detection result should return empty PageResult."""
        page = self.reconstructor.reconstruct([], [], 800, 1200, 1)
        assert len(page.lines) == 0
        assert page.page_number == 1


# ══════════════════════════════════════════════════════════════════════════════
# Edge Case Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Handle handwritten text with noise, skew, and extreme sizes."""

    def setup_method(self):
        self.cropper = LineCropper(min_height=8, min_width=20, padding=4)
        self.reconstructor = LayoutReconstructor(row_tolerance_ratio=0.5)

    # ── Noisy detections ──────────────────────────────────────────────────────

    def test_very_short_line_filtered(self):
        """Lines below min_height or min_width should be dropped silently."""
        # Create a tiny "noise" polygon (3px height)
        tiny_polygon = make_polygon(10, 50, 50, 53)   # height=3 < min_height=8
        image = Image.new("RGB", (400, 200), "white")
        crops = self.cropper.crop_all(image, [tiny_polygon], [0.9])
        assert crops == [], "Tiny lines should be filtered out"

    def test_min_width_filter(self):
        """Lines narrower than min_width should be dropped."""
        narrow_polygon = make_polygon(10, 50, 15, 80)   # width=5 < min_width=20
        image = Image.new("RGB", (400, 200), "white")
        crops = self.cropper.crop_all(image, [narrow_polygon], [0.9])
        assert crops == [], "Narrow lines should be filtered out"

    def test_valid_line_passes_filter(self):
        """A normal-sized line should pass the filter."""
        valid_polygon = make_polygon(10, 50, 300, 82)   # 32px height, 290px width
        image = Image.new("RGB", (400, 200), "white")
        crops = self.cropper.crop_all(image, [valid_polygon], [0.9])
        assert len(crops) == 1, "Valid line should pass the filter"

    def test_vertical_crop_auto_rotates(self):
        """Vertical crops should be rotated to a horizontal reading orientation."""
        vertical_polygon = make_polygon(100, 20, 130, 150)
        image = Image.new("RGB", (240, 200), "white")
        crops = self.cropper.crop_all(image, [vertical_polygon], [0.9])
        assert len(crops) == 1
        crop_img = crops[0].image
        assert crop_img.width >= crop_img.height

    # ── Skewed and rotated text ────────────────────────────────────────────────

    def test_skewed_polygon_perspective_transform(self):
        """Skewed polygon should be corrected via perspective transform."""
        # Simulate slightly rotated text (trapezoid shape)
        skewed_polygon = np.array([
            [20,  60],   # top-left (shifted right)
            [320, 50],   # top-right
            [330, 90],   # bottom-right
            [ 10, 100],  # bottom-left
        ], dtype=np.float32)
        image = Image.new("RGB", (400, 200), "white")
        crops = self.cropper.crop_all(image, [skewed_polygon], [0.9])
        assert len(crops) == 1
        # Output should be a rectangle (width >> height)
        crop_img = crops[0].image
        assert crop_img.width > crop_img.height, (
            f"Expected width > height for horizontal text, got {crop_img.size}"
        )

    # ── Very long text lines ───────────────────────────────────────────────────

    def test_very_long_line(self):
        """Lines wider than image_max_width (690) should still be cropped."""
        # 800px wide line — wider than model's max_width
        wide_polygon = make_polygon(10, 50, 810, 82)
        image = Image.new("RGB", (1200, 200), "white")
        crops = self.cropper.crop_all(image, [wide_polygon], [0.9])
        assert len(crops) == 1, "Very long lines should still be cropped"
        # VietOCR handles rescaling internally; crop should preserve full width
        assert crops[0].image.width > 500

    # ── Multiple noisy + valid lines ──────────────────────────────────────────

    def test_mixed_valid_and_noisy_lines(self):
        """Only valid lines should survive filtering."""
        polygons = [
            make_polygon(10, 50, 300, 82),   # valid
            make_polygon(10, 100, 15, 103),  # too small
            make_polygon(10, 150, 400, 182), # valid
            make_polygon(5, 200, 8, 210),    # too narrow
        ]
        image = Image.new("RGB", (500, 300), "white")
        crops = self.cropper.crop_all(image, polygons, [0.9] * 4)
        assert len(crops) == 2, f"Expected 2 valid crops, got {len(crops)}"

    # ── Polygon boundary clamping ─────────────────────────────────────────────

    def test_polygon_outside_image_clamped(self):
        """Polygons partially outside image bounds should be clamped, not error."""
        out_of_bounds = make_polygon(-10, -5, 600, 40)  # extends beyond 400px width
        image = Image.new("RGB", (400, 200), "white")
        # Should not raise any exception
        try:
            self.cropper.crop_all(image, [out_of_bounds], [0.9])
        except Exception as e:
            pytest.fail(f"Out-of-bounds polygon raised exception: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Performance Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Measure inference latency and batch processing efficiency."""

    def test_reconstructor_is_fast(self):
        """Layout reconstruction for 100 lines should complete in <100ms."""
        reconstructor = LayoutReconstructor()
        n = 100

        polygons = [
            make_polygon(10, i * 35, 400, i * 35 + 30)
            for i in range(n)
        ]
        crops = [make_crop(p, line_index=i) for i, p in enumerate(polygons)]
        texts = [f"line {i}" for i in range(n)]

        t0 = time.perf_counter()
        page = reconstructor.reconstruct(crops, texts, 800, n * 36, 1)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert len(page.lines) == n
        assert elapsed_ms < 100, (
            f"Layout reconstruction for {n} lines took {elapsed_ms:.1f}ms (>100ms)"
        )

    def test_cropper_handles_large_batch(self):
        """Cropper should process 50 polygons without performance issues."""
        cropper = LineCropper()
        image = Image.new("RGB", (800, 2000), "white")
        polygons = [
            make_polygon(10, i * 38, 600, i * 38 + 32)
            for i in range(50)
        ]

        t0 = time.perf_counter()
        crops = cropper.crop_all(image, polygons, [0.9] * 50)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert len(crops) == 50
        assert elapsed_ms < 5000, (
            f"Cropping 50 lines took {elapsed_ms:.1f}ms (>5000ms)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Output Validation Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputValidation:
    """Validate exported formats and bbox alignment."""

    def setup_method(self):
        self.ocr_result = make_ocr_result(
            filename="test.jpg",
            lines=[
                ("Xin chào Việt Nam", 10.0, 20.0, 500.0, 52.0),
                ("Đây là dòng thứ hai", 10.0, 60.0, 600.0, 92.0),
                ("Kết thúc văn bản", 10.0, 100.0, 450.0, 132.0),
            ],
        )

    # ── TXT export ────────────────────────────────────────────────────────────

    def test_txt_export_contains_all_lines(self):
        """TXT export should contain all recognized text lines."""
        content_bytes, mime = export_result(self.ocr_result, ExportFormat.TXT)
        content = content_bytes.decode("utf-8")

        assert "Xin chào Việt Nam" in content
        assert "Đây là dòng thứ hai" in content
        assert "Kết thúc văn bản" in content

    def test_txt_export_mime_type(self):
        _, mime = export_result(self.ocr_result, ExportFormat.TXT)
        assert "text/plain" in mime

    def test_txt_export_is_utf8(self):
        """TXT output must be valid UTF-8 (Vietnamese diacritics)."""
        content_bytes, _ = export_result(self.ocr_result, ExportFormat.TXT)
        # Should not raise
        content = content_bytes.decode("utf-8")
        assert "Việt Nam" in content

    def test_txt_export_preserves_line_order(self):
        """Lines in TXT must appear in correct reading order."""
        content_bytes, _ = export_result(self.ocr_result, ExportFormat.TXT)
        content = content_bytes.decode("utf-8")
        pos_1 = content.index("Xin chào")
        pos_2 = content.index("Đây là")
        pos_3 = content.index("Kết thúc")
        assert pos_1 < pos_2 < pos_3, "Lines must appear in reading order in TXT"

    # ── JSON export ───────────────────────────────────────────────────────────

    def test_json_export_is_valid_json(self):
        """JSON export must be parseable."""
        content_bytes, _ = export_result(self.ocr_result, ExportFormat.JSON)
        data = json.loads(content_bytes.decode("utf-8"))
        assert isinstance(data, dict)

    def test_json_export_structure(self):
        """JSON must have expected top-level keys."""
        content_bytes, _ = export_result(self.ocr_result, ExportFormat.JSON)
        data = json.loads(content_bytes.decode("utf-8"))
        assert "filename" in data
        assert "pages" in data
        assert "total_lines" in data
        assert data["filename"] == "test.jpg"

    def test_json_bbox_values_are_finite(self):
        """All bbox values in JSON must be finite numbers (no NaN/inf)."""
        content_bytes, _ = export_result(self.ocr_result, ExportFormat.JSON)
        data = json.loads(content_bytes.decode("utf-8"))
        for page in data["pages"]:
            for line in page["lines"]:
                bbox = line["bbox"]
                for key, val in bbox.items():
                    assert math.isfinite(val), (
                        f"bbox.{key}={val} is not finite for line '{line['text']}'"
                    )

    def test_json_bbox_alignment(self):
        """bbox x2 > x1 and y2 > y1 for all lines."""
        content_bytes, _ = export_result(self.ocr_result, ExportFormat.JSON)
        data = json.loads(content_bytes.decode("utf-8"))
        for page in data["pages"]:
            for line in page["lines"]:
                bb = line["bbox"]
                assert bb["x2"] > bb["x1"], (
                    f"x2 ({bb['x2']}) must be > x1 ({bb['x1']})"
                )
                assert bb["y2"] > bb["y1"], (
                    f"y2 ({bb['y2']}) must be > y1 ({bb['y1']})"
                )

    def test_json_confidence_in_range(self):
        """Confidence scores must be in [0, 1]."""
        content_bytes, _ = export_result(self.ocr_result, ExportFormat.JSON)
        data = json.loads(content_bytes.decode("utf-8"))
        for page in data["pages"]:
            for line in page["lines"]:
                conf = line["confidence"]
                assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of [0,1]"

    def test_json_mime_type(self):
        _, mime = export_result(self.ocr_result, ExportFormat.JSON)
        assert "application/json" in mime

    # ── BoundingBox schema ────────────────────────────────────────────────────

    def test_polygon_to_bbox_conversion(self):
        """polygon_to_bbox must return axis-aligned bbox from polygon points."""
        polygon = np.array([
            [10, 20], [300, 15], [305, 55], [5, 60]
        ], dtype=np.float32)
        bbox = polygon_to_bbox(polygon)
        assert bbox.x1 == pytest.approx(5.0)
        assert bbox.y1 == pytest.approx(15.0)
        assert bbox.x2 == pytest.approx(305.0)
        assert bbox.y2 == pytest.approx(60.0)
        assert bbox.width == pytest.approx(300.0)
        assert bbox.height == pytest.approx(45.0)

    def test_multi_page_json_export(self):
        """JSON export for multi-page result should include all pages."""
        pages = [
            PageResult(
                page_number=i,
                width=800,
                height=1200,
                lines=[TextLine(
                    line_index=0,
                    text=f"Page {i} text",
                    confidence=0.9,
                    bbox=BoundingBox(x1=10, y1=10, x2=200, y2=40),
                )],
            )
            for i in range(1, 4)
        ]
        result = OCRResult(
            filename="doc.pdf",
            file_type="pdf",
            model_used="experiment_B",
            total_pages=3,
            pages=pages,
            processing_time_ms=100.0,
        )
        content_bytes, _ = export_result(result, ExportFormat.JSON)
        data = json.loads(content_bytes.decode("utf-8"))
        assert data["total_pages"] == 3
        assert len(data["pages"]) == 3

    # ── PDF export (white background, black text) ────────────────────────────

    def test_pdf_export_returns_pdf_bytes(self):
        """PDF export must return a non-empty bytes payload starting with %PDF."""
        pytest.importorskip("reportlab")
        content_bytes, mime = export_result(self.ocr_result, ExportFormat.PDF)
        assert mime == "application/pdf"
        assert content_bytes.startswith(b"%PDF"), "missing PDF magic bytes"
        # 1-page PDF with text only is small but never empty.
        assert len(content_bytes) > 500

    def test_pdf_page_size_is_a4(self):
        """PDF page size must be A4 (595x842 pt) for readability."""
        pytest.importorskip("reportlab")
        fitz = pytest.importorskip("fitz")
        content_bytes, _ = export_result(self.ocr_result, ExportFormat.PDF)
        doc = fitz.open(stream=content_bytes, filetype="pdf")
        page = doc[0]
        assert int(round(page.rect.width)) == 595
        assert int(round(page.rect.height)) == 842
        doc.close()

    def test_pdf_does_not_embed_source_image(self):
        """PDF export must NOT include the original image as background.

        The user-facing PDF should be a clean transcription on white
        paper. Embedding the source image defeats the structural-OCR
        purpose of the export and bloats the payload.
        """
        pytest.importorskip("reportlab")
        fitz = pytest.importorskip("fitz")
        content_bytes, _ = export_result(
            self.ocr_result, ExportFormat.PDF, images=[Image.new("RGB", (1200, 1600), "red")]
        )
        doc = fitz.open(stream=content_bytes, filetype="pdf")
        # No images on any page.
        for page in doc:
            assert page.get_images(full=True) == [], (
                "PDF export must not embed the source image; got "
                f"{page.get_images(full=True)} on page {page.number + 1}"
            )
        doc.close()

    def test_pdf_renders_recognized_text(self):
        """All recognised line texts must appear in the PDF text layer."""
        pytest.importorskip("reportlab")
        fitz = pytest.importorskip("fitz")
        # Build a page large enough that the line bboxes (which go up
        # to x=600 / y=132) sit entirely inside the page rect; the
        # shared ``self.ocr_result`` fixture uses 200×300 which clips
        # everything off-page and would mask any rendering errors.
        result = OCRResult(
            filename="test.jpg",
            file_type="image",
            model_used="experiment_B",
            total_pages=1,
            processing_time_ms=10.0,
            pages=[PageResult(
                page_number=1,
                width=1200,
                height=1600,
                lines=list(self.ocr_result.pages[0].lines),
            )],
        )
        content_bytes, _ = export_result(result, ExportFormat.PDF)
        doc = fitz.open(stream=content_bytes, filetype="pdf")
        page_text = doc[0].get_text("text")
        for line in result.pages[0].lines:
            assert line.text in page_text, (
                f"recognised text {line.text!r} not present in the PDF"
            )
        doc.close()
