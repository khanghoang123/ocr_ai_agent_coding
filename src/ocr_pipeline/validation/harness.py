from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from ocr_pipeline.cropper.line_cropper import CropResult, LineCropper
from ocr_pipeline.exporters import export_result
from ocr_pipeline.layout.reconstructor import LayoutReconstructor
from ocr_pipeline.schemas import BoundingBox, ExportFormat, OCRResult, PageResult, TextLine


@dataclass
class ValidationResult:
    name: str
    passed: bool
    details: str
    metrics: dict[str, float] | None = None


def _polygon(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _crop(poly: np.ndarray, idx: int) -> CropResult:
    return CropResult(
        image=Image.new("RGB", (200, 32), color=(255, 255, 255)),
        polygon=poly,
        confidence=0.9,
        line_index=idx,
    )


def _sample_result(filename: str = "sample.jpg") -> OCRResult:
    line = TextLine(
        line_index=0,
        text="xin chao viet nam",
        confidence=0.95,
        bbox=BoundingBox(x1=10, y1=10, x2=150, y2=40),
    )
    page = PageResult(page_number=1, width=200, height=100, lines=[line])
    return OCRResult(
        filename=filename,
        file_type="image",
        model_used="experiment_B",
        total_pages=1,
        pages=[page],
        processing_time_ms=12.0,
    )


def validate_layout_correctness() -> ValidationResult:
    reconstructor = LayoutReconstructor(row_tolerance_ratio=0.5)
    polygons = [
        _polygon(400, 50, 780, 80),   # row1-right
        _polygon(10, 50, 380, 80),    # row1-left
        _polygon(10, 150, 380, 180),  # row2-left
        _polygon(400, 150, 780, 180), # row2-right
    ]
    texts = ["r1-right", "r1-left", "r2-left", "r2-right"]
    crops = [_crop(poly, idx) for idx, poly in enumerate(polygons)]
    page = reconstructor.reconstruct(crops, texts, image_width=800, image_height=1200, page_number=1)
    actual = [line.text for line in page.lines]
    expected = ["r1-left", "r1-right", "r2-left", "r2-right"]
    passed = actual == expected
    details = f"expected={expected}, actual={actual}"
    return ValidationResult(name="layout_correctness", passed=passed, details=details)


def validate_edge_cases() -> ValidationResult:
    cropper = LineCropper(min_height=8, min_width=20, padding=4)
    image = Image.new("RGB", (400, 200), color=(255, 255, 255))

    tiny = _polygon(10, 50, 50, 53)
    tiny_crops = cropper.crop_all(image, [tiny], [0.9])
    tiny_ok = len(tiny_crops) == 0

    skew = np.array([[20, 60], [320, 50], [330, 90], [10, 100]], dtype=np.float32)
    skew_crops = cropper.crop_all(image, [skew], [0.9])
    skew_ok = len(skew_crops) == 1 and skew_crops[0].image.width > skew_crops[0].image.height

    out_of_bounds = _polygon(-10, -5, 600, 40)
    try:
        cropper.crop_all(image, [out_of_bounds], [0.9])
        clamp_ok = True
    except Exception:
        clamp_ok = False

    passed = tiny_ok and skew_ok and clamp_ok
    details = f"tiny_ok={tiny_ok}, skew_ok={skew_ok}, clamp_ok={clamp_ok}"
    return ValidationResult(name="edge_cases", passed=passed, details=details)


def validate_output_format() -> ValidationResult:
    result = _sample_result()
    txt_bytes, txt_mime = export_result(result, ExportFormat.TXT)
    json_bytes, json_mime = export_result(result, ExportFormat.JSON)
    txt_content = txt_bytes.decode("utf-8")
    payload = json.loads(json_bytes.decode("utf-8"))

    txt_ok = "[Page 1]" in txt_content and "xin chao viet nam" in txt_content
    json_ok = payload["pages"][0]["lines"][0]["bbox"]["x2"] > payload["pages"][0]["lines"][0]["bbox"]["x1"]
    mime_ok = "text/plain" in txt_mime and "application/json" in json_mime
    passed = txt_ok and json_ok and mime_ok
    details = f"txt_ok={txt_ok}, json_ok={json_ok}, mime_ok={mime_ok}"
    return ValidationResult(name="output_format", passed=passed, details=details)


def validate_performance(max_reconstruct_ms: float = 100.0) -> ValidationResult:
    reconstructor = LayoutReconstructor(row_tolerance_ratio=0.5)
    n = 100
    polygons = [_polygon(10, i * 35, 400, i * 35 + 30) for i in range(n)]
    crops = [_crop(poly, idx) for idx, poly in enumerate(polygons)]
    texts = [f"line-{i}" for i in range(n)]

    t0 = time.perf_counter()
    page = reconstructor.reconstruct(crops, texts, image_width=800, image_height=n * 36, page_number=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    passed = len(page.lines) == n and elapsed_ms <= max_reconstruct_ms
    details = f"lines={len(page.lines)}, elapsed_ms={elapsed_ms:.2f}, budget_ms={max_reconstruct_ms:.2f}"
    return ValidationResult(
        name="performance",
        passed=passed,
        details=details,
        metrics={"elapsed_ms": elapsed_ms, "budget_ms": max_reconstruct_ms},
    )


def run_all_validations() -> dict:
    results = [
        validate_layout_correctness(),
        validate_edge_cases(),
        validate_output_format(),
        validate_performance(),
    ]
    passed = all(r.passed for r in results)
    return {
        "passed": passed,
        "results": [asdict(r) for r in results],
    }
