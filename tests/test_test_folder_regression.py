from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from ocr_pipeline.config import settings
from ocr_pipeline.cropper.line_cropper import LineCropper
from ocr_pipeline.detector.paddle_detector import PaddleDetector
from ocr_pipeline.preprocess.notebook_preprocessor import NotebookPreprocessor
from ocr_pipeline.recognizer.vietocr_recognizer import VietOCRRecognizer
from ocr_pipeline.refiner.line_refiner import LineRefiner


ROOT = Path(__file__).resolve().parents[1]
TEST_FOLDER = ROOT / "tests" / "test"
OUTPUT_ROOT = ROOT / "tests" / "output" / "test_folder_regression"


def _runtime_ready() -> tuple[bool, str]:
    try:
        import paddleocr  # noqa: F401
        import vietocr  # noqa: F401
    except ImportError as exc:
        return False, f"runtime dependency missing: {exc}"

    try:
        config = settings.get_model_config(settings.get_default_model_key())
    except Exception as exc:
        return False, f"model config unavailable: {exc}"

    if not Path(config["weights_path"]).exists():
        return False, f"weights not found: {config['weights_path']}"

    return True, "ready"


def _real_regression_enabled() -> bool:
    return os.getenv("OCR_REAL_REGRESSION", "").strip() == "1"


def _collect_test_images() -> list[Path]:
    if not TEST_FOLDER.exists():
        return []
    allowed = {".jpg", ".jpeg", ".png"}
    return sorted(
        path
        for path in TEST_FOLDER.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    )


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    return "application/octet-stream"


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "item"


def _render_overlay(
    image: Image.Image,
    raw_polygons,
    refined_polygons,
    tight_bboxes,
) -> Image.Image:
    from PIL import ImageDraw

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for polygon in raw_polygons:
        draw.polygon([tuple(point) for point in polygon.reshape(-1, 2)], outline="#f59e0b", width=3)
    for polygon in refined_polygons:
        points = polygon.reshape(-1, 2)
        draw.rectangle(
            [
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            ],
            outline="#2563eb",
            width=2,
        )
    for bbox in tight_bboxes:
        draw.rectangle(list(bbox), outline="#10b981", width=2)
    return overlay


def _render_crop_gallery(
    crops: list[Image.Image],
    texts: list[str],
    output_path: Path,
    columns: int = 3,
) -> None:
    from math import ceil
    from PIL import ImageDraw, ImageOps

    if not crops:
        return

    thumb_w, thumb_h = 360, 132
    text_h = 44
    rows = ceil(len(crops) / columns)
    canvas = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + text_h)), "white")
    draw = ImageDraw.Draw(canvas)

    for index, crop in enumerate(crops):
        row = index // columns
        col = index % columns
        x = col * thumb_w
        y = row * (thumb_h + text_h)
        thumb = ImageOps.contain(crop.convert("RGB"), (thumb_w - 12, thumb_h - 12))
        canvas.paste(thumb, (x + 6, y + 6))
        prefix = texts[index] if index < len(texts) else ""
        draw.text((x + 8, y + thumb_h), f"#{index} {prefix[:52]}", fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


@dataclass
class _PerImageSummary:
    filename: str
    line_count: int
    fallback_rate: float
    curve_actions: int
    split_actions: int
    mean_blank_margin: float
    texts_preview: list[str]


def _measure_blank_margin(preprocessor: NotebookPreprocessor, crop_rgb) -> float:
    import numpy as np

    artifacts = preprocessor.build_mask(np.array(crop_rgb))
    bounds = preprocessor.compute_tight_bounds(artifacts.text_mask, min_padding=0)
    if bounds is None:
        return 1.0
    return preprocessor.blank_margin_ratio(artifacts.text_mask.shape, bounds)


def _run_folder_regression(images: list[Path]) -> dict:
    detector = PaddleDetector.from_settings()
    cropper = LineCropper.from_settings()
    recognizer = VietOCRRecognizer.from_settings(settings.get_default_model_key())
    refiner = LineRefiner()
    preprocessor = NotebookPreprocessor()

    summaries: list[dict] = []
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        detection = detector.detect_with_notebook_fallback(image)
        refinement = refiner.refine(
            image=image,
            detection=detection,
            cropper=cropper,
            recognizer=recognizer,
        )
        recognized = (
            recognizer.recognize_batch([crop.image for crop in refinement.crops], return_prob=True)
            if refinement.crops
            else []
        )
        texts = [text for text, _ in recognized]

        mean_blank_margin = 1.0
        if refinement.crops:
            mean_blank_margin = float(
                sum(_measure_blank_margin(preprocessor, crop.image) for crop in refinement.crops)
                / len(refinement.crops)
            )

        fallback_count = sum(1 for decision in refinement.decisions if decision.fallback_reason)
        curve_actions = sum(1 for decision in refinement.decisions if decision.action == "curve_rectify")
        split_actions = sum(1 for decision in refinement.decisions if decision.action == "split")

        per_dir = OUTPUT_ROOT / _slugify(image_path.stem)
        per_dir.mkdir(parents=True, exist_ok=True)

        overlay = _render_overlay(
            image=image,
            raw_polygons=detection.polygons,
            refined_polygons=[crop.polygon for crop in refinement.crops],
            tight_bboxes=[d.tight_bbox for d in refinement.decisions if d.tight_bbox],
        )
        overlay.save(per_dir / "overlay.png")
        _render_crop_gallery(
            [crop.image for crop in refinement.crops],
            texts,
            per_dir / "crops.png",
        )

        summary = _PerImageSummary(
            filename=image_path.name,
            line_count=len(refinement.crops),
            fallback_rate=fallback_count / max(len(refinement.decisions), 1),
            curve_actions=curve_actions,
            split_actions=split_actions,
            mean_blank_margin=mean_blank_margin,
            texts_preview=texts[:6],
        )
        (per_dir / "summary.json").write_text(
            json.dumps(summary.__dict__, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summaries.append(summary.__dict__)

    final = {
        "model_key": settings.get_default_model_key(),
        "folder": str(TEST_FOLDER),
        "count": len(images),
        "images": summaries,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return final


def test_test_folder_contains_images():
    images = _collect_test_images()
    assert images, f"No images found under {TEST_FOLDER}"


@pytest.mark.slow
def test_api_upload_all_test_images_and_run_ocr():
    ready, reason = _runtime_ready()
    if not ready:
        pytest.skip(reason)
    if not _real_regression_enabled():
        pytest.skip("Set OCR_REAL_REGRESSION=1 to enable running real OCR on fixture images.")

    images = _collect_test_images()
    assert images

    with TestClient(app) as client:
        files = [
            ("files", (path.name, path.read_bytes(), _guess_mime(path)))
            for path in images
        ]
        resp = client.post(
            "/ocr",
            files=files,
            data={"model": settings.get_default_model_key(), "postprocess": "false"},
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] == "success"
        assert len(payload["results"]) == len(images)


@pytest.mark.slow
def test_generate_overlays_and_crop_galleries_for_test_folder():
    ready, reason = _runtime_ready()
    if not ready:
        pytest.skip(reason)
    if not _real_regression_enabled():
        pytest.skip("Set OCR_REAL_REGRESSION=1 to enable running real OCR on fixture images.")

    images = _collect_test_images()
    assert images

    summary = _run_folder_regression(images)
    assert summary["count"] == len(images)
    assert Path(summary["folder"]).exists()
