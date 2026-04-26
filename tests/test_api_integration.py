from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as app_main
import app.routes.ocr as ocr_routes
from app.main import app
from ocr_pipeline.schemas import (
    BoundingBox,
    DebugCropPreview,
    DebugDetectionDecision,
    DebugPageResult,
    OCRDebugItem,
    OCRResult,
    PageResult,
    TextLine,
)


def _sample_image_bytes(fmt: str = "JPEG") -> bytes:
    img = Image.new("RGB", (120, 40), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _fake_result(filename: str, model_key: str) -> OCRResult:
    line = TextLine(
        line_index=0,
        text="xin chao",
        confidence=0.95,
        bbox=BoundingBox(x1=10, y1=10, x2=100, y2=30),
    )
    page = PageResult(page_number=1, width=120, height=40, lines=[line])
    return OCRResult(
        filename=filename,
        file_type="image",
        model_used=model_key,
        total_pages=1,
        pages=[page],
        processing_time_ms=12.3,
    )


class _FakePipeline:
    def __init__(self, model_key: str = "experiment_B"):
        self.model_key = model_key

    def process_bytes(self, data: bytes, filename: str) -> OCRResult:
        if not data:
            raise ValueError("empty payload")
        return _fake_result(filename=filename, model_key=self.model_key)

    def process_bytes_debug(self, data: bytes, filename: str) -> OCRDebugItem:
        result = self.process_bytes(data, filename)
        debug_page = DebugPageResult(
            page_number=1,
            raw_polygons=[[[10.0, 10.0], [100.0, 10.0], [100.0, 30.0], [10.0, 30.0]]],
            refined_boxes=[BoundingBox(x1=10, y1=10, x2=100, y2=30)],
            decisions=[
                DebugDetectionDecision(
                    source_index=0,
                    action="kept",
                    note="fake",
                    decision_stage="patch_0",
                    fallback_reason="split_score_lower",
                    raw_polygon=[[10.0, 10.0], [100.0, 10.0], [100.0, 30.0], [10.0, 30.0]],
                    refined_boxes=[BoundingBox(x1=10, y1=10, x2=100, y2=30)],
                    tight_bbox=BoundingBox(x1=12, y1=11, x2=98, y2=29),
                    projection_bbox=BoundingBox(x1=11, y1=10, x2=99, y2=30),
                    clamped_bbox=BoundingBox(x1=12, y1=12, x2=98, y2=28),
                    curve_score=0.08,
                    neighbor_strategy="overlap_or_distance",
                    projection_threshold=12.0,
                    notebook_mode=True,
                    ruled_line_mode=True,
                    mask_preview_base64="",
                    rectified_preview_base64="",
                )
            ],
            final_crops=[
                DebugCropPreview(
                    line_index=0,
                    text="xin chao",
                    detection_confidence=0.95,
                    recognition_score=0.88,
                    bbox=BoundingBox(x1=10, y1=10, x2=100, y2=30),
                    preview_base64="",
                    decode_mode="beam",
                    pre_norm_ratio=2.7,
                    post_norm_ratio=4.0,
                    baseline_confidence=0.86,
                    baseline_offset=2.0,
                    normalization_preview_base64="",
                )
            ],
        )
        return OCRDebugItem(result=result, debug_pages=[debug_page])


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Prevent heavy model loading in FastAPI lifespan startup.
    monkeypatch.setattr(app_main, "init_pipeline", lambda model_key=None: None)
    monkeypatch.setattr(
        ocr_routes,
        "get_pipeline",
        lambda model_key=None: _FakePipeline(model_key or "experiment_B_50k"),
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_expected_payload(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] == "experiment_B_50k"
    assert "gpu_available" in data
    assert "version" in data


def test_models_endpoint_returns_available_models(client: TestClient):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["default_model"] == "experiment_B_50k"
    keys = [item["key"] for item in data["models"]]
    assert keys == ["baseline_10k", "baseline_50k", "experiment_B_50k"]


def test_ocr_rejects_unsupported_file_type(client: TestClient):
    resp = client.post(
        "/ocr",
        files=[("files", ("bad.txt", b"not-an-image", "text/plain"))],
    )
    assert resp.status_code == 415
    assert "Unsupported file type" in resp.text


def test_ocr_success_single_image(client: TestClient):
    resp = client.post(
        "/ocr",
        files=[("files", ("sample.jpg", _sample_image_bytes("JPEG"), "image/jpeg"))],
        data={"model": "baseline_10k"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "success"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["filename"] == "sample.jpg"
    assert payload["results"][0]["model_used"] == "baseline_10k"
    assert payload["results"][0]["pages"][0]["lines"][0]["text"] == "xin chao"


def test_ocr_rejects_invalid_model_key(client: TestClient):
    resp = client.post(
        "/ocr",
        files=[("files", ("sample.jpg", _sample_image_bytes("JPEG"), "image/jpeg"))],
        data={"model": "does_not_exist"},
    )
    assert resp.status_code == 422
    assert "Model 'does_not_exist' not found" in resp.text


def test_ocr_debug_returns_debug_payload(client: TestClient):
    resp = client.post(
        "/ocr/debug",
        files=[("files", ("sample.jpg", _sample_image_bytes("JPEG"), "image/jpeg"))],
        data={"model": "experiment_B_50k"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "success"
    assert payload["results"][0]["model_used"] == "experiment_B_50k"
    decision = payload["debug_results"][0]["debug_pages"][0]["decisions"][0]
    assert decision["action"] == "kept"
    assert decision["decision_stage"] == "patch_0"
    assert decision["fallback_reason"] == "split_score_lower"
    assert decision["tight_bbox"]["x1"] == 12
    assert decision["curve_score"] == 0.08
    assert decision["projection_bbox"]["x1"] == 11
    assert decision["clamped_bbox"]["y1"] == 12
    assert decision["neighbor_strategy"] == "overlap_or_distance"
    crop = payload["debug_results"][0]["debug_pages"][0]["final_crops"][0]
    assert crop["decode_mode"] == "beam"
    assert crop["post_norm_ratio"] == 4.0


def test_ocr_export_single_file_txt(client: TestClient):
    resp = client.post(
        "/ocr/export",
        files=[("files", ("sample.jpg", _sample_image_bytes("JPEG"), "image/jpeg"))],
        data={"fmt": "txt"},
    )
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert resp.headers["content-disposition"].startswith(
        "attachment; filename=\"sample_ocr.txt\""
    )
    assert "[Page 1]" in resp.text
    assert "xin chao" in resp.text


def test_ocr_export_multi_file_zip(client: TestClient):
    resp = client.post(
        "/ocr/export",
        files=[
            ("files", ("a.jpg", _sample_image_bytes("JPEG"), "image/jpeg")),
            ("files", ("b.png", _sample_image_bytes("PNG"), "image/png")),
        ],
        data={"fmt": "json"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/zip")
    assert "attachment; filename=\"ocr_results.zip\"" == resp.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = sorted(zf.namelist())
        assert names == ["a_ocr.json", "b_ocr.json"]
