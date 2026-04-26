from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import scripts.run_ocr_experiment as runner
from ocr_pipeline.schemas import BoundingBox, OCRResult, PageResult, TextLine


def _ocr_result(filename: str, text: str) -> OCRResult:
    line = TextLine(
        line_index=0,
        text=text,
        confidence=0.9,
        bbox=BoundingBox(x1=1, y1=2, x2=30, y2=12),
    )
    page = PageResult(page_number=1, width=64, height=24, lines=[line])
    return OCRResult(
        filename=filename,
        file_type="image",
        model_used="fake",
        total_pages=1,
        pages=[page],
        processing_time_ms=1.0,
    )


class _FakePipeline:
    def process_bytes_debug(self, data: bytes, filename: str):
        if "bad" in filename:
            raise RuntimeError("synthetic failure")
        return type(
            "DebugItem",
            (),
            {
                "result": _ocr_result(filename, "xin chào"),
                "debug_pages": [],
            },
        )()


def test_experiment_runner_logs_errors_and_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    good = tmp_path / "good.png"
    bad = tmp_path / "bad.png"
    Image.new("RGB", (64, 24), "white").save(good)
    Image.new("RGB", (64, 24), "white").save(bad)
    dataset = tmp_path / "dataset.tsv"
    dataset.write_text(f"{good}\txin chào\n{bad}\tbad text\n", encoding="utf-8")
    output_dir = tmp_path / "run"

    monkeypatch.setattr(runner, "create_pipeline", lambda config: _FakePipeline())

    metrics = runner.run_experiment(
        "configs/baseline.yaml",
        dataset,
        output_dir,
        limit=None,
    )

    assert metrics["processed_count"] == 1
    assert metrics["error_count"] == 1
    assert metrics["cer"] == 0.0
    assert (output_dir / "errors.jsonl").read_text(encoding="utf-8").strip()
    assert (output_dir / "metrics.json").exists()
