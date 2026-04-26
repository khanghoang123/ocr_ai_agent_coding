from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ocr_pipeline.validation.debug_analyzer import analyze_debug_dir, write_debug_analysis


def _make_crop(path: Path, size: tuple[int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return str(path)


def test_debug_analyzer_computes_geometry_rates(tmp_path: Path):
    debug_dir = tmp_path / "debug"
    crop_dir = debug_dir / "crops"
    mapping = [
        {
            "image": "sample.jpg",
            "id": "sample",
            "lines": [
                {
                    "line_index": 0,
                    "crop_path": _make_crop(crop_dir / "normal.jpg", (200, 40)),
                    "raw_text": "xin chao",
                    "cleaned_text": "xin chao",
                    "bbox": {"x1": 0, "y1": 0, "x2": 200, "y2": 30},
                },
                {
                    "line_index": 1,
                    "crop_path": _make_crop(crop_dir / "tiny.jpg", (30, 20)),
                    "raw_text": "x",
                    "cleaned_text": "x",
                    "bbox": {"x1": 0, "y1": 50, "x2": 20, "y2": 58},
                },
                {
                    "line_index": 2,
                    "crop_path": _make_crop(crop_dir / "merged.jpg", (220, 70)),
                    "raw_text": "merged line",
                    "cleaned_text": "merged line",
                    "bbox": {"x1": 0, "y1": 100, "x2": 220, "y2": 170},
                },
                {
                    "line_index": 3,
                    "crop_path": _make_crop(crop_dir / "digit.jpg", (200, 40)),
                    "raw_text": "V1et N4m",
                    "cleaned_text": "V1et N4m",
                    "bbox": {"x1": 0, "y1": 200, "x2": 200, "y2": 230},
                },
            ],
        }
    ]
    (debug_dir / "debug_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")

    summary = analyze_debug_dir(debug_dir)
    aggregate = summary["aggregate"]

    assert aggregate["line_count"] == 4
    assert aggregate["tiny_box_count"] == 1
    assert aggregate["merged_box_count"] == 1
    assert aggregate["abnormal_crop_count"] == 1
    assert aggregate["high_digit_noise_line_count"] == 1
    assert aggregate["tiny_box_rate"] == 0.25
    assert aggregate["merged_box_rate"] == 0.25
    assert aggregate["abnormal_crop_ratio"] == 0.25
    assert aggregate["high_digit_noise_line_rate"] == 0.25
    assert round(aggregate["average_crop_aspect_ratio"], 4) == 3.6607


def test_write_debug_analysis_creates_output(tmp_path: Path):
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    (debug_dir / "debug_mapping.json").write_text("[]", encoding="utf-8")
    output = tmp_path / "analysis.json"

    write_debug_analysis(debug_dir, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["aggregate"]["line_count"] == 0
