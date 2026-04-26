from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_pipeline.config import settings
from ocr_pipeline.cropper.line_cropper import LineCropper
from ocr_pipeline.detector.paddle_detector import PaddleDetector
from ocr_pipeline.preprocess.notebook_preprocessor import NotebookPreprocessor
from ocr_pipeline.recognizer.vietocr_recognizer import VietOCRRecognizer
from ocr_pipeline.refiner.line_refiner import LineRefiner


MANIFEST_PATH = ROOT / "tests" / "fixtures" / "portfolio_inference" / "manifest.json"
OUTPUT_ROOT = ROOT / "tests" / "output" / "portfolio_regression"


@dataclass
class PortfolioFixture:
    fixture_id: str
    image_path: Path
    line_min: int
    line_max: int
    curved_lines: bool
    fallback_rate_max: float
    blank_margin_gain_min: float
    sentinel_prefixes: list[str]


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> list[PortfolioFixture]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixtures: list[PortfolioFixture] = []
    for item in payload["fixtures"]:
        fixtures.append(
            PortfolioFixture(
                fixture_id=item["id"],
                image_path=(manifest_path.parent / item["image_path"]).resolve(),
                line_min=int(item["expected_line_count"]["min"]),
                line_max=int(item["expected_line_count"]["max"]),
                curved_lines=bool(item["curved_lines"]),
                fallback_rate_max=float(item["fallback_rate_max"]),
                blank_margin_gain_min=float(item["blank_margin_gain_min"]),
                sentinel_prefixes=list(item.get("sentinel_prefixes", [])),
            )
        )
    return fixtures


def load_runtime_components(model_key: str | None = None):
    effective_model = model_key or settings.get_default_model_key()
    detector = PaddleDetector.from_settings()
    cropper = LineCropper.from_settings()
    recognizer = VietOCRRecognizer.from_settings(effective_model)
    refiner = LineRefiner()
    preprocessor = NotebookPreprocessor()
    return effective_model, detector, cropper, recognizer, refiner, preprocessor


def measure_blank_margin(preprocessor: NotebookPreprocessor, crop_rgb: np.ndarray) -> float:
    artifacts = preprocessor.build_mask(crop_rgb)
    bounds = preprocessor.compute_tight_bounds(artifacts.text_mask, min_padding=0)
    if bounds is None:
        return 1.0
    return preprocessor.blank_margin_ratio(artifacts.text_mask.shape, bounds)


def render_overlay(
    image: Image.Image,
    raw_polygons: list[np.ndarray],
    refined_polygons: list[np.ndarray],
    tight_bboxes: list[tuple[float, float, float, float]],
) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    for polygon in raw_polygons:
        draw.polygon([tuple(point) for point in polygon.reshape(-1, 2)], outline="#f59e0b", width=3)
    for polygon in refined_polygons:
        points = polygon.reshape(-1, 2)
        draw.rectangle(
            [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())],
            outline="#2563eb",
            width=2,
        )
    for bbox in tight_bboxes:
        draw.rectangle(list(bbox), outline="#10b981", width=2)
    return overlay


def render_crop_gallery(
    crops: list[Image.Image],
    texts: list[str],
    output_path: Path,
    columns: int = 3,
) -> None:
    if not crops:
        return

    thumb_w, thumb_h = 360, 132
    text_h = 44
    rows = math.ceil(len(crops) / columns)
    canvas = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + text_h)), "white")
    draw = ImageDraw.Draw(canvas)

    for index, crop in enumerate(crops):
        row = index // columns
        col = index % columns
        x = col * thumb_w
        y = row * (thumb_h + text_h)
        thumb = ImageOps.contain(crop.convert("RGB"), (thumb_w - 12, thumb_h - 12))
        canvas.paste(thumb, (x + 6, y + 6))
        draw.text((x + 8, y + thumb_h), f"#{index} {texts[index][:52]}", fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def evaluate_fixture(
    fixture: PortfolioFixture,
    detector: PaddleDetector,
    cropper: LineCropper,
    recognizer: VietOCRRecognizer,
    refiner: LineRefiner,
    preprocessor: NotebookPreprocessor,
    output_root: Path,
) -> dict:
    image = Image.open(fixture.image_path).convert("RGB")
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

    raw_blank_margins = []
    for polygon in detection.polygons:
        warped = cropper.warp_polygon(image, polygon)
        if warped is None:
            continue
        raw_blank_margins.append(measure_blank_margin(preprocessor, warped.image))

    final_blank_margins = [
        measure_blank_margin(preprocessor, np.array(crop.image))
        for crop in refinement.crops
    ]

    fixture_output_dir = output_root / fixture.fixture_id
    overlay = render_overlay(
        image=image,
        raw_polygons=detection.polygons,
        refined_polygons=[crop.polygon for crop in refinement.crops],
        tight_bboxes=[decision.tight_bbox for decision in refinement.decisions if decision.tight_bbox],
    )
    fixture_output_dir.mkdir(parents=True, exist_ok=True)
    overlay.save(fixture_output_dir / "overlay.png")
    render_crop_gallery(
        [crop.image for crop in refinement.crops],
        [text for text, _ in recognized],
        fixture_output_dir / "crops.png",
    )

    fallback_count = sum(1 for decision in refinement.decisions if decision.fallback_reason)
    curve_actions = sum(1 for decision in refinement.decisions if decision.action == "curve_rectify")
    split_actions = sum(1 for decision in refinement.decisions if decision.action == "split")
    mean_raw_margin = float(mean(raw_blank_margins)) if raw_blank_margins else 1.0
    mean_final_margin = float(mean(final_blank_margins)) if final_blank_margins else 1.0
    texts = [text for text, _ in recognized]

    summary = {
        "fixture_id": fixture.fixture_id,
        "image_path": str(fixture.image_path),
        "line_count": len(refinement.crops),
        "expected_line_count": {"min": fixture.line_min, "max": fixture.line_max},
        "fallback_rate": fallback_count / max(len(refinement.decisions), 1),
        "curve_actions": curve_actions,
        "split_actions": split_actions,
        "patchwise_used": bool(detection.diagnostics.get("patchwise_used", False)),
        "mean_raw_blank_margin": mean_raw_margin,
        "mean_final_blank_margin": mean_final_margin,
        "texts_preview": texts[:6],
        "decode_modes": sorted(
            {
                item.decode_mode
                for item in recognizer.recognize_batch_detailed(
                    [crop.image for crop in refinement.crops[: min(3, len(refinement.crops))]],
                    include_preview=False,
                )
            }
        )
        if refinement.crops
        else [],
    }
    (fixture_output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def assert_summary(summary: dict, fixture: PortfolioFixture) -> None:
    line_count = summary["line_count"]
    if not fixture.line_min <= line_count <= fixture.line_max:
        raise AssertionError(
            f"{fixture.fixture_id}: expected line count in [{fixture.line_min}, {fixture.line_max}], got {line_count}"
        )
    if summary["fallback_rate"] > fixture.fallback_rate_max:
        raise AssertionError(
            f"{fixture.fixture_id}: fallback_rate {summary['fallback_rate']:.2f} > {fixture.fallback_rate_max:.2f}"
        )
    gain = summary["mean_raw_blank_margin"] - summary["mean_final_blank_margin"]
    if gain < fixture.blank_margin_gain_min:
        raise AssertionError(
            f"{fixture.fixture_id}: blank-margin gain {gain:.3f} < {fixture.blank_margin_gain_min:.3f}"
        )


def run_portfolio_regression(
    output_root: Path = OUTPUT_ROOT,
    model_key: str | None = None,
) -> dict:
    fixtures = load_manifest()
    effective_model, detector, cropper, recognizer, refiner, preprocessor = load_runtime_components(model_key)

    summaries = []
    for fixture in fixtures:
        summary = evaluate_fixture(
            fixture,
            detector=detector,
            cropper=cropper,
            recognizer=recognizer,
            refiner=refiner,
            preprocessor=preprocessor,
            output_root=output_root,
        )
        assert_summary(summary, fixture)
        summaries.append(summary)

    curved_hits = [
        summary["fixture_id"]
        for summary, fixture in zip(summaries, fixtures)
        if fixture.curved_lines and summary["curve_actions"] > 0
    ]
    if not curved_hits:
        raise AssertionError("No curved fixture triggered the curve-aware branch.")

    final_summary = {
        "model_key": effective_model,
        "fixtures": summaries,
        "curved_branch_fixtures": curved_hits,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return final_summary


def compare_decode_modes(
    crops: list[Image.Image],
    recognizer: VietOCRRecognizer,
) -> dict:
    if not crops:
        return {"beam": None, "greedy": None}

    original_mode = getattr(recognizer, "_decode_mode", "greedy")
    detailed = recognizer.recognize_batch_detailed(crops, include_preview=False)
    default_scores = [item.probability for item in detailed]
    return {
        "default": {
            "decode_mode": detailed[0].decode_mode if detailed else original_mode,
            "mean_confidence": float(mean(default_scores)) if default_scores else 0.0,
        }
    }


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


if __name__ == "__main__":
    summary = run_portfolio_regression()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
