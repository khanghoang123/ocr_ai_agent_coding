#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_pipeline.experiment_config import (  # noqa: E402
    ExperimentConfig,
    dump_experiment_config,
    load_experiment_config,
)
from ocr_pipeline.pipeline import OCRPipeline  # noqa: E402
from ocr_pipeline.recognizer.vietnamese_postprocess import safe_postprocess_lines  # noqa: E402
from ocr_pipeline.validation.metrics import (  # noqa: E402
    aggregate_nullable,
    auto_research_score,
    character_error_rate,
    digit_noise_rate,
    normalized_line_count_error,
    word_error_rate,
)
from ocr_pipeline.validation.debug_analyzer import analyze_debug_dir  # noqa: E402

logger = logging.getLogger("run_ocr_experiment")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DatasetItem:
    image_path: Path
    ground_truth: str | None = None
    expected_line_count: int | None = None
    item_id: str | None = None


def create_pipeline(config: ExperimentConfig) -> OCRPipeline:
    pipeline = OCRPipeline.from_experiment_config(config)
    recognizer = getattr(pipeline, "recognizer", None)
    if recognizer is None or not hasattr(recognizer, "_ensure_loaded"):
        return pipeline
    try:
        recognizer._ensure_loaded()
    except ModuleNotFoundError as exc:
        if exc.name != "torchvision":
            raise
        logger.warning(
            "VietOCR recognizer dependency missing (%s); "
            "running experiment in detection/crop-only mode.",
            exc,
        )
        pipeline.recognizer = None
        pipeline.unsupported_options.append("vietocr_recognizer_missing_torchvision")
    return pipeline


def _resolve_image_path(raw_path: str, base_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    candidates = [
        base_dir / candidate,
        ROOT / candidate,
        ROOT / "Dataset" / candidate,
        ROOT / "Dataset" / "data" / candidate.name,
        ROOT / "data" / candidate,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_dataset(dataset: str | Path, limit: int | None = None) -> list[DatasetItem]:
    dataset_path = Path(dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    items: list[DatasetItem] = []
    if dataset_path.is_dir():
        for image_path in sorted(path for path in dataset_path.rglob("*") if path.suffix.lower() in IMAGE_EXTS):
            items.append(DatasetItem(image_path=image_path, item_id=image_path.stem))
    elif dataset_path.suffix.lower() == ".json":
        with dataset_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        fixtures = payload.get("fixtures", payload if isinstance(payload, list) else [])
        for fixture in fixtures:
            raw_image = fixture.get("image_path") or fixture.get("path")
            if not raw_image:
                continue
            expected = fixture.get("expected_line_count")
            if isinstance(expected, dict):
                lo = expected.get("min")
                hi = expected.get("max")
                expected_count = int(round((int(lo) + int(hi)) / 2)) if lo is not None and hi is not None else None
            else:
                expected_count = int(expected) if expected is not None else None
            items.append(
                DatasetItem(
                    image_path=_resolve_image_path(str(raw_image), dataset_path.parent),
                    ground_truth=fixture.get("ground_truth") or fixture.get("text"),
                    expected_line_count=expected_count,
                    item_id=fixture.get("id"),
                )
            )
    else:
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t", 1)
                raw_image = parts[0]
                gt = parts[1] if len(parts) > 1 else None
                items.append(
                    DatasetItem(
                        image_path=_resolve_image_path(raw_image, dataset_path.parent),
                        ground_truth=gt,
                        expected_line_count=1 if gt is not None else None,
                        item_id=Path(raw_image).stem,
                    )
                )

    return items[:limit] if limit is not None else items


def _safe_name(path: Path, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or f"item_{index:04d}"
    return f"{index:04d}_{stem}"


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _decode_preview(preview: str, target: Path) -> str | None:
    if not preview:
        return None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(preview))
        return str(target)
    except Exception as exc:
        logger.debug("Could not decode crop preview %s: %s", target, exc)
        return None


def _draw_overlay(image_path: Path, debug_pages: list[Any], target: Path) -> str | None:
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for page in debug_pages:
            for poly in getattr(page, "raw_polygons", []) or []:
                points = [(float(x), float(y)) for x, y in poly]
                if len(points) >= 2:
                    draw.line(points + [points[0]], fill=(255, 0, 0), width=3)
            for box in getattr(page, "refined_boxes", []) or []:
                draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=(0, 180, 0), width=2)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target)
        return str(target)
    except Exception as exc:
        logger.debug("Could not draw overlay for %s: %s", image_path, exc)
        return None


def _result_to_text(result: Any) -> str:
    return getattr(result, "full_text", "") or ""


def _line_texts(result: Any) -> list[str]:
    lines: list[str] = []
    for page in getattr(result, "pages", []) or []:
        for line in getattr(page, "lines", []) or []:
            lines.append(getattr(line, "text", ""))
    return lines


def _write_debug_artifacts(
    item: DatasetItem,
    run_name: str,
    debug_item: Any,
    debug_dir: Path,
    config: ExperimentConfig,
) -> tuple[dict[str, Any], int, list[str]]:
    unsupported: list[str] = []
    mapping: dict[str, Any] = {
        "image": str(item.image_path),
        "id": item.item_id,
        "overlay_path": None,
        "intermediate_crops": [],
        "lines": [],
    }
    debug_pages = getattr(debug_item, "debug_pages", []) or []
    if not debug_pages:
        unsupported.append("debug_pages")
    overlay_path = _draw_overlay(item.image_path, debug_pages, debug_dir / f"{run_name}_overlay.jpg")
    mapping["overlay_path"] = overlay_path

    crop_flag_count = 0
    for page in debug_pages:
        if config.debug_save_intermediate_crops:
            for decision in getattr(page, "decisions", []) or []:
                rectified_path = _decode_preview(
                    getattr(decision, "rectified_preview_base64", ""),
                    debug_dir
                    / "intermediate"
                    / f"{run_name}_p{page.page_number}_d{decision.source_index}_rectified.png",
                )
                mask_path = _decode_preview(
                    getattr(decision, "mask_preview_base64", ""),
                    debug_dir
                    / "intermediate"
                    / f"{run_name}_p{page.page_number}_d{decision.source_index}_mask.png",
                )
                if rectified_path or mask_path:
                    mapping["intermediate_crops"].append(
                        {
                            "page_number": page.page_number,
                            "source_index": decision.source_index,
                            "action": decision.action,
                            "note": decision.note,
                            "rectified_path": rectified_path,
                            "mask_path": mask_path,
                        }
                    )
        final_crops = getattr(page, "final_crops", []) or []
        if not final_crops:
            unsupported.append("final_crops")
        for crop in final_crops:
            crop_path = _decode_preview(
                getattr(crop, "preview_base64", ""),
                debug_dir / "crops" / f"{run_name}_p{page.page_number}_l{crop.line_index}.jpg",
            )
            before_crop_path = _decode_preview(
                getattr(crop, "before_preview_base64", ""),
                debug_dir / "before_crops" / f"{run_name}_p{page.page_number}_l{crop.line_index}_before.jpg",
            )
            raw_text = getattr(crop, "text", "")
            processed = safe_postprocess_lines([raw_text], flag_digit_noise=config.flag_digit_noise)[0]
            crop_flags = list(getattr(crop, "crop_flags", []) or [])
            flags = sorted(set(crop_flags + processed.flags))
            crop_flag_count += len(flags)
            mapping["lines"].append(
                {
                    "page_number": page.page_number,
                    "line_index": crop.line_index,
                    "crop_path": crop_path,
                    "before_crop_path": before_crop_path,
                    "raw_text": raw_text,
                    "cleaned_text": processed.cleaned_text,
                    "recognition_score": getattr(crop, "recognition_score", None),
                    "flags": flags,
                    "crop_flags": crop_flags,
                    "suspicious_tokens": processed.suspicious_tokens,
                    "bbox": getattr(crop, "bbox", None).model_dump() if getattr(crop, "bbox", None) else None,
                }
            )
    return mapping, crop_flag_count, unsupported


def run_experiment(
    config_path: str | Path,
    dataset: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    items = load_dataset(dataset, limit=limit)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text_dir = out_dir / "texts"
    debug_dir = out_dir / "debug"
    dump_experiment_config(config, out_dir / "config_snapshot.yaml")

    pipeline = create_pipeline(config)
    predictions_path = out_dir / "predictions.jsonl"
    errors_path = out_dir / "errors.jsonl"
    mapping_path = debug_dir / "debug_mapping.json"
    errors_path.write_text("", encoding="utf-8")
    predictions_path.write_text("", encoding="utf-8")

    per_image_metrics = []
    debug_mappings = []
    unsupported_debug = set()
    started = time.perf_counter()
    recognition_available = getattr(pipeline, "recognizer", None) is not None

    for index, item in enumerate(items):
        run_name = _safe_name(item.image_path, index)
        try:
            data = item.image_path.read_bytes()
            if config.debug and hasattr(pipeline, "process_bytes_debug"):
                debug_item = pipeline.process_bytes_debug(data, item.image_path.name)
                result = debug_item.result
                mapping, crop_flag_count, unsupported = _write_debug_artifacts(
                    item,
                    run_name,
                    debug_item,
                    debug_dir,
                    config,
                )
                unsupported_debug.update(unsupported)
                debug_mappings.append(mapping)
            else:
                if config.debug:
                    unsupported_debug.add("process_bytes_debug")
                result = pipeline.process_bytes(data, item.image_path.name)
                crop_flag_count = 0

            prediction = _result_to_text(result)
            if config.enable_vietnamese_postprocess:
                processed_lines = safe_postprocess_lines(
                    _line_texts(result),
                    flag_digit_noise=config.flag_digit_noise,
                )
                prediction = "\n".join(item.cleaned_text for item in processed_lines)
                crop_flag_count += sum(len(item.flags) for item in processed_lines)

            text_dir.mkdir(parents=True, exist_ok=True)
            (text_dir / f"{run_name}.txt").write_text(prediction, encoding="utf-8")
            detection_count = int(getattr(result, "total_lines", 0))
            cer = (
                character_error_rate(prediction, item.ground_truth)
                if item.ground_truth is not None
                else None
            )
            wer = (
                word_error_rate(prediction, item.ground_truth)
                if item.ground_truth is not None
                else None
            )
            n_lce = normalized_line_count_error(detection_count, item.expected_line_count)
            image_metric = {
                "image_path": str(item.image_path),
                "ground_truth_available": item.ground_truth is not None,
                "cer": cer,
                "wer": wer,
                "digit_noise_rate": digit_noise_rate(prediction),
                "line_count_error": (
                    abs(detection_count - item.expected_line_count)
                    if item.expected_line_count is not None
                    else None
                ),
                "normalized_line_count_error": n_lce,
                "detection_count": detection_count,
                "crop_flag_count": crop_flag_count,
            }
            per_image_metrics.append(image_metric)
            _append_jsonl(
                predictions_path,
                {
                    "image_path": str(item.image_path),
                    "prediction": prediction,
                    "ground_truth": item.ground_truth,
                    "metrics": image_metric,
                },
            )
        except Exception as exc:
            logger.exception("Experiment item failed: %s", item.image_path)
            _append_jsonl(errors_path, {"image_path": str(item.image_path), "error": repr(exc)})

    _dump_json(mapping_path, debug_mappings)
    debug_analysis = analyze_debug_dir(debug_dir)
    geometry_metrics = {
        key: debug_analysis["aggregate"].get(key)
        for key in (
            "tiny_box_rate",
            "merged_box_rate",
            "abnormal_crop_ratio",
            "high_digit_noise_line_rate",
            "average_crop_aspect_ratio",
            "valid_line_ratio",
            "garbage_text_ratio",
            "confidence_score_avg",
            "too_thin_crop_rate",
            "blank_crop_rate",
            "neighbor_fragment_rate",
            "deskew_applied_rate",
            "average_abs_deskew_angle",
        )
    }

    metrics = {
        "config_name": config.name,
        "dataset": str(dataset),
        "processed_count": len(per_image_metrics),
        "error_count": _count_jsonl(errors_path),
        "cer": aggregate_nullable(item["cer"] for item in per_image_metrics),
        "wer": aggregate_nullable(item["wer"] for item in per_image_metrics),
        "digit_noise_rate": aggregate_nullable(item["digit_noise_rate"] for item in per_image_metrics),
        "normalized_line_count_error": aggregate_nullable(
            item["normalized_line_count_error"] for item in per_image_metrics
        ),
        "line_count_error": aggregate_nullable(item["line_count_error"] for item in per_image_metrics),
        "detection_count": aggregate_nullable(item["detection_count"] for item in per_image_metrics),
        "crop_flag_count": int(sum(item["crop_flag_count"] for item in per_image_metrics)),
        "recognition_available": recognition_available,
        "unsupported_options": sorted(
            set(config.unsupported_options + getattr(pipeline, "unsupported_options", []))
        ),
        "unsupported_debug_artifacts": sorted(unsupported_debug),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        **geometry_metrics,
        "per_image": per_image_metrics,
    }
    metrics["score"] = auto_research_score(metrics) if recognition_available else None
    _dump_json(out_dir / "metrics.json", metrics)
    return metrics


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OCR experiment from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--dataset", required=True, help="Dataset TSV, manifest JSON, or image directory.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of items to process.")
    parser.add_argument("--output-dir", required=True, help="Directory where artifacts are written.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    metrics = run_experiment(args.config, args.dataset, args.output_dir, limit=args.limit)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics.get("processed_count", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
