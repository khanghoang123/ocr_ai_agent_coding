from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image

from ocr_pipeline.validation.metrics import digit_noise_rate


GEOMETRY_METRIC_KEYS = (
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


@dataclass
class AnalyzerThresholds:
    tiny_box_height_ratio: float = 0.45
    tiny_box_width_ratio: float = 0.08
    min_tiny_crop_width: int = 80
    merged_box_height_ratio: float = 1.8
    overlap_ratio: float = 0.35
    min_aspect_ratio: float = 3.0
    max_aspect_ratio: float = 55.0
    high_digit_noise_threshold: float = 0.20


@dataclass
class AnalyzedLine:
    image: str
    line_index: int | None
    crop_path: str | None
    bbox: dict[str, float] | None
    crop_width: int | None
    crop_height: int | None
    crop_aspect_ratio: float | None
    digit_noise_rate: float
    recognition_score: float | None = None
    flags: list[str] = field(default_factory=list)


def analyze_debug_dir(
    debug_dir: str | Path,
    thresholds: AnalyzerThresholds | None = None,
) -> dict[str, Any]:
    debug_path = Path(debug_dir)
    mapping_path = debug_path / "debug_mapping.json"
    if not mapping_path.exists():
        return _empty_summary(debug_path, reason="missing_debug_mapping")

    thresholds = thresholds or AnalyzerThresholds()
    with mapping_path.open("r", encoding="utf-8") as f:
        mappings = json.load(f)

    all_lines: list[AnalyzedLine] = []
    image_summaries: list[dict[str, Any]] = []

    for image_item in mappings:
        lines = image_item.get("lines") or []
        analyzed = [_analyze_line(image_item, line, debug_path) for line in lines]
        _apply_image_relative_flags(analyzed, thresholds)
        all_lines.extend(analyzed)
        image_summaries.append(_summarize_lines(image_item.get("image"), analyzed))

    aggregate = _summarize_lines("all", all_lines)
    return {
        "debug_dir": str(debug_path),
        "mapping_path": str(mapping_path),
        "thresholds": asdict(thresholds),
        "aggregate": aggregate,
        "images": image_summaries,
        "lines": [asdict(line) for line in all_lines],
    }


def write_debug_analysis(
    debug_dir: str | Path,
    output: str | Path,
    thresholds: AnalyzerThresholds | None = None,
) -> dict[str, Any]:
    summary = analyze_debug_dir(debug_dir, thresholds=thresholds)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _analyze_line(image_item: dict[str, Any], line: dict[str, Any], debug_dir: Path) -> AnalyzedLine:
    crop_path = line.get("crop_path")
    crop_width = None
    crop_height = None
    if crop_path:
        path = Path(crop_path)
        if not path.is_absolute():
            candidates = [path, debug_dir / path, debug_dir / "crops" / path.name]
            path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        try:
            with Image.open(path) as image:
                crop_width, crop_height = image.size
        except Exception:
            crop_width = None
            crop_height = None

    aspect = None
    if crop_width is not None and crop_height:
        aspect = crop_width / max(crop_height, 1)

    text = line.get("cleaned_text") or line.get("raw_text") or ""
    flags = list(line.get("flags") or [])
    recognition_score = line.get("recognition_score")
    try:
        recognition_score = float(recognition_score) if recognition_score is not None else None
    except (TypeError, ValueError):
        recognition_score = None

    analyzed = AnalyzedLine(
        image=image_item.get("image") or image_item.get("id") or "",
        line_index=line.get("line_index"),
        crop_path=crop_path,
        bbox=line.get("bbox"),
        crop_width=crop_width,
        crop_height=crop_height,
        crop_aspect_ratio=aspect,
        digit_noise_rate=digit_noise_rate(text),
        recognition_score=recognition_score,
        flags=flags,
    )
    if _is_garbage_text(text):
        analyzed.flags.append("garbage_text")
    return analyzed


def _apply_image_relative_flags(
    lines: list[AnalyzedLine],
    thresholds: AnalyzerThresholds,
) -> None:
    bbox_widths = [_bbox_width(line.bbox) for line in lines if _bbox_width(line.bbox) is not None]
    bbox_heights = [_bbox_height(line.bbox) for line in lines if _bbox_height(line.bbox) is not None]
    median_width = median(bbox_widths) if bbox_widths else 0.0
    median_height = median(bbox_heights) if bbox_heights else 0.0

    sorted_lines = sorted(
        [line for line in lines if line.bbox is not None],
        key=lambda line: float(line.bbox.get("y1", 0.0)),
    )

    for index, line in enumerate(sorted_lines):
        bbox_h = _bbox_height(line.bbox) or 0.0
        bbox_w = _bbox_width(line.bbox) or 0.0
        crop_w = line.crop_width or 0

        tiny_by_width = bbox_w < max(1.0, median_width * thresholds.tiny_box_width_ratio)
        tiny_by_crop = crop_w > 0 and crop_w < thresholds.min_tiny_crop_width
        tiny_by_height = median_height > 0 and bbox_h < median_height * thresholds.tiny_box_height_ratio
        if tiny_by_width or tiny_by_crop or (tiny_by_height and tiny_by_crop):
            line.flags.append("tiny_box")

        if median_height > 0 and bbox_h > median_height * thresholds.merged_box_height_ratio:
            line.flags.append("merged_box")

        if _has_significant_y_overlap(line, _neighbor(sorted_lines, index - 1), thresholds.overlap_ratio):
            line.flags.append("merged_box")
        elif _has_significant_y_overlap(line, _neighbor(sorted_lines, index + 1), thresholds.overlap_ratio):
            line.flags.append("merged_box")

        if line.crop_aspect_ratio is not None and (
            line.crop_aspect_ratio < thresholds.min_aspect_ratio
            or line.crop_aspect_ratio > thresholds.max_aspect_ratio
        ):
            line.flags.append("abnormal_crop")

        if line.digit_noise_rate >= thresholds.high_digit_noise_threshold:
            line.flags.append("high_digit_noise")


def _summarize_lines(image: str | None, lines: list[AnalyzedLine]) -> dict[str, Any]:
    count = len(lines)
    aspects = [line.crop_aspect_ratio for line in lines if line.crop_aspect_ratio is not None]
    scores = [line.recognition_score for line in lines if line.recognition_score is not None]
    invalid_flags = {
        "tiny_box",
        "merged_box",
        "abnormal_crop",
        "high_digit_noise",
        "too_thin_crop",
        "low_ink_density",
        "abnormal_aspect_ratio",
        "full_width_background_risk",
        "garbage_text",
    }
    valid_count = sum(1 for line in lines if not (set(line.flags) & invalid_flags))
    summary = {
        "image": image,
        "line_count": count,
        "tiny_box_count": _flag_count(lines, "tiny_box"),
        "merged_box_count": _flag_count(lines, "merged_box"),
        "abnormal_crop_count": _flag_count(lines, "abnormal_crop"),
        "high_digit_noise_line_count": _flag_count(lines, "high_digit_noise"),
        "garbage_text_count": _flag_count(lines, "garbage_text"),
        "too_thin_crop_count": _flag_count(lines, "too_thin_crop"),
        "blank_crop_count": _flag_count(lines, "low_ink_density"),
        "neighbor_fragment_count": _flag_count(lines, "vertical_neighbor_risk"),
        "deskew_applied_count": _flag_count(lines, "line_deskewed"),
        "valid_line_count": valid_count,
        "average_crop_aspect_ratio": (sum(aspects) / len(aspects)) if aspects else None,
        "confidence_score_avg": (sum(scores) / len(scores)) if scores else None,
        "average_abs_deskew_angle": _average_abs_deskew_angle(lines),
    }
    summary["tiny_box_rate"] = _rate(summary["tiny_box_count"], count)
    summary["merged_box_rate"] = _rate(summary["merged_box_count"], count)
    summary["abnormal_crop_ratio"] = _rate(summary["abnormal_crop_count"], count)
    summary["high_digit_noise_line_rate"] = _rate(summary["high_digit_noise_line_count"], count)
    summary["garbage_text_ratio"] = _rate(summary["garbage_text_count"], count)
    summary["valid_line_ratio"] = _rate(summary["valid_line_count"], count)
    summary["too_thin_crop_rate"] = _rate(summary["too_thin_crop_count"], count)
    summary["blank_crop_rate"] = _rate(summary["blank_crop_count"], count)
    summary["neighbor_fragment_rate"] = _rate(summary["neighbor_fragment_count"], count)
    summary["deskew_applied_rate"] = _rate(summary["deskew_applied_count"], count)
    return summary


def _empty_summary(debug_dir: Path, reason: str) -> dict[str, Any]:
    aggregate = {
        "image": "all",
        "line_count": 0,
        "tiny_box_count": 0,
        "merged_box_count": 0,
        "abnormal_crop_count": 0,
        "high_digit_noise_line_count": 0,
        "garbage_text_count": 0,
        "too_thin_crop_count": 0,
        "blank_crop_count": 0,
        "neighbor_fragment_count": 0,
        "deskew_applied_count": 0,
        "valid_line_count": 0,
        "average_crop_aspect_ratio": None,
        "confidence_score_avg": None,
        "average_abs_deskew_angle": None,
        "tiny_box_rate": 0.0,
        "merged_box_rate": 0.0,
        "abnormal_crop_ratio": 0.0,
        "high_digit_noise_line_rate": 0.0,
        "garbage_text_ratio": 0.0,
        "valid_line_ratio": 0.0,
        "too_thin_crop_rate": 0.0,
        "blank_crop_rate": 0.0,
        "neighbor_fragment_rate": 0.0,
        "deskew_applied_rate": 0.0,
    }
    return {
        "debug_dir": str(debug_dir),
        "reason": reason,
        "aggregate": aggregate,
        "images": [],
        "lines": [],
    }


def _flag_count(lines: list[AnalyzedLine], flag: str) -> int:
    return sum(1 for line in lines if flag in line.flags)


def _rate(value: int, total: int) -> float:
    return value / total if total else 0.0


def _bbox_width(bbox: dict[str, Any] | None) -> float | None:
    if not bbox:
        return None
    return float(bbox["x2"]) - float(bbox["x1"])


def _bbox_height(bbox: dict[str, Any] | None) -> float | None:
    if not bbox:
        return None
    return float(bbox["y2"]) - float(bbox["y1"])


def _neighbor(lines: list[AnalyzedLine], index: int) -> AnalyzedLine | None:
    if index < 0 or index >= len(lines):
        return None
    return lines[index]


def _has_significant_y_overlap(
    line: AnalyzedLine,
    other: AnalyzedLine | None,
    threshold: float,
) -> bool:
    if other is None or line.bbox is None or other.bbox is None:
        return False
    y1 = max(float(line.bbox["y1"]), float(other.bbox["y1"]))
    y2 = min(float(line.bbox["y2"]), float(other.bbox["y2"]))
    if y2 <= y1:
        return False
    overlap = y2 - y1
    min_height = min(_bbox_height(line.bbox) or 0.0, _bbox_height(other.bbox) or 0.0)
    return min_height > 0 and (overlap / min_height) >= threshold


def _is_garbage_text(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    letters = sum(ch.isalpha() for ch in stripped)
    digits = sum(ch.isdigit() for ch in stripped)
    alnum = letters + digits
    punctuation = sum((not ch.isalnum() and not ch.isspace()) for ch in stripped)
    if alnum == 0:
        return True
    digit_ratio = digits / max(alnum, 1)
    punctuation_ratio = punctuation / max(len(stripped), 1)
    uppercase_tokens = [token for token in stripped.split() if len(token) >= 2 and token.isupper()]
    uppercase_ratio = len(uppercase_tokens) / max(len(stripped.split()), 1)
    return (
        digit_ratio > 0.55
        or punctuation_ratio > 0.35
        or (uppercase_ratio > 0.65 and letters > 8)
        or len(stripped) <= 2
    )


def _average_abs_deskew_angle(lines: list[AnalyzedLine]) -> float | None:
    values: list[float] = []
    for line in lines:
        for flag in line.flags:
            if not flag.startswith("deskew_angle="):
                continue
            try:
                values.append(abs(float(flag.split("=", 1)[1])))
            except (IndexError, ValueError):
                continue
    if not values:
        return None
    return sum(values) / len(values)
