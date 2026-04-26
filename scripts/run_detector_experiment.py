#!/usr/bin/env python3
"""Phase 3 — detector backend leaderboard.

Runs every image in a target directory through the OCR pipeline once per
detector backend (Tier-1 candidates: Paddle / Surya / CRAFT / Kraken
BLLA). Phase 2C rectifier and Phase 2B refiner/cropper stay unchanged;
only the detector swaps.

Per (experiment, image) we save:
  debug/<exp_id>/<image_id>/
    original.jpg           input image (rectified copy if rectifier ran)
    overlay.jpg            polygons drawn on the rectified image
    crops/*.jpg            per-line crops fed to VietOCR
    text.txt               final OCR text
    metrics.json           detector + recognizer diagnostic block

And aggregate to:
  per_image_metrics.jsonl  one row per (experiment, image)
  summary.json             aggregate metrics per experiment
  summary.md               human-readable comparison table

Why all four metrics blocks (detector + recognizer):
  * detector_metrics — measure of segmentation quality.
  * recognizer diagnostic metrics — measure of *downstream* damage.
    A great detector may still feed a weak recognizer; we want to see
    that confound clearly.
"""
from __future__ import annotations

import argparse
import json
import logging
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

from ocr_pipeline.experiment_config import ExperimentConfig  # noqa: E402
from ocr_pipeline.pipeline import OCRPipeline  # noqa: E402
from ocr_pipeline.validation.metrics import (  # noqa: E402
    detector_metrics,
    diagnostic_metrics,
    digit_noise_rate,
    is_hallucinated_line,
)

logger = logging.getLogger("detector_experiment")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ── Experiment definitions ──────────────────────────────────────────────────


@dataclass
class ExperimentSpec:
    """One row of the leaderboard."""

    exp_id: str
    description: str
    detector_backend: str
    detector_kwargs: dict[str, Any]
    detector_allow_grid_fallback: bool = False
    enable_document_perspective_correction: bool = True
    rectifier_backend: str = "hybrid"


# E0' — Paddle control. The silent grid fallback is **off** even for the
# control: the brief is to compare on the modern, no-fallback contract.
# We separately log when Paddle returns 0 boxes so the failure is visible.
# E1 — Surya. E2 — CRAFT (vendored). E3 — Kraken BLLA.
DEFAULT_EXPERIMENTS: list[ExperimentSpec] = [
    ExperimentSpec(
        exp_id="E0_paddle",
        description="Control: PaddleOCR DBNet, silent grid fallback DISABLED.",
        detector_backend="paddle",
        detector_kwargs={},
        detector_allow_grid_fallback=False,
    ),
    ExperimentSpec(
        exp_id="E1_surya",
        description="Surya document detector (per-line polygons natively).",
        detector_backend="surya",
        detector_kwargs={"min_confidence": 0.4},
    ),
    ExperimentSpec(
        exp_id="E2_craft",
        description="CRAFT word detector + row-clustering line merger.",
        detector_backend="craft",
        detector_kwargs={
            "row_overlap_ratio": 0.4,
            "x_pad_ratio": 0.02,
            "y_pad_ratio": 0.10,
        },
    ),
    ExperimentSpec(
        exp_id="E3_kraken_blla",
        description="Kraken BLLA baseline-aware line segmenter.",
        detector_backend="kraken",
        detector_kwargs={},
    ),
]


# ── Helpers ─────────────────────────────────────────────────────────────────


def safe_id(image_path: Path, index: int) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in image_path.stem)
    return f"{index:04d}_{stem.strip('_') or image_path.name}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def draw_overlay(image: Image.Image, debug_pages: list, target: Path) -> None:
    rgb = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rgb)
    for page in debug_pages:
        for poly in getattr(page, "raw_polygons", []) or []:
            pts = [(float(x), float(y)) for x, y in poly]
            if len(pts) >= 2:
                pts.append(pts[0])
                draw.line(pts, fill=(255, 0, 0), width=2)
    target.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(target, "JPEG", quality=85)


def save_crops(debug_pages: list, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    import base64
    import io

    counter = 0
    for page in debug_pages:
        for crop in getattr(page, "final_crops", []) or []:
            blob = getattr(crop, "preview_base64", None)
            if not blob:
                continue
            try:
                data = base64.b64decode(blob)
                Image.open(io.BytesIO(data)).save(target_dir / f"line_{counter:03d}.jpg", "JPEG", quality=88)
                counter += 1
            except Exception:
                continue


def line_texts(result) -> list[str]:
    if result is None:
        return []
    out: list[str] = []
    for page_result in getattr(result, "pages", []) or []:
        for line in page_result.lines:
            out.append(line.text)
    return out


def collect_polygons(debug_pages: list) -> list[list[list[float]]]:
    polys: list[list[list[float]]] = []
    for page in debug_pages:
        for poly in getattr(page, "raw_polygons", []) or []:
            polys.append([[float(x), float(y)] for x, y in poly])
    return polys


def collect_image_size(debug_pages: list, fallback: tuple[int, int]) -> tuple[int, int]:
    """Pull the rectified image size from the first debug page, falling back to
    the original image's size."""
    for page in debug_pages:
        diag = getattr(page, "rectifier_diagnostics", None) or {}
        size = diag.get("output_size") or diag.get("image_size")
        if size and len(size) == 2:
            return int(size[0]), int(size[1])
    return fallback


# ── Per-image runner ────────────────────────────────────────────────────────


def run_one(
    spec: ExperimentSpec,
    pipeline: OCRPipeline,
    image_path: Path,
    image_index: int,
    out_dir: Path,
    aggregate_jsonl: Path,
) -> dict:
    """Run a single (experiment, image), save artifacts, return one row."""
    image_id = safe_id(image_path, image_index)
    image_dir = out_dir / spec.exp_id / "debug" / image_id
    image_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = image_dir / "crops"

    raw_bytes = image_path.read_bytes()
    pil_image = Image.open(image_path).convert("RGB")
    pil_image.save(image_dir / "original.jpg", "JPEG", quality=88)

    t_start = time.perf_counter()
    try:
        debug = pipeline.process_bytes_debug(raw_bytes, image_path.name)
    except Exception as exc:  # pragma: no cover - logged for the report
        logger.exception("[%s] %s failed", spec.exp_id, image_path.name)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        row = {
            "exp_id": spec.exp_id,
            "image": image_path.name,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": elapsed_ms,
        }
        append_jsonl(aggregate_jsonl, row)
        return row
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    debug_pages = list(debug.debug_pages or [])
    texts = line_texts(debug.result)
    polygons = collect_polygons(debug_pages)
    image_w, image_h = pil_image.size
    rect_w, rect_h = collect_image_size(debug_pages, (image_w, image_h))

    det_diag_block = {}
    low_recall_flag = False
    detector_backend = spec.detector_backend
    if debug_pages:
        first = debug_pages[0]
        det_diag_block = dict(getattr(first, "detector_diagnostics", None) or {})
        low_recall_flag = bool(getattr(first, "low_detector_recall", False))
        detector_backend = getattr(first, "detector_backend", None) or detector_backend

    det_metrics = detector_metrics(polygons, rect_w, rect_h)
    rec_metrics = diagnostic_metrics(texts)
    rec_metrics["worst_lines"] = sorted(
        (
            {
                "text": t,
                "digit_noise_rate": digit_noise_rate(t),
                "is_hallucinated": is_hallucinated_line(t),
            }
            for t in texts
        ),
        key=lambda x: -float(x["digit_noise_rate"]),
    )[:5]

    write_text(image_dir / "text.txt", "\n".join(texts))
    draw_overlay(pil_image, debug_pages, image_dir / "overlay.jpg")
    save_crops(debug_pages, crops_dir)

    write_json(
        image_dir / "metrics.json",
        {
            "exp_id": spec.exp_id,
            "image": image_path.name,
            "elapsed_ms": elapsed_ms,
            "low_detector_recall": low_recall_flag,
            "detector_backend": detector_backend,
            "detector_diagnostics": det_diag_block,
            "detector_metrics": det_metrics,
            "recognizer_metrics": rec_metrics,
        },
    )

    row = {
        "exp_id": spec.exp_id,
        "description": spec.description,
        "image": image_path.name,
        "image_id": image_id,
        "elapsed_ms": elapsed_ms,
        "detector_backend": detector_backend,
        "low_detector_recall": low_recall_flag,
        "detection_count": det_metrics["detection_count"],
        "full_width_band_rate": det_metrics["full_width_band_rate"],
        "mean_box_aspect_ratio": det_metrics["mean_box_aspect_ratio"],
        "mean_distinct_x1_per_page": det_metrics["mean_distinct_x1_per_page"],
        "median_box_height": det_metrics["median_box_height"],
        "image_width": det_metrics["image_width"],
        "image_height": det_metrics["image_height"],
        "digit_noise_rate": rec_metrics.get("digit_noise_rate"),
        "garbage_text_ratio": rec_metrics.get("garbage_text_ratio"),
        "hallucinated_line_rate": rec_metrics.get("hallucinated_line_rate"),
        "repeated_number_sequence_count": rec_metrics.get("repeated_number_sequence_count"),
        "uppercase_garbage_token_rate": rec_metrics.get("uppercase_garbage_token_rate"),
        "abnormal_symbol_rate": rec_metrics.get("abnormal_symbol_rate"),
        "line_count": rec_metrics.get("line_count"),
    }
    append_jsonl(aggregate_jsonl, row)
    return row


# ── Aggregation ─────────────────────────────────────────────────────────────


def aggregate(rows: list[dict]) -> dict:
    """Per-experiment averages over the per-image rows."""
    out: dict[str, dict[str, float]] = {}
    by_exp: dict[str, list[dict]] = {}
    for row in rows:
        if "error" in row:
            continue
        by_exp.setdefault(row["exp_id"], []).append(row)
    for exp_id, exp_rows in by_exp.items():
        n = len(exp_rows)
        keys = [
            "detection_count",
            "full_width_band_rate",
            "mean_box_aspect_ratio",
            "mean_distinct_x1_per_page",
            "median_box_height",
            "digit_noise_rate",
            "garbage_text_ratio",
            "hallucinated_line_rate",
            "repeated_number_sequence_count",
            "uppercase_garbage_token_rate",
            "abnormal_symbol_rate",
            "line_count",
            "elapsed_ms",
        ]
        agg: dict[str, float] = {}
        for k in keys:
            vals = [float(r[k]) for r in exp_rows if r.get(k) is not None]
            agg[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
        agg["n_images"] = n
        agg["low_recall_rate"] = round(
            sum(1 for r in exp_rows if r.get("low_detector_recall")) / max(n, 1),
            4,
        )
        out[exp_id] = agg
    return out


def write_summary_md(summary: dict, specs: dict[str, ExperimentSpec], target: Path) -> None:
    lines = ["# Detector backend leaderboard", ""]
    lines.append("Aggregate metrics per experiment (mean over all test images).")
    lines.append("")
    lines.append(
        "| exp_id | description | n | low_recall | det/page | full_band | aspect | distinct_x1 | "
        "halluc_rate | digit_noise | garbage | up_garbage | repeated# | latency_ms |"
    )
    lines.append(
        "|--------|-------------|---|------------|----------|-----------|--------|-------------|"
        "-------------|-------------|---------|------------|-----------|------------|"
    )
    for exp_id, agg in summary.items():
        spec = specs.get(exp_id)
        desc = spec.description if spec else ""
        lines.append(
            f"| {exp_id} | {desc} | {int(agg['n_images'])} | {agg['low_recall_rate']:.2f} | "
            f"{agg['detection_count']:.1f} | {agg['full_width_band_rate']:.3f} | "
            f"{agg['mean_box_aspect_ratio']:.2f} | {agg['mean_distinct_x1_per_page']:.1f} | "
            f"{agg['hallucinated_line_rate']:.3f} | {agg['digit_noise_rate']:.3f} | "
            f"{agg['garbage_text_ratio']:.3f} | {agg['uppercase_garbage_token_rate']:.3f} | "
            f"{agg['repeated_number_sequence_count']:.1f} | {agg['elapsed_ms']:.0f} |"
        )
    lines.append("")
    lines.append(
        "**Reading guide.** ``low_recall`` = fraction of pages where the "
        "detector returned zero boxes (the explicit replacement for the silent "
        "OpenCV grid fallback). ``full_band`` should be ~0; values near 1 mean "
        "the detector is emitting full-page-wide bands. ``halluc_rate`` is "
        "VietOCR's hallucinated-line rate downstream of the detector — a "
        "lower bound on detector-induced damage."
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


# ── Entrypoint ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 detector leaderboard.")
    p.add_argument("--input-dir", type=Path, default=ROOT / "tests" / "test")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "runs" / "phase3_detector_comparison",
    )
    p.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Subset of experiment ids to run (default: all).",
    )
    p.add_argument(
        "--model-key",
        default=None,
        help="VietOCR registry key (default: project default; "
        "for Phase 3 we run with `baseline_pretrained` since "
        "`experiment_B_50k` weights are not in the repo).",
    )
    p.add_argument("--limit", type=int, default=None, help="Process at most N images.")
    p.add_argument(
        "--enable-rectifier",
        action="store_true",
        default=True,
        help="Enable Phase 2C rectifier (kept for parity with rectification eval).",
    )
    return p.parse_args()


def find_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def build_pipeline(spec: ExperimentSpec, model_key: str | None) -> OCRPipeline:
    config = ExperimentConfig(
        name=spec.exp_id,
        description=spec.description,
        model_key=model_key,
        detector_backend=spec.detector_backend,
        detector_allow_grid_fallback=spec.detector_allow_grid_fallback,
        detector_kwargs=dict(spec.detector_kwargs),
        enable_document_perspective_correction=spec.enable_document_perspective_correction,
        rectifier_backend=spec.rectifier_backend,
        rectifier_save_debug=False,
    )
    return OCRPipeline.from_experiment_config(config)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(args.input_dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        logger.error("No images found in %s", args.input_dir)
        return 1

    chosen_specs = [s for s in DEFAULT_EXPERIMENTS if (not args.experiments or s.exp_id in args.experiments)]
    spec_map = {s.exp_id: s for s in chosen_specs}

    aggregate_jsonl = args.output_dir / "per_image_metrics.jsonl"
    # Only truncate when running the full default set; otherwise we'd lose
    # results from previous backends when re-running a single experiment.
    if aggregate_jsonl.exists() and not args.experiments:
        aggregate_jsonl.unlink()

    all_rows: list[dict] = []
    for spec in chosen_specs:
        logger.info("=" * 70)
        logger.info("Experiment %s — %s", spec.exp_id, spec.description)
        logger.info("=" * 70)
        try:
            pipeline = build_pipeline(spec, args.model_key)
        except Exception as exc:
            logger.exception("Failed to build pipeline for %s: %s", spec.exp_id, exc)
            row = {"exp_id": spec.exp_id, "error": f"build:{type(exc).__name__}: {exc}"}
            append_jsonl(aggregate_jsonl, row)
            all_rows.append(row)
            continue

        for index, image_path in enumerate(images):
            logger.info("[%s] %s (%d/%d)", spec.exp_id, image_path.name, index + 1, len(images))
            row = run_one(spec, pipeline, image_path, index, args.output_dir, aggregate_jsonl)
            all_rows.append(row)

    # Re-read the full JSONL so summary covers all experiments — including
    # those run in a previous invocation.
    full_rows: list[dict] = []
    if aggregate_jsonl.exists():
        with aggregate_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    full_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    summary = aggregate(full_rows or all_rows)
    # Build spec_map from all known specs (default + currently chosen) so the
    # summary description column is populated even for experiments rerun later.
    full_spec_map = {s.exp_id: s for s in DEFAULT_EXPERIMENTS}
    full_spec_map.update(spec_map)
    write_json(args.output_dir / "summary.json", summary)
    write_summary_md(summary, full_spec_map, args.output_dir / "summary.md")
    logger.info("Wrote %s", args.output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
