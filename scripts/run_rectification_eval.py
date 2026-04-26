#!/usr/bin/env python3
"""Phase 2C — page-level rectification before/after evaluation harness.

Runs every image in a target directory through the OCR pipeline TWICE:
  1. Without rectification (baseline / phase 2B config).
  2. With rectification (phase 2C config).

Saves, per image:
  - debug/<id>/original.jpg              raw input
  - debug/<id>/rectified.jpg             after rectification
  - debug/<id>/overlay_before.jpg        bbox overlay on raw
  - debug/<id>/overlay_after.jpg         bbox overlay on rectified
  - debug/<id>/crops_before/*.jpg        line crops without rectification
  - debug/<id>/crops_after/*.jpg         line crops with rectification
  - debug/<id>/text_before.txt           raw OCR output
  - debug/<id>/text_after.txt            OCR output with rectification
  - debug/<id>/metrics.json              before/after diagnostic metrics
                                         + verdict (improved/worsened/no_change)
                                         + worst lines (sorted by digit_noise)

And aggregates everything to:
  - per_image_metrics.jsonl
  - summary.json
  - summary.md  (human-readable comparison table)

This is a *diagnostic* evaluation: when ground truth is unavailable (the
common case for hand-photographed Vietnamese student notebooks), we cannot
compute CER/WER. Instead we score before/after via:
  - digit_noise_rate
  - garbage_text_ratio
  - high_digit_noise_line_rate
  - suspicious_digit_token_count
  - abnormal_symbol_rate
  - uppercase_garbage_token_rate
  - repeated_number_sequence_count
  - hallucinated_line_rate
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
from ocr_pipeline.validation.metrics import (  # noqa: E402
    diagnostic_diff,
    diagnostic_metrics,
    is_hallucinated_line,
    digit_noise_rate,
)

logger = logging.getLogger("rectification_eval")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline construction


def build_pipeline(config_path: Path | None, override: dict[str, Any] | None = None) -> tuple[OCRPipeline, ExperimentConfig]:
    if config_path is None:
        config = ExperimentConfig()
    else:
        config = load_experiment_config(config_path)
    if override:
        for key, value in override.items():
            setattr(config, key, value)
    pipeline = OCRPipeline.from_experiment_config(config)
    return pipeline, config


# ──────────────────────────────────────────────────────────────────────────────
# Helpers


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


def draw_bbox_overlay(image: Image.Image, debug_pages: list, target: Path) -> None:
    rgb = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rgb)
    for page in debug_pages:
        for poly in getattr(page, "raw_polygons", []) or []:
            pts = [(float(x), float(y)) for x, y in poly]
            if len(pts) >= 2:
                draw.line(pts + [pts[0]], fill=(255, 50, 50), width=3)
        for box in getattr(page, "refined_boxes", []) or []:
            draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=(50, 200, 50), width=2)
    target.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(target, format="JPEG", quality=85)


def line_texts(result) -> list[str]:
    texts: list[str] = []
    for page in getattr(result, "pages", []) or []:
        for line in getattr(page, "lines", []) or []:
            texts.append(line.text or "")
    return texts


def save_debug_crops(debug_pages: list, dir_: Path) -> list[str]:
    """Write `final_crops` previews to disk; return list of relative paths."""
    import base64

    saved: list[str] = []
    dir_.mkdir(parents=True, exist_ok=True)
    for page in debug_pages:
        for crop in getattr(page, "final_crops", []) or []:
            preview = getattr(crop, "preview_base64", None)
            if not preview:
                continue
            target = dir_ / f"p{page.page_number}_l{crop.line_index:03d}.jpg"
            try:
                target.write_bytes(base64.b64decode(preview))
                saved.append(str(target.relative_to(dir_.parent)))
            except Exception:
                continue
    return saved


def save_rectifier_image(debug_pages: list, kind: str, path: Path) -> bool:
    """Decode the original_image_base64 / rectified_image_base64 from debug."""
    import base64

    for page in debug_pages:
        attr = "original_image_base64" if kind == "original" else "rectified_image_base64"
        b64 = getattr(page, attr, None)
        if not b64:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(b64))
        return True
    return False


def worst_lines_report(texts: list[str], crop_paths: list[str], top_k: int = 5) -> list[dict[str, Any]]:
    scored: list[tuple[float, int, str, str]] = []
    for i, text in enumerate(texts):
        score = digit_noise_rate(text)
        if is_hallucinated_line(text):
            score += 1.0
        crop_path = crop_paths[i] if i < len(crop_paths) else ""
        scored.append((score, i, text, crop_path))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"line_index": i, "score": round(s, 4), "text": t, "crop_path": c}
        for s, i, t, c in scored[:top_k]
        if s > 0
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Main


def evaluate_image(
    image_path: Path,
    pipeline_off: OCRPipeline,
    pipeline_on: OCRPipeline,
    debug_dir: Path,
    image_id: str,
) -> dict[str, Any]:
    image_debug_dir = debug_dir / image_id
    image_debug_dir.mkdir(parents=True, exist_ok=True)

    raw_bytes = image_path.read_bytes()
    raw_image = Image.open(image_path).convert("RGB")
    raw_image.save(image_debug_dir / "original.jpg", format="JPEG", quality=85)

    # ── Run BEFORE (no rectification) ──
    t0 = time.perf_counter()
    debug_off = pipeline_off.process_bytes_debug(raw_bytes, image_path.name)
    elapsed_off_ms = (time.perf_counter() - t0) * 1000.0
    texts_off = line_texts(debug_off.result)
    write_text(image_debug_dir / "text_before.txt", "\n".join(texts_off))
    draw_bbox_overlay(raw_image, debug_off.debug_pages, image_debug_dir / "overlay_before.jpg")
    crops_off = save_debug_crops(debug_off.debug_pages, image_debug_dir / "crops_before")

    # ── Run AFTER (rectification on) ──
    t0 = time.perf_counter()
    debug_on = pipeline_on.process_bytes_debug(raw_bytes, image_path.name)
    elapsed_on_ms = (time.perf_counter() - t0) * 1000.0
    texts_on = line_texts(debug_on.result)
    write_text(image_debug_dir / "text_after.txt", "\n".join(texts_on))

    rect_saved = save_rectifier_image(debug_on.debug_pages, "rectified", image_debug_dir / "rectified.jpg")
    if rect_saved:
        rect_image = Image.open(image_debug_dir / "rectified.jpg").convert("RGB")
    else:
        rect_image = raw_image  # rectification was a no-op
    draw_bbox_overlay(rect_image, debug_on.debug_pages, image_debug_dir / "overlay_after.jpg")
    crops_on = save_debug_crops(debug_on.debug_pages, image_debug_dir / "crops_after")

    # ── Diagnostic metrics ──
    metrics_before = diagnostic_metrics(texts_off)
    metrics_after = diagnostic_metrics(texts_on)
    diff = diagnostic_diff(metrics_before, metrics_after)

    rect_meta: dict[str, Any] = {}
    for page in debug_on.debug_pages:
        rect_meta = {
            "rectifier_backend": page.rectifier_backend,
            "rectifier_applied": page.rectifier_applied,
            "rectifier_confidence": page.rectifier_confidence,
            "rectifier_diagnostics": page.rectifier_diagnostics,
        }
        break

    sample = {
        "image_id": image_id,
        "image_name": image_path.name,
        "image_path": str(image_path),
        "elapsed_ms_before": round(elapsed_off_ms, 2),
        "elapsed_ms_after": round(elapsed_on_ms, 2),
        "detection_count_before": int(getattr(debug_off.result, "total_lines", 0)),
        "detection_count_after": int(getattr(debug_on.result, "total_lines", 0)),
        "rectification": rect_meta,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "diff": diff,
        "verdict": diff.get("verdict"),
        "sample_text_before": "\n".join(texts_off[:10]),
        "sample_text_after": "\n".join(texts_on[:10]),
        "worst_lines_before": worst_lines_report(texts_off, crops_off),
        "worst_lines_after": worst_lines_report(texts_on, crops_on),
    }
    write_json(image_debug_dir / "metrics.json", sample)
    return sample


def aggregate_summary(per_image: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_image:
        return {"image_count": 0}
    keys = (
        "digit_noise_rate",
        "garbage_text_ratio",
        "high_digit_noise_line_rate",
        "abnormal_symbol_rate",
        "uppercase_garbage_token_rate",
        "suspicious_digit_token_count",
        "repeated_number_sequence_count",
        "hallucinated_line_rate",
        "hallucinated_line_count",
        "line_count",
    )

    def avg(field: str, source: str) -> float:
        vals = [float(item[source].get(field, 0.0) or 0.0) for item in per_image if source in item]
        return sum(vals) / len(vals) if vals else 0.0

    summary: dict[str, Any] = {
        "image_count": len(per_image),
        "rectifier_applied_rate": (
            sum(1 for item in per_image if item.get("rectification", {}).get("rectifier_applied"))
            / max(len(per_image), 1)
        ),
        "verdict_counts": {
            verdict: sum(1 for item in per_image if item.get("verdict") == verdict)
            for verdict in ("improved", "worsened", "no_change")
        },
        "before": {key: avg(key, "metrics_before") for key in keys},
        "after": {key: avg(key, "metrics_after") for key in keys},
    }
    summary["delta"] = {
        key: round(summary["after"][key] - summary["before"][key], 6)
        for key in keys
    }
    summary["hallucination_reduction_rate"] = (
        sum(
            1
            for item in per_image
            if item["metrics_after"].get("hallucinated_line_count", 0) <
               item["metrics_before"].get("hallucinated_line_count", 0)
        )
        / max(len(per_image), 1)
    )
    summary["hallucination_introduction_rate"] = (
        sum(
            1
            for item in per_image
            if item["metrics_after"].get("hallucinated_line_count", 0) >
               item["metrics_before"].get("hallucinated_line_count", 0)
        )
        / max(len(per_image), 1)
    )
    return summary


def render_summary_md(summary: dict[str, Any], per_image: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Phase 2C — Page-level rectification: before/after diagnostic report")
    lines.append("")
    lines.append(f"- Images evaluated: **{summary['image_count']}**")
    lines.append(f"- Rectifier applied rate: **{summary['rectifier_applied_rate']:.1%}**")
    lines.append("- Verdict counts:")
    for verdict, count in summary.get("verdict_counts", {}).items():
        lines.append(f"  - {verdict}: {count}")
    lines.append(f"- Hallucinated-line **reduction rate** (after < before): {summary['hallucination_reduction_rate']:.1%}")
    lines.append(f"- Hallucinated-line **introduction rate** (after > before): {summary['hallucination_introduction_rate']:.1%}")
    lines.append("")
    lines.append("## Aggregate diagnostic metrics")
    lines.append("")
    lines.append("| Metric | Before | After | Delta |")
    lines.append("|---|---:|---:|---:|")
    for key in summary.get("before", {}):
        b = summary["before"][key]
        a = summary["after"][key]
        d = summary["delta"][key]
        lines.append(f"| {key} | {b:.4f} | {a:.4f} | {d:+.4f} |")
    lines.append("")
    lines.append("## Per-image verdicts")
    lines.append("")
    lines.append("| # | Image | Verdict | Δ hallucinated_line_rate | Δ digit_noise_rate | Δ garbage_text_ratio |")
    lines.append("|---|---|---|---:|---:|---:|")
    for i, item in enumerate(per_image, start=1):
        d = item.get("diff", {})
        lines.append(
            "| {i} | `{name}` | **{v}** | {h:+.4f} | {dn:+.4f} | {gt:+.4f} |".format(
                i=i,
                name=item.get("image_name", ""),
                v=item.get("verdict", "?"),
                h=float(d.get("delta_hallucinated_line_rate", 0.0) or 0.0),
                dn=float(d.get("delta_digit_noise_rate", 0.0) or 0.0),
                gt=float(d.get("delta_garbage_text_ratio", 0.0) or 0.0),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2C rectification before/after eval")
    parser.add_argument("--images-dir", required=True, help="Directory of images (recursively scanned)")
    parser.add_argument(
        "--baseline-config",
        default=str(ROOT / "configs" / "phase2B_crop_geometry.yaml"),
        help="Config used for the BEFORE run (no rectification)",
    )
    parser.add_argument(
        "--rectified-config",
        default=str(ROOT / "configs" / "phase2C_rectified.yaml"),
        help="Config used for the AFTER run (rectification on)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "experiments" / "runs" / "phase2C_rectification_eval"),
        help="Where to save artifacts",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    images_root = Path(args.images_dir)
    if not images_root.exists():
        raise FileNotFoundError(f"--images-dir not found: {images_root}")
    image_paths = sorted(p for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {images_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    pipeline_off, config_off = build_pipeline(
        Path(args.baseline_config),
        override={"enable_document_perspective_correction": False},
    )
    pipeline_on, config_on = build_pipeline(Path(args.rectified_config))

    dump_experiment_config(config_off, output_dir / "config_before.yaml")
    dump_experiment_config(config_on, output_dir / "config_after.yaml")

    per_image_path = output_dir / "per_image_metrics.jsonl"
    per_image_path.write_text("", encoding="utf-8")

    per_image: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths):
        image_id = safe_id(image_path, index)
        logger.info("(%d/%d) %s", index + 1, len(image_paths), image_path.name)
        try:
            sample = evaluate_image(image_path, pipeline_off, pipeline_on, debug_dir, image_id)
        except Exception as exc:
            logger.exception("Failed on %s: %s", image_path, exc)
            sample = {
                "image_id": image_id,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "error": str(exc),
            }
        per_image.append(sample)
        append_jsonl(per_image_path, sample)

    summary = aggregate_summary([item for item in per_image if "error" not in item])
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(render_summary_md(summary, per_image), encoding="utf-8")
    print(f"Done. Output: {output_dir}")
    print(f"  - summary.md: {output_dir / 'summary.md'}")
    print(f"  - summary.json: {output_dir / 'summary.json'}")
    print(f"  - per_image_metrics.jsonl: {per_image_path}")


if __name__ == "__main__":
    main()
