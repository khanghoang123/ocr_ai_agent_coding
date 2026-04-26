from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_pipeline.detector.paddle_detector import PaddleDetector


@dataclass
class TrialMetrics:
    db_thresh: float
    box_thresh: float
    unclip_ratio: float
    aggregate_score: float
    min_line_count: int
    mean_line_count: float
    details: list[dict]


def _score_one_image(num_boxes: int, diagnostics: dict, image_height: int) -> float:
    median_height = float(diagnostics.get("median_height", 0.0))
    coverage_ratio = float(diagnostics.get("coverage_ratio", 0.0))
    height_ratio = median_height / max(float(image_height), 1.0)

    # Prefer higher recall, but penalize obviously merged lines (too tall boxes).
    score = float(num_boxes)
    score += min(coverage_ratio, 1.2) * 5.0
    if height_ratio > 0.055:
        score -= (height_ratio - 0.055) * 120.0
    return score


def _draw_overlay(image: Image.Image, polygons: list[np.ndarray], out_path: Path) -> None:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    for poly in polygons:
        points = [(float(p[0]), float(p[1])) for p in poly.reshape(-1, 2)]
        draw.polygon(points, outline="#ef4444", width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_path)


def _evaluate_combo(
    images: list[tuple[Path, Image.Image]],
    db_thresh: float,
    box_thresh: float,
    unclip_ratio: float,
    output_dir: Path | None,
) -> TrialMetrics:
    detector = PaddleDetector(
        use_gpu=False,
        db_thresh=db_thresh,
        db_box_thresh=box_thresh,
        unclip_ratio=unclip_ratio,
    )

    details: list[dict] = []
    scores: list[float] = []
    line_counts: list[int] = []

    for image_path, image in images:
        result = detector.detect_with_notebook_fallback(image)
        count = int(result.num_boxes)
        line_counts.append(count)
        score = _score_one_image(count, result.diagnostics, image.height)
        scores.append(score)

        details.append(
            {
                "image": str(image_path),
                "line_count": count,
                "diagnostics": result.diagnostics,
            }
        )

        if output_dir is not None:
            stem = image_path.stem.replace(" ", "_")
            fname = f"{stem}_t{db_thresh:.2f}_b{box_thresh:.2f}_u{unclip_ratio:.1f}.jpg"
            _draw_overlay(image, result.polygons, output_dir / fname)

    return TrialMetrics(
        db_thresh=float(db_thresh),
        box_thresh=float(box_thresh),
        unclip_ratio=float(unclip_ratio),
        aggregate_score=float(np.mean(scores)) if scores else 0.0,
        min_line_count=min(line_counts) if line_counts else 0,
        mean_line_count=float(np.mean(line_counts)) if line_counts else 0.0,
        details=details,
    )


def _build_fine_grid(center: float, lower: float, upper: float, step: float) -> list[float]:
    values = []
    cur = max(lower, center - step)
    stop = min(upper, center + step) + 1e-9
    while cur <= stop:
        values.append(round(cur, 3))
        cur += step / 2.0
    return sorted(set(values))


def tune(
    image_paths: list[Path],
    output_dir: Path | None = None,
) -> dict:
    images = [(path, Image.open(path).convert("RGB")) for path in image_paths]

    coarse_t = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    coarse_b = [0.08, 0.10, 0.12, 0.15, 0.18]
    coarse_u = [1.6, 1.8, 2.0]

    coarse_trials: list[TrialMetrics] = []
    for t in coarse_t:
        for b in coarse_b:
            for u in coarse_u:
                coarse_trials.append(_evaluate_combo(images, t, b, u, output_dir))

    coarse_trials.sort(
        key=lambda item: (item.aggregate_score, item.min_line_count, item.mean_line_count),
        reverse=True,
    )
    best_coarse = coarse_trials[0]

    fine_t = _build_fine_grid(best_coarse.db_thresh, 0.05, 0.25, 0.04)
    fine_b = _build_fine_grid(best_coarse.box_thresh, 0.06, 0.25, 0.04)
    fine_u = sorted(set([max(1.4, best_coarse.unclip_ratio - 0.2), best_coarse.unclip_ratio, min(2.2, best_coarse.unclip_ratio + 0.2)]))

    fine_trials: list[TrialMetrics] = []
    for t in fine_t:
        for b in fine_b:
            for u in fine_u:
                fine_trials.append(_evaluate_combo(images, t, b, u, output_dir))

    fine_trials.sort(
        key=lambda item: (item.aggregate_score, item.min_line_count, item.mean_line_count),
        reverse=True,
    )
    best_fine = fine_trials[0]

    return {
        "images": [str(path) for path in image_paths],
        "recommended": asdict(best_fine),
        "top5_coarse": [asdict(item) for item in coarse_trials[:5]],
        "top5_fine": [asdict(item) for item in fine_trials[:5]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune PaddleOCR detection parameters for line recall.")
    parser.add_argument("--images", nargs="+", required=True, help="Image paths to tune on.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "output" / "detector_tuning"),
        help="Directory to save visualization overlays.",
    )
    parser.add_argument(
        "--report-path",
        default=str(ROOT / "tests" / "output" / "detector_tuning" / "report.json"),
        help="Path to save JSON report.",
    )
    args = parser.parse_args()

    image_paths = [Path(p).resolve() for p in args.images]
    for path in image_paths:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = tune(image_paths, output_dir=output_dir)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rec = report["recommended"]
    print("Recommended detector parameters:")
    print(f"  det_db_thresh={rec['db_thresh']}")
    print(f"  det_db_box_thresh={rec['box_thresh']}")
    print(f"  det_db_unclip_ratio={rec['unclip_ratio']}")
    print(f"  aggregate_score={rec['aggregate_score']:.3f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
