"""Evaluate detector-leaderboard runs against line-level GT in tests/test_gt/.

For every <run-dir>/<experiment>/debug/<image_id>/text.txt produced by
``scripts/run_detector_experiment.py``, this script:

1. Strips the leading numeric prefix from ``image_id`` and joins to GT by
   stem (e.g. ``0006_thumb_1200_1698`` -> ``thumb_1200_1698`` ->
   ``tests/test_gt/thumb_1200_1698.json``). Images without GT are skipped.
2. Computes two metric families per matched (experiment, image) pair:

   - **Page-level CER/WER** — concatenate predicted lines (in reading order
     as written by the runner) and GT lines into a single page string, then
     compute CER / WER with NFC + collapsed-whitespace normalisation
     (matching ``src/ocr_pipeline/validation/metrics.py``).
   - **Line-level matched CER/WER** — minimum-cost line assignment via the
     Hungarian algorithm with edit-distance cost; unmatched lines (missed
     detections or false positives) are penalised with ``unmatched_penalty``
     (defaults to ``1.0`` CER, i.e. fully-wrong line).

3. Writes a JSON summary (``--out`` / defaults to ``cer_wer.json`` under the
   run dir) and a Markdown table (``cer_wer.md``).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running without `pip install -e .` -- script lives next to src/.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ocr_pipeline.validation.metrics import (  # noqa: E402
    _edit_distance,
    character_error_rate,
    normalize_for_metric,
    word_error_rate,
)

logger = logging.getLogger("eval_cer_wer")

_IMAGE_ID_PREFIX = re.compile(r"^\d+_(.*)$")


@dataclass(frozen=True)
class GTLine:
    text: str


@dataclass(frozen=True)
class GroundTruth:
    image: str
    quality: str
    notes: str
    lines: list[GTLine]

    @property
    def stem(self) -> str:
        return Path(self.image).stem


def load_ground_truth(gt_dir: Path) -> dict[str, GroundTruth]:
    """Load every ``*.json`` GT file under ``gt_dir`` keyed by image stem."""
    result: dict[str, GroundTruth] = {}
    for path in sorted(gt_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed GT %s: %s", path, exc)
            continue
        image = payload.get("image") or path.stem
        quality = payload.get("ground_truth_quality", "unknown")
        notes = payload.get("notes", "")
        lines = [GTLine(text=item["text"]) for item in payload.get("lines", [])]
        gt = GroundTruth(image=image, quality=quality, notes=notes, lines=lines)
        result[gt.stem] = gt
    return result


def _strip_image_id_prefix(image_id: str) -> str:
    match = _IMAGE_ID_PREFIX.match(image_id)
    return match.group(1) if match else image_id


def _read_predictions(text_file: Path) -> list[str]:
    """Read one prediction per line, drop empty lines.

    The detector runner writes one line per detected box; blank entries are
    artefacts of trailing newlines and are excluded.
    """
    if not text_file.exists():
        return []
    raw = text_file.read_text(encoding="utf-8").splitlines()
    return [line for line in (s.strip() for s in raw) if line]


def _hungarian(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Square-pad cost matrix and solve via scipy if available, else greedy.

    Returns list of (row, col) pairs covering ``min(rows, cols)`` matches.
    """
    rows = len(cost)
    cols = len(cost[0]) if rows else 0
    if rows == 0 or cols == 0:
        return []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        matrix = np.array(cost, dtype=float)
        r, c = linear_sum_assignment(matrix)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        # Greedy fallback -- pick globally-cheapest pair iteratively.
        used_r: set[int] = set()
        used_c: set[int] = set()
        flat = sorted(
            (
                (cost[i][j], i, j)
                for i in range(rows)
                for j in range(cols)
            ),
            key=lambda x: x[0],
        )
        out: list[tuple[int, int]] = []
        for _, i, j in flat:
            if i in used_r or j in used_c:
                continue
            used_r.add(i)
            used_c.add(j)
            out.append((i, j))
        return out


def _line_level_cer_wer(
    predictions: list[str],
    references: list[str],
    unmatched_penalty: float = 1.0,
) -> dict[str, float]:
    """Compute Hungarian-matched line-level CER/WER.

    Unmatched predictions (false positives) contribute ``unmatched_penalty``
    CER each; unmatched references (missed lines) likewise. Line-level CER
    therefore aggregates per-line errors over a denominator of
    ``max(len(predictions), len(references))`` when normalised.
    """
    pred_norm = [normalize_for_metric(p) for p in predictions]
    ref_norm = [normalize_for_metric(r) for r in references]
    if not pred_norm and not ref_norm:
        return {
            "matched_pairs": 0,
            "unmatched_pred": 0,
            "unmatched_ref": 0,
            "line_cer": 0.0,
            "line_wer": 0.0,
        }
    if not pred_norm or not ref_norm:
        return {
            "matched_pairs": 0,
            "unmatched_pred": len(pred_norm),
            "unmatched_ref": len(ref_norm),
            "line_cer": float(unmatched_penalty),
            "line_wer": float(unmatched_penalty),
        }

    cost: list[list[float]] = []
    for p in pred_norm:
        row = []
        for r in ref_norm:
            denom = max(1, len(r))
            row.append(_edit_distance(p, r) / denom)
        cost.append(row)

    pairs = _hungarian(cost)
    matched_cer: list[float] = []
    matched_wer: list[float] = []
    matched_p: set[int] = set()
    matched_r: set[int] = set()
    for i, j in pairs:
        c = character_error_rate(pred_norm[i], ref_norm[j])
        w = word_error_rate(pred_norm[i], ref_norm[j])
        matched_cer.append(c)
        matched_wer.append(w)
        matched_p.add(i)
        matched_r.add(j)

    unmatched_p = len(pred_norm) - len(matched_p)
    unmatched_r = len(ref_norm) - len(matched_r)
    total = len(matched_cer) + unmatched_p + unmatched_r
    cer = (sum(matched_cer) + unmatched_penalty * (unmatched_p + unmatched_r)) / max(1, total)
    wer = (sum(matched_wer) + unmatched_penalty * (unmatched_p + unmatched_r)) / max(1, total)
    return {
        "matched_pairs": len(matched_cer),
        "unmatched_pred": unmatched_p,
        "unmatched_ref": unmatched_r,
        "line_cer": cer,
        "line_wer": wer,
    }


def evaluate_run(run_dir: Path, gt: dict[str, GroundTruth]) -> dict:
    """Walk every experiment subdir and compute per-image + aggregate metrics."""
    experiments: dict[str, dict] = {}
    for exp_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("E")):
        exp_id = exp_dir.name
        debug = exp_dir / "debug"
        if not debug.is_dir():
            logger.info("[%s] no debug dir, skipping", exp_id)
            continue
        per_image: list[dict] = []
        for image_dir in sorted(p for p in debug.iterdir() if p.is_dir()):
            stem = _strip_image_id_prefix(image_dir.name)
            ground = gt.get(stem)
            if ground is None:
                continue
            predictions = _read_predictions(image_dir / "text.txt")
            references = [line.text for line in ground.lines]
            page_pred = " ".join(predictions)
            page_ref = " ".join(references)
            page_cer = character_error_rate(page_pred, page_ref)
            page_wer = word_error_rate(page_pred, page_ref)
            line_metrics = _line_level_cer_wer(predictions, references)
            per_image.append({
                "image": ground.image,
                "image_stem": stem,
                "ground_truth_quality": ground.quality,
                "predicted_lines": len(predictions),
                "reference_lines": len(references),
                "page_cer": page_cer,
                "page_wer": page_wer,
                **line_metrics,
            })
        if not per_image:
            continue
        verified = [r for r in per_image if r["ground_truth_quality"] == "verified"]
        all_rows = per_image
        experiments[exp_id] = {
            "per_image": per_image,
            "aggregate_all": _aggregate(all_rows),
            "aggregate_verified_only": _aggregate(verified) if verified else None,
        }
    return {
        "run_dir": str(run_dir),
        "experiments": experiments,
        "ground_truth_files": [
            {"image": g.image, "quality": g.quality, "lines": len(g.lines)}
            for g in sorted(gt.values(), key=lambda v: v.image)
        ],
    }


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"page_cer": math.nan, "page_wer": math.nan, "line_cer": math.nan, "line_wer": math.nan, "n": 0}
    return {
        "page_cer": sum(r["page_cer"] for r in rows) / len(rows),
        "page_wer": sum(r["page_wer"] for r in rows) / len(rows),
        "line_cer": sum(r["line_cer"] for r in rows) / len(rows),
        "line_wer": sum(r["line_wer"] for r in rows) / len(rows),
        "n": len(rows),
    }


def _format_summary_md(report: dict) -> str:
    lines = [
        f"# CER/WER evaluation: `{report['run_dir']}`",
        "",
        "## Aggregate (verified GT only)",
        "",
        "| experiment | n | page_cer | page_wer | line_cer | line_wer |",
        "|---|---|---|---|---|---|",
    ]
    for exp_id, data in report["experiments"].items():
        agg = data["aggregate_verified_only"]
        if agg is None:
            continue
        lines.append(
            f"| {exp_id} | {agg['n']} | {agg['page_cer']:.4f} | {agg['page_wer']:.4f} "
            f"| {agg['line_cer']:.4f} | {agg['line_wer']:.4f} |"
        )
    lines += [
        "",
        "## Aggregate (all GT, including best_effort/partial)",
        "",
        "| experiment | n | page_cer | page_wer | line_cer | line_wer |",
        "|---|---|---|---|---|---|",
    ]
    for exp_id, data in report["experiments"].items():
        agg = data["aggregate_all"]
        lines.append(
            f"| {exp_id} | {agg['n']} | {agg['page_cer']:.4f} | {agg['page_wer']:.4f} "
            f"| {agg['line_cer']:.4f} | {agg['line_wer']:.4f} |"
        )
    lines += ["", "## Per-image (page-level CER)", "", "| image | quality |"]
    exp_ids = list(report["experiments"].keys())
    lines[-1] = "| image | quality | " + " | ".join(exp_ids) + " |"
    lines.append("|---|---| " + " | ".join("---" for _ in exp_ids) + " |")

    images = sorted({
        r["image_stem"]
        for data in report["experiments"].values()
        for r in data["per_image"]
    })
    for stem in images:
        first = next(
            (
                r
                for data in report["experiments"].values()
                for r in data["per_image"]
                if r["image_stem"] == stem
            ),
            None,
        )
        quality = first["ground_truth_quality"] if first else "?"
        cells = []
        for exp_id in exp_ids:
            row = next(
                (r for r in report["experiments"][exp_id]["per_image"] if r["image_stem"] == stem),
                None,
            )
            cells.append(f"{row['page_cer']:.4f}" if row else "—")
        lines.append(f"| {stem} | {quality} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Directory under experiments/runs/ produced by run_detector_experiment.py.")
    parser.add_argument("--gt-dir", default=Path("tests/test_gt"), type=Path,
                        help="Directory of ground-truth JSON files (default tests/test_gt).")
    parser.add_argument("--out", default=None, type=Path,
                        help="Output JSON path. Defaults to <run-dir>/cer_wer.json.")
    parser.add_argument("--md-out", default=None, type=Path,
                        help="Output Markdown path. Defaults to <run-dir>/cer_wer.md.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if not args.run_dir.is_dir():
        parser.error(f"--run-dir does not exist or is not a directory: {args.run_dir}")
    if not args.gt_dir.is_dir():
        parser.error(f"--gt-dir does not exist or is not a directory: {args.gt_dir}")

    gt = load_ground_truth(args.gt_dir)
    if not gt:
        parser.error(f"No GT files found under {args.gt_dir}")

    report = evaluate_run(args.run_dir, gt)
    out_json = args.out or (args.run_dir / "cer_wer.json")
    out_md = args.md_out or (args.run_dir / "cer_wer.md")
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(_format_summary_md(report), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
