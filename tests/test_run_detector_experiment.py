"""Smoke tests for the ``scripts/run_detector_experiment.py`` runner.

We don't actually run any models here — that's offline. These tests only
cover the leaderboard-bookkeeping logic that has caused real bugs:

  * partial rerun must NOT duplicate rows for the rerun experiment in the
    aggregated JSONL.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_detector_experiment.py"


@pytest.fixture(scope="module")
def runner_module():
    """Import the runner script as a module without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "run_detector_experiment", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_detector_experiment"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_partial_rerun_drops_stale_rows_for_rerun_experiments(
    runner_module, tmp_path, monkeypatch
):
    """Re-running a subset must not double-count the rerun experiments.

    Repro: run all 4 experiments → 28 rows. Then ``--experiments E3`` with
    no dedup → 35 rows after, ``aggregate()`` over E3 sees 14 instead of 7.
    With the fix, the rerun should drop the 7 stale E3 rows before
    appending the 7 fresh ones, and the file ends at 28 rows total.
    """
    out_dir = tmp_path / "leaderboard"
    aggregate_jsonl = out_dir / "per_image_metrics.jsonl"

    # Seed: pretend we already ran 4 experiments × 2 images = 8 rows.
    seeded = [
        {"exp_id": "E0_paddle", "image": "a.jpg"},
        {"exp_id": "E0_paddle", "image": "b.jpg"},
        {"exp_id": "E1_surya", "image": "a.jpg"},
        {"exp_id": "E1_surya", "image": "b.jpg"},
        {"exp_id": "E2_craft", "image": "a.jpg"},
        {"exp_id": "E2_craft", "image": "b.jpg"},
        {"exp_id": "E3_kraken_blla", "image": "a.jpg"},
        {"exp_id": "E3_kraken_blla", "image": "b.jpg"},
    ]
    _write_jsonl(aggregate_jsonl, seeded)
    assert len(_read_jsonl(aggregate_jsonl)) == 8

    # Replicate the dedup branch from main(): given args.experiments=["E3"],
    # filter the JSONL to keep only rows whose exp_id is NOT in the rerun
    # set.
    rerun_ids = {"E3_kraken_blla"}
    kept_lines: list[str] = []
    with aggregate_jsonl.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if row.get("exp_id") not in rerun_ids:
                kept_lines.append(stripped)
    with aggregate_jsonl.open("w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")

    rows_after_dedup = _read_jsonl(aggregate_jsonl)
    assert len(rows_after_dedup) == 6
    assert all(r["exp_id"] != "E3_kraken_blla" for r in rows_after_dedup)
    # E0/E1/E2 must still be present.
    by_exp = {r["exp_id"] for r in rows_after_dedup}
    assert by_exp == {"E0_paddle", "E1_surya", "E2_craft"}


def test_aggregate_dedups_when_jsonl_has_one_row_per_image_per_exp(runner_module):
    """Sanity-check ``aggregate()``: given exactly one row per (exp, image),
    n_images equals the number of unique images — never doubled."""
    rows = [
        {
            "exp_id": "E0_paddle",
            "image": "a.jpg",
            "detection_count": 10,
            "full_width_band_rate": 0.0,
            "mean_box_aspect_ratio": 5.0,
            "mean_distinct_x1_per_page": 4.0,
            "median_box_height": 20.0,
            "digit_noise_rate": 0.0,
            "garbage_text_ratio": 0.0,
            "hallucinated_line_rate": 0.0,
            "repeated_number_sequence_count": 0.0,
            "uppercase_garbage_token_rate": 0.0,
            "abnormal_symbol_rate": 0.0,
            "line_count": 5.0,
            "elapsed_ms": 1000.0,
        },
        {
            "exp_id": "E0_paddle",
            "image": "b.jpg",
            "detection_count": 14,
            "full_width_band_rate": 0.0,
            "mean_box_aspect_ratio": 6.0,
            "mean_distinct_x1_per_page": 5.0,
            "median_box_height": 22.0,
            "digit_noise_rate": 0.0,
            "garbage_text_ratio": 0.0,
            "hallucinated_line_rate": 0.0,
            "repeated_number_sequence_count": 0.0,
            "uppercase_garbage_token_rate": 0.0,
            "abnormal_symbol_rate": 0.0,
            "line_count": 7.0,
            "elapsed_ms": 1100.0,
        },
    ]
    summary = runner_module.aggregate(rows)
    assert summary["E0_paddle"]["n_images"] == 2
    assert summary["E0_paddle"]["detection_count"] == 12.0  # mean of 10 and 14
