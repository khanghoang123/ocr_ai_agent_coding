#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_pipeline.experiment_config import load_experiment_config  # noqa: E402
from ocr_pipeline.validation.metrics import auto_research_score  # noqa: E402

CONFIG_DIR = ROOT / "configs"
RUNS_DIR = ROOT / "experiments" / "runs"
LEADERBOARD_DIR = ROOT / "experiments" / "leaderboard"
LEADERBOARD_CSV = LEADERBOARD_DIR / "leaderboard.csv"
LEADERBOARD_JSON = LEADERBOARD_DIR / "leaderboard.json"
REPORT_PATH = ROOT / "REPORT.md"

CROP_PADDING_VALUES = [0.03, 0.05, 0.08, 0.12, 0.15]
UPSCALE_VALUES = [1.0, 1.5, 2.0]
BOOLEAN_KEYS = [
    "rectify_line",
    "enable_deskew",
    "enable_document_perspective_correction",
    "enable_local_contrast",
    "detect_on_upscaled_image",
    "enable_vietnamese_postprocess",
    "flag_digit_noise",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def discover_configs() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.yaml"))


def run_config(config_path: Path, dataset: str, limit: int | None, run_root: Path) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = run_root / f"{stamp}_{config.name}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_ocr_experiment.py"),
        "--config",
        str(config_path),
        "--dataset",
        dataset,
        "--output-dir",
        str(out_dir),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {
            "config_name": config.name,
            "processed_count": 0,
            "error_count": 1,
            "score": None,
            "unsupported_options": config.unsupported_options,
        }
    metrics.update(
        {
            "config_path": str(config_path),
            "run_dir": str(out_dir),
            "return_code": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    )
    if metrics.get("score") is None:
        metrics["score"] = auto_research_score(metrics)
    return metrics


def read_leaderboard() -> list[dict[str, Any]]:
    if not LEADERBOARD_JSON.exists():
        return []
    with LEADERBOARD_JSON.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, list) else []


def write_leaderboard(rows: list[dict[str, Any]]) -> None:
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: float("inf") if row.get("score") is None else float(row["score"]))
    LEADERBOARD_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "rank",
        "config_name",
        "score",
        "cer",
        "wer",
        "digit_noise_rate",
        "normalized_line_count_error",
        "detection_count",
        "crop_flag_count",
        "processed_count",
        "error_count",
        "config_path",
        "run_dir",
    ]
    with LEADERBOARD_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = {key: row.get(key) for key in fieldnames}
            out["rank"] = rank
            writer.writerow(out)


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [row for row in rows if row.get("score") is not None]
    if not scored:
        return None
    return min(scored, key=lambda row: float(row["score"]))


def perturb_config(best: dict[str, Any], iteration: int) -> Path:
    source_path = Path(best["config_path"])
    payload = load_yaml(source_path)
    payload["name"] = f"auto_iter_{iteration:03d}_{payload.get('name', source_path.stem)}"
    payload["description"] = f"AutoResearch perturbation from {source_path.name}"

    current_padding = float(payload.get("crop_padding_ratio") or 0.08)
    padding_index = CROP_PADDING_VALUES.index(current_padding) if current_padding in CROP_PADDING_VALUES else 2
    payload["crop_padding_ratio"] = CROP_PADDING_VALUES[(padding_index + 1) % len(CROP_PADDING_VALUES)]

    current_upscale = float(payload.get("upscale_factor") or 1.0)
    upscale_index = UPSCALE_VALUES.index(current_upscale) if current_upscale in UPSCALE_VALUES else 0
    payload["upscale_factor"] = UPSCALE_VALUES[(upscale_index + (1 if iteration % 2 == 0 else 0)) % len(UPSCALE_VALUES)]

    key = BOOLEAN_KEYS[iteration % len(BOOLEAN_KEYS)]
    payload[key] = not bool(payload.get(key, False))
    target = CONFIG_DIR / f"auto_iter_{iteration:03d}.yaml"
    dump_yaml(target, payload)
    return target


def update_report(rows: list[dict[str, Any]], dataset: str, iterations: int) -> None:
    best = best_row(rows)
    leaderboard_lines = []
    for rank, row in enumerate(rows[:10], start=1):
        leaderboard_lines.append(
            f"| {rank} | {row.get('config_name')} | {row.get('score')} | "
            f"{row.get('cer')} | {row.get('wer')} | {row.get('run_dir')} |"
        )
    if not leaderboard_lines:
        leaderboard_lines.append("| - | - | - | - | - | - |")

    best_config = best.get("config_path") if best else "N/A"
    best_score = best.get("score") if best else "N/A"
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# AutoResearch Report",
                "",
                "## Mục tiêu",
                "Tự động tạo hypothesis, chạy OCR experiment, đo metric, so sánh kết quả và sinh config mới cho OCR tiếng Việt viết tay.",
                "",
                "## Pipeline hiện tại",
                "File/PDF -> PaddleOCR detector -> LineRefiner -> LineCropper -> VietOCR -> LayoutReconstructor -> optional safe Vietnamese postprocess.",
                "",
                "## Cách chạy baseline",
                "```bash",
                "python scripts/run_ocr_experiment.py --config configs/baseline.yaml --dataset data/processed/val.txt --limit 10 --output-dir experiments/runs/baseline_smoke",
                "```",
                "",
                "## Cách chạy AutoResearch",
                "```bash",
                "python scripts/auto_research.py --dataset data/processed/val.txt --limit 10 --iterations 5",
                "```",
                "",
                "## Leaderboard",
                "| Rank | Config | Score | CER | WER | Run |",
                "|---:|---|---:|---:|---:|---|",
                *leaderboard_lines,
                "",
                "## Best config hiện tại",
                f"- Config: `{best_config}`",
                f"- Score: `{best_score}`",
                "",
                "## Metric trước/sau",
                "So sánh trước/sau nằm trong `experiments/leaderboard/leaderboard.csv` sau mỗi lần chạy.",
                "",
                "## Lỗi còn lại",
                "- `enable_deskew` và `enable_document_perspective_correction` được track nhưng chưa implement đầy đủ; runner ghi trong `unsupported_options`.",
                "- CER/WER là `null` nếu dataset không có ground truth text.",
                "",
                "## Next steps",
                "- Chạy AutoResearch với `--limit` lớn hơn trên validation set ổn định.",
                "- Implement deskew/perspective correction khi có bộ page-level benchmark đủ ground truth.",
                f"- Dataset lần cập nhật gần nhất: `{dataset}`; iterations requested: `{iterations}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def auto_research(dataset: str, limit: int | None, iterations: int) -> list[dict[str, Any]]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    run_root = RUNS_DIR / time.strftime("autoresearch_%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    rows = read_leaderboard()
    evaluated_configs = {row.get("config_path") for row in rows}

    for config_path in discover_configs():
        if str(config_path) in evaluated_configs:
            continue
        rows.append(run_config(config_path, dataset, limit, run_root))
        write_leaderboard(rows)

    for iteration in range(1, iterations + 1):
        best = best_row(rows)
        if best is None:
            break
        candidate = perturb_config(best, iteration)
        rows.append(run_config(candidate, dataset, limit, run_root))
        write_leaderboard(rows)

    rows = sorted(rows, key=lambda row: float("inf") if row.get("score") is None else float(row["score"]))
    update_report(rows, dataset=dataset, iterations=iterations)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AutoResearch OCR loop.")
    parser.add_argument("--dataset", default="data/processed/val.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = auto_research(args.dataset, args.limit, args.iterations)
    print(json.dumps(rows[:10], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
