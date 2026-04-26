#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_pipeline.validation.debug_analyzer import write_debug_analysis  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze OCR debug crops and bbox geometry.")
    parser.add_argument("--debug-dir", required=True, help="Directory containing debug_mapping.json.")
    parser.add_argument("--output", required=True, help="Path to write debug analysis JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = write_debug_analysis(args.debug_dir, args.output)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
