"""Pytest configuration: add src/ to sys.path for package imports."""
import sys
from pathlib import Path

# Allow `from ocr_pipeline.xxx import yyy` in tests without pip install
src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
