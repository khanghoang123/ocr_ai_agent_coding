"""OCR validation helpers for production-readiness checks and experiments."""

from ocr_pipeline.validation.harness import run_all_validations
from ocr_pipeline.validation.metrics import (
    auto_research_score,
    character_error_rate,
    digit_noise_rate,
    normalized_line_count_error,
    word_error_rate,
)
from ocr_pipeline.validation.debug_analyzer import analyze_debug_dir, write_debug_analysis

__all__ = [
    "auto_research_score",
    "character_error_rate",
    "digit_noise_rate",
    "normalized_line_count_error",
    "run_all_validations",
    "word_error_rate",
    "analyze_debug_dir",
    "write_debug_analysis",
]
