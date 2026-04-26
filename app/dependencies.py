"""
Dependency injection for FastAPI.

Manages a single OCRPipeline instance per process (singleton pattern).
Models are loaded once at startup via the lifespan handler in main.py,
avoiding the high cost of loading VietOCR on every request.
"""

from __future__ import annotations

import logging
from typing import Optional

from ocr_pipeline.pipeline import OCRPipeline

logger = logging.getLogger(__name__)

# Module-level singleton — set by lifespan on startup
_pipeline: Optional[OCRPipeline] = None


def init_pipeline(model_key: Optional[str] = None) -> None:
    """Initialize the global OCRPipeline singleton.

    Called once during application startup (lifespan event).
    """
    global _pipeline
    logger.info("Initializing OCR pipeline (model=%s) ...", model_key or "default")
    _pipeline = OCRPipeline.from_settings(model_key=model_key)
    # Eagerly trigger lazy model loads so the first request isn't slow
    logger.info("OCR pipeline ready — model '%s' loaded.", _pipeline.model_key)


def get_pipeline(model_key: Optional[str] = None) -> OCRPipeline:
    """Return the global singleton pipeline.

    If model_key matches the loaded model, return the singleton.
    If a different model is requested, create a temporary pipeline for
    that request (avoids reloading the main model).
    """
    global _pipeline

    if _pipeline is None:
        # Fallback: lazy init if called before lifespan (e.g. in tests)
        init_pipeline(model_key)
        return _pipeline

    if model_key and model_key != _pipeline.model_key:
        logger.info("Creating per-request pipeline for model '%s'.", model_key)
        return OCRPipeline.from_settings(model_key=model_key)

    return _pipeline
