"""
FastAPI application factory with lifespan management.

Key design decisions:
- Models are loaded ONCE at startup via lifespan (not per request).
- PaddleOCR and VietOCR are both heavy; lazy-loading is triggered here
  so the first API request is not slow.
- CORS is enabled so the Streamlit frontend can communicate freely
  during local development.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.dependencies import init_pipeline
from app.routes.ocr import router as ocr_router
from ocr_pipeline.config import settings

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Lifespan: load models on startup ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load OCR models at startup; clean up on shutdown."""
    default_model_key = settings.get_default_model_key()
    logger.info("=" * 60)
    logger.info("Starting Vietnamese Handwritten OCR API v%s", settings.api_version)
    logger.info("Default model: %s", default_model_key)
    logger.info("=" * 60)

    init_pipeline()

    logger.info("API ready — listening on %s:%d", settings.api_host, settings.api_port)
    yield
    # Shutdown
    logger.info("Shutting down OCR API.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=(
            "Production OCR API for Vietnamese handwritten text recognition.\n\n"
            "**Default model:** Experiment B (50k, best)\n\n"
            "Upload one or more images / PDFs and receive structured results "
            "with bounding boxes, confidence scores, and full text."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS — allow Streamlit frontend ───────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # Tighten for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(ocr_router, prefix="")

    return app


# App instance — used by uvicorn and tests
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,   # Disable reload in production (models would reload)
        log_level="info",
    )
