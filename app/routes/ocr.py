"""
FastAPI OCR routes.

Endpoints:
  POST /ocr        — Accept one or more files, return JSON OCR results
  POST /ocr/export — Accept one or more files, return downloadable file
  GET  /health     — Service health + model status
"""

from __future__ import annotations

import io
import logging
import zipfile
from typing import Annotated, Optional
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response, StreamingResponse

from ocr_pipeline.exporters import export_result, suggested_filename
from ocr_pipeline.schemas import (
    OCRDebugResponse,
    ExportFormat,
    HealthResponse,
    ModelCatalogResponse,
    OCRResponse,
    OCRResult,
)

from app.dependencies import get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["utility"],
)
async def health() -> HealthResponse:
    """Returns model status, device info, and API version.

    The `torch` import is guarded so `/health` keeps working on lightweight
    deployments (e.g. the CI quality-gate image defined in
    `requirements-ci.txt`) that intentionally exclude the heavy recognizer
    stack. When torch is unavailable we report `gpu_available=False`
    instead of crashing with `ModuleNotFoundError`.
    """
    from ocr_pipeline.config import settings

    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False

    pipeline = get_pipeline()

    return HealthResponse(
        status="ok",
        model_loaded=pipeline.model_key,
        device=settings.rec_device,
        gpu_available=gpu_available,
        version=settings.api_version,
    )


@router.get(
    "/models",
    response_model=ModelCatalogResponse,
    summary="List inference models available to the UI",
    tags=["utility"],
)
async def list_models() -> ModelCatalogResponse:
    from ocr_pipeline.config import settings

    return ModelCatalogResponse(
        default_model=settings.get_default_model_key(),
        models=settings.get_model_catalog(),
    )


# ── OCR — JSON response ────────────────────────────────────────────────────────

@router.post(
    "/ocr",
    response_model=OCRResponse,
    summary="Run OCR on one or more files",
    tags=["ocr"],
)
async def run_ocr(
    files: Annotated[
        list[UploadFile],
        File(description="One or more image (.jpg/.png) or PDF files"),
    ],
    model: Optional[str] = Form(default=None),
    postprocess: bool = Form(default=False),
) -> OCRResponse:
    """Process uploaded files through the OCR pipeline.

    - Accepts: JPEG, PNG, PDF
    - Returns: Structured JSON with lines, bboxes, and confidence scores
    - Model: use `model` param to switch (default = experiment_B)
    """
    from ocr_pipeline.config import settings

    _validate_files(files, settings.max_upload_size)
    model_key = _validate_model_key(model)

    pipeline = get_pipeline(model_key=model_key)
    post_processor = None
    if postprocess:
        try:
            from ocr_pipeline.post_processing import (
                PostProcessError,
                get_post_processor,
            )
            post_processor = get_post_processor()
        except PostProcessError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            )
    results: list[OCRResult] = []

    for upload in files:
        data = await upload.read()
        try:
            ocr_result = pipeline.process_bytes(data, upload.filename or "upload")
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e),
            )
        except Exception as e:
            logger.exception("OCR failed for file '%s'", upload.filename)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"OCR processing failed: {e}",
            )
        if post_processor is not None:
            ocr_result = _apply_postprocess(ocr_result, post_processor)
        results.append(ocr_result)

    return OCRResponse(status="success", results=results)


@router.post(
    "/ocr/debug",
    response_model=OCRDebugResponse,
    summary="Run OCR on a single file and return debug artefacts",
    tags=["ocr"],
)
async def run_ocr_debug(
    files: Annotated[
        list[UploadFile],
        File(description="A single image (.jpg/.png) or PDF file"),
    ],
    model: Optional[str] = Form(default=None),
) -> OCRDebugResponse:
    from ocr_pipeline.config import settings

    _validate_files(files, settings.max_upload_size)
    if len(files) != 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Debug OCR accepts exactly one file per request.",
        )

    model_key = _validate_model_key(model)
    upload = files[0]
    pipeline = get_pipeline(model_key=model_key)
    data = await upload.read()

    try:
        debug_item = pipeline.process_bytes_debug(data, upload.filename or "upload")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Debug OCR failed for '%s'", upload.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debug OCR processing failed: {e}",
        )

    return OCRDebugResponse(
        status="success",
        results=[debug_item.result],
        debug_results=[debug_item],
    )


# ── OCR/Export — file download ────────────────────────────────────────────────

@router.post(
    "/ocr/export",
    summary="Run OCR and download result as a file",
    tags=["ocr"],
)
async def run_ocr_export(
    files: Annotated[
        list[UploadFile],
        File(description="One or more image (.jpg/.png) or PDF files"),
    ],
    fmt: ExportFormat = Form(default=ExportFormat.TXT),
    model: Optional[str] = Form(default=None),
) -> Response:
    """Process files and return a downloadable export file.

    - Single file  → downloads as single .txt or .json
    - Multiple files → downloads as .zip containing one file per input
    - Format options: txt, json
    """
    from ocr_pipeline.config import settings

    _validate_files(files, settings.max_upload_size)
    model_key = _validate_model_key(model)

    pipeline = get_pipeline(model_key=model_key)
    exports: list[tuple[str, bytes]] = []   # (filename, bytes)

    for upload in files:
        data = await upload.read()
        original_name = upload.filename or "upload"
        try:
            ocr_result = pipeline.process_bytes(data, original_name)
        except Exception as e:
            logger.exception("Export OCR failed for '%s'", original_name)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

        file_bytes, _ = export_result(ocr_result, fmt, images=ocr_result.page_images)
        out_name = suggested_filename(original_name, fmt)
        exports.append((out_name, file_bytes))

    if len(exports) == 1:
        # Single file — return directly
        out_name, file_bytes = exports[0]
        if fmt == ExportFormat.TXT:
            media_type = "text/plain; charset=utf-8"
        elif fmt == ExportFormat.JSON:
            media_type = "application/json"
        else:
            media_type = "application/pdf"
        return Response(
            content=file_bytes,
            media_type=media_type,
            headers={"Content-Disposition": _content_disposition(out_name)},
        )
    else:
        # Multiple files — zip them
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, content in exports:
                zf.writestr(name, content)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="ocr_results.zip"'},
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_files(files: list[UploadFile], max_size: int) -> None:
    """Validate uploaded files: count, extension, and size."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No files uploaded.",
        )

    allowed = {".jpg", ".jpeg", ".png", ".pdf"}
    for upload in files:
        if not upload.filename:
            continue
        ext = "." + upload.filename.rsplit(".", 1)[-1].lower()
        if ext not in allowed:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}",
            )


def _content_disposition(filename: str) -> str:
    ascii_name = filename.encode("ascii", "ignore").decode("ascii")
    if not ascii_name:
        ascii_name = "download"
    return (
        f'attachment; filename="{ascii_name}"; '
        f"filename*=UTF-8''{quote(filename)}"
    )


def _validate_model_key(model: Optional[str]) -> Optional[str]:
    from ocr_pipeline.config import settings

    if not model:
        return None

    try:
        settings.get_model_config(model)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    return model


def _apply_postprocess(ocr_result: OCRResult, post_processor) -> OCRResult:
    pages = []
    for page in ocr_result.pages:
        lines = [line.text for line in page.lines]
        post_lines = post_processor.process_lines(lines)
        post_text = {k: "\n".join(v) for k, v in post_lines.items()}
        pages.append(
            page.model_copy(
                update={
                    "postprocessed": post_text,
                    "postprocessed_lines": post_lines,
                }
            )
        )
    return ocr_result.model_copy(update={"pages": pages})
